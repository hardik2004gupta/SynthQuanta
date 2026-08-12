"""EvaluationEngine — loads a trained model and computes all evaluation metrics.

Architecture (§35 of phase spec):
    EvaluationEngine
        ↓
    load checkpoint (reusing Phase 3 loader)
        ↓
    load IID test split
        ↓
    batch inference (no_grad, eval mode)
        ↓
    collect predictions + ground truth
        ↓
    compute classification metrics
        ↓
    compute localization
        ↓
    construct EvaluationResult

This module must be independently testable without FastAPI.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.ml.datasets.dataset import DatasetFactory, SensorWindowDataset
from app.ml.evaluation.localization import LocalizationResult, compute_localization
from app.ml.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from app.ml.models.sensor_transformer import SensorTransformer
from app.ml.training.reproducibility import collect_hardware_info
from app.ml.training.trainer import load_model_from_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result from the IID test split."""
    status: str                     # "COMPLETED" | "FAILED"
    metrics: ClassificationMetrics | None
    localization: LocalizationResult | None
    n_windows: int
    duration_seconds: float
    hardware_info: dict[str, Any]
    checkpoint_dir: Path
    dataset_dir: Path
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "n_windows": self.n_windows,
            "duration_seconds": self.duration_seconds,
            "hardware_info": self.hardware_info,
            "error": self.error,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "localization": self.localization.to_dict() if self.localization else None,
        }


class EvaluationEngine:
    """Runs a trained SensorTransformer against the IID test split.

    Args:
        checkpoint_dir: Directory containing base/model.pt + metadata.json
        dataset_dir:    Directory containing data.npz + metadata.json
        batch_size:     Inference batch size (default 64)
        device:         PyTorch device string (default "cpu")
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        dataset_dir: Path,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._dataset_dir = Path(dataset_dir)
        self._batch_size = batch_size
        self._device = torch.device(device)

    def evaluate(self) -> EvaluationResult:
        """Run full IID evaluation.  Returns EvaluationResult (never raises)."""
        t0 = time.perf_counter()
        hw = collect_hardware_info()

        try:
            model, ckpt_meta = load_model_from_checkpoint(
                self._checkpoint_dir, device=str(self._device)
            )
            model.eval()

            window_size = ckpt_meta["config"]["model"]["window_size"]
            norm_stats_raw = ckpt_meta.get("norm_stats", {})
            norm_stats = (
                float(norm_stats_raw.get("mean", 0.0)),
                float(norm_stats_raw.get("std", 1.0)),
            )

            splits = DatasetFactory.from_artifact(self._dataset_dir, window_size=window_size)
            iid_ds = splits.iid_test

            if len(iid_ds) == 0:
                raise ValueError("IID test split is empty — cannot evaluate.")

            # Force evaluation dataset to use training norm_stats (no leakage)
            # (DatasetFactory already passes training norm_stats to all splits)
            logger.info(
                "Evaluating on %d IID test windows (checkpoint: %s)",
                len(iid_ds), self._checkpoint_dir.name,
            )

            y_true, y_pred = self._run_inference(model, iid_ds)
            metrics = compute_classification_metrics(y_true, y_pred)
            localization = compute_localization(y_true, y_pred)

            duration = time.perf_counter() - t0
            logger.info(
                "Evaluation complete: macro_f1=%.4f | weighted_f1=%.4f | %.1fs",
                metrics.macro_f1, metrics.weighted_f1, duration,
            )
            return EvaluationResult(
                status="COMPLETED",
                metrics=metrics,
                localization=localization,
                n_windows=len(iid_ds),
                duration_seconds=duration,
                hardware_info=hw,
                checkpoint_dir=self._checkpoint_dir,
                dataset_dir=self._dataset_dir,
            )

        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.exception("EvaluationEngine failed: %s", exc)
            return EvaluationResult(
                status="FAILED",
                metrics=None,
                localization=None,
                n_windows=0,
                duration_seconds=duration,
                hardware_info=hw,
                checkpoint_dir=self._checkpoint_dir,
                dataset_dir=self._dataset_dir,
                error=str(exc),
            )

    @torch.no_grad()
    def _run_inference(
        self,
        model: SensorTransformer,
        dataset: SensorWindowDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run batch inference and return (y_true, y_pred) numpy arrays."""
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []

        model.eval()
        for x, y in loader:
            x = x.to(self._device)
            output = model(x)
            all_true.append(y.numpy())
            all_pred.append(output.fault_class.cpu().numpy())

        return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Convenience: evaluate a pre-loaded dataset (used by shift engine)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model: SensorTransformer,
    dataset: SensorWindowDataset,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[ClassificationMetrics, LocalizationResult, np.ndarray, np.ndarray]:
    """Evaluate a model against an arbitrary SensorWindowDataset.

    Returns (metrics, localization, y_true, y_pred).
    The model must already be in eval() mode with no_grad context active.
    """
    if device is None:
        device = torch.device("cpu")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    model.eval()
    for x, y in loader:
        x = x.to(device)
        output = model(x)
        all_true.append(y.numpy())
        all_pred.append(output.fault_class.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

    metrics = compute_classification_metrics(y_true, y_pred)
    localization = compute_localization(y_true, y_pred)
    return metrics, localization, y_true, y_pred
