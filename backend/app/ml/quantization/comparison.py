"""QuantizationComparison — FP32 vs INT8 quality and performance comparison.

Compares:
  - Classification quality: F1 macro on the IID test split
  - Model size: file size ratio (FP32 vs INT8 state dict)
  - Inference latency: mean single-window inference time

Rules (§11 CLAUDE.md; Phase 5 spec):
  - All numbers from actual execution — no hardcoded results
  - Uses evaluate_dataset() from EvaluationEngine (same function used in Phase 4)
  - Norm_stats come from the FP32 checkpoint — no leakage
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.ml.datasets.dataset import DatasetFactory
from app.ml.evaluation.engine import evaluate_dataset
from app.ml.evaluation.metrics import ClassificationMetrics
from app.ml.training.trainer import load_model_from_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """FP32 vs INT8 comparison metrics — all from actual measurements."""

    fp32_macro_f1: float
    int8_macro_f1: float
    f1_delta: float               # fp32 - int8 (positive = int8 degraded, negative = improved)
    fp32_size_bytes: int
    int8_size_bytes: int
    size_reduction_ratio: float   # fp32_size / int8_size (>1 means int8 is smaller)
    fp32_latency_ms: float        # mean single-window latency
    int8_latency_ms: float
    latency_speedup: float        # fp32_latency / int8_latency (>1 means int8 faster)
    n_test_windows: int

    def to_dict(self) -> dict:
        return {
            "fp32_macro_f1": round(self.fp32_macro_f1, 6),
            "int8_macro_f1": round(self.int8_macro_f1, 6),
            "f1_delta": round(self.f1_delta, 6),
            "fp32_size_bytes": self.fp32_size_bytes,
            "int8_size_bytes": self.int8_size_bytes,
            "size_reduction_ratio": round(self.size_reduction_ratio, 4),
            "fp32_latency_ms": round(self.fp32_latency_ms, 4),
            "int8_latency_ms": round(self.int8_latency_ms, 4),
            "latency_speedup": round(self.latency_speedup, 4),
            "n_test_windows": self.n_test_windows,
        }


def _latency_benchmark_ms(
    model: nn.Module,
    window_size: int,
    n_iter: int = 50,
    warmup: int = 10,
) -> float:
    """Return mean inference latency in milliseconds for single-window inputs."""
    x = torch.randn(1, window_size, 1)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times))


def run_comparison(
    fp32_checkpoint_dir: Path,
    int8_model: nn.Module,
    int8_model_file: Path,
    dataset_dir: Path,
    evaluation_batch_size: int = 64,
    benchmark_iterations: int = 50,
    benchmark_warmup: int = 10,
) -> ComparisonResult:
    """Compare FP32 and INT8 models on the IID test split.

    Args:
        fp32_checkpoint_dir:  Directory with base/model.pt + metadata.json
        int8_model:           Already-loaded quantized model (from QuantizationEngine)
        int8_model_file:      Path to the INT8 state dict file (for size measurement)
        dataset_dir:          Dataset artifact directory (for IID test split)
        evaluation_batch_size: Batch size for classification metric computation
        benchmark_iterations: Number of timed forward passes per model
        benchmark_warmup:     Warmup passes excluded from timing stats
    """
    fp32_checkpoint_dir = Path(fp32_checkpoint_dir)
    dataset_dir = Path(dataset_dir)

    # --- Load FP32 model ---
    fp32_model, ckpt_meta = load_model_from_checkpoint(fp32_checkpoint_dir, device="cpu")
    fp32_model.eval()

    window_size: int = ckpt_meta["config"]["model"]["window_size"]
    norm_stats_raw = ckpt_meta.get("norm_stats", {})
    norm_stats = (
        float(norm_stats_raw.get("mean", 0.0)),
        float(norm_stats_raw.get("std", 1.0)),
    )

    # --- Load IID test split (same split used in Phase 4 IID eval) ---
    splits = DatasetFactory.from_artifact(dataset_dir, window_size=window_size)
    iid_test = splits.iid_test
    n_windows = len(iid_test)

    if n_windows == 0:
        raise ValueError("IID test split is empty — cannot run comparison")

    # Override norm_stats with FP32 training norm_stats (no leakage)
    iid_test.mean = norm_stats[0]
    iid_test.std = norm_stats[1]

    # --- FP32 quality ---
    fp32_metrics, _, _, _ = evaluate_dataset(
        fp32_model, iid_test, batch_size=evaluation_batch_size
    )
    fp32_f1 = fp32_metrics.macro_f1

    # --- INT8 quality (same test split, same norm_stats) ---
    int8_model.eval()
    int8_metrics, _, _, _ = evaluate_dataset(
        int8_model, iid_test, batch_size=evaluation_batch_size
    )
    int8_f1 = int8_metrics.macro_f1

    # --- File sizes ---
    fp32_file = fp32_checkpoint_dir / "base" / "model.pt"
    fp32_size = int(fp32_file.stat().st_size) if fp32_file.exists() else 0
    int8_size = int(int8_model_file.stat().st_size) if int8_model_file.exists() else 0
    size_ratio = fp32_size / max(int8_size, 1)

    # --- Latency benchmark ---
    fp32_lat = _latency_benchmark_ms(
        fp32_model, window_size, benchmark_iterations, benchmark_warmup
    )
    int8_lat = _latency_benchmark_ms(
        int8_model, window_size, benchmark_iterations, benchmark_warmup
    )
    lat_speedup = fp32_lat / max(int8_lat, 1e-9)

    logger.info(
        "Comparison: FP32 F1=%.4f | INT8 F1=%.4f | size ratio=%.2fx | latency speedup=%.2fx",
        fp32_f1, int8_f1, size_ratio, lat_speedup,
    )

    return ComparisonResult(
        fp32_macro_f1=fp32_f1,
        int8_macro_f1=int8_f1,
        f1_delta=fp32_f1 - int8_f1,
        fp32_size_bytes=fp32_size,
        int8_size_bytes=int8_size,
        size_reduction_ratio=size_ratio,
        fp32_latency_ms=fp32_lat,
        int8_latency_ms=int8_lat,
        latency_speedup=lat_speedup,
        n_test_windows=n_windows,
    )
