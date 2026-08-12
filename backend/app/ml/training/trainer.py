"""Trainer — the core training loop for SynthQuanta Phase 3.

Responsibilities:
  - Accept DataLoaders + model + config
  - Optionally apply LoRA or detect/fail QLoRA
  - Run training loop with per-epoch validation
  - Persist checkpoint to artifact directory
  - Return TrainingResult with all metrics, configuration, and lineage

Architecture contract (§3, §16 of phase spec):
  TrainingService
      ↓
  Trainer          ← this module
      ↓
  DataLoaders, Model, Checkpoint

The Trainer must be instantiatable and callable directly in tests WITHOUT
going through FastAPI routes or the TrainingService thread.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from app.ml.adapters.lora import LoRAConfig, apply_lora, count_lora_parameters
from app.ml.adapters.qlora import QLoRAError, check_qlora_capability, qlora_diagnostic
from app.ml.datasets.dataset import SensorWindowDataset
from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer
from app.ml.training.config import TrainingConfig
from app.ml.training.reproducibility import collect_hardware_info, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    duration_seconds: float
    learning_rate: float


@dataclass
class TrainingResult:
    training_history: list[EpochMetrics]
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    duration_seconds: float
    checkpoint_dir: Path
    model_checkpoint_path: Path    # the .pt file for the full model state
    adapter_checkpoint_path: Path | None  # LoRA-only weights if applicable
    norm_stats: tuple[float, float]
    total_parameters: int
    trainable_parameters: int
    hardware_info: dict
    config: TrainingConfig
    error: str | None = None       # set if training failed partway through


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TrainerError(Exception):
    """Raised when training fails for a recoverable or deterministic reason."""


class Trainer:
    """Executes a complete training run.

    Args:
        train_dataset:   Training split (already normalized).
        val_dataset:     Validation split.
        config:          Full training configuration.
        checkpoint_dir:  Where to write the model checkpoint.
        norm_stats:      (mean, std) used to normalize the datasets — stored
                         with the checkpoint for inference.
        progress_cb:     Optional callback called after each epoch with the
                         EpochMetrics dict.  Used by TrainingService to write
                         incremental training_history to the database.
    """

    def __init__(
        self,
        train_dataset: SensorWindowDataset,
        val_dataset: SensorWindowDataset,
        config: TrainingConfig,
        checkpoint_dir: Path,
        norm_stats: tuple[float, float],
        progress_cb: Callable[[EpochMetrics], None] | None = None,
    ) -> None:
        config.validate()
        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._config = config
        self._checkpoint_dir = Path(checkpoint_dir)
        self._norm_stats = norm_stats
        self._progress_cb = progress_cb
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run the full training loop and return results."""
        t_start = time.perf_counter()
        cfg = self._config

        # 1. QLoRA capability check (fail early, never silently downgrade)
        if cfg.method == "qlora":
            try:
                check_qlora_capability()
            except QLoRAError as exc:
                raise TrainerError(
                    f"QLoRA is not supported in this environment: {exc}"
                ) from exc

        # 2. Seed
        set_seed(cfg.seed)
        hardware_info = collect_hardware_info()
        logger.info(
            "Training on device=%s | method=%s | epochs=%d | batch=%d",
            hardware_info["device"],
            cfg.method,
            cfg.epochs,
            cfg.batch_size,
        )

        # 3. Build model
        model = SensorTransformer(cfg.model).to(self._device)

        # 4. Apply adaptation (LoRA / QLoRA / full)
        adapter_checkpoint_path: Path | None = None
        if cfg.method in ("lora", "qlora"):
            apply_lora(model, cfg.lora)
            total_p, trainable_p = count_lora_parameters(model)
        else:
            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = total_p

        logger.info(
            "Parameters: total=%d, trainable=%d (%.2f%%)",
            total_p,
            trainable_p,
            100.0 * trainable_p / max(total_p, 1),
        )

        # 5. DataLoaders
        train_loader = DataLoader(
            self._train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            self._val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        if len(train_loader) == 0:
            raise TrainerError("Training dataset is empty — cannot train.")

        # 6. Loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # 7. Training loop
        history: list[EpochMetrics] = []
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, cfg.epochs + 1):
            ep_start = time.perf_counter()

            train_loss = self._train_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, val_acc = self._val_epoch(model, val_loader, loss_fn)

            ep_dur = time.perf_counter() - ep_start
            current_lr = optimizer.param_groups[0]["lr"]

            ep_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                duration_seconds=ep_dur,
                learning_rate=current_lr,
            )
            history.append(ep_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | %.1fs",
                epoch, cfg.epochs, train_loss, val_loss, val_acc, ep_dur,
            )

            if self._progress_cb is not None:
                try:
                    self._progress_cb(ep_metrics)
                except Exception:
                    logger.warning("progress_cb raised — ignoring", exc_info=True)

        # 8. Checkpoint
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        base_dir = self._checkpoint_dir / "base"
        base_dir.mkdir(exist_ok=True)
        model_ckpt_path = base_dir / "model.pt"
        torch.save(model.state_dict(), model_ckpt_path)

        if cfg.method in ("lora", "qlora"):
            adapter_dir = self._checkpoint_dir / "adapter"
            adapter_dir.mkdir(exist_ok=True)
            adapter_ckpt_path = adapter_dir / "lora.pt"
            lora_state = {
                k: v for k, v in model.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            }
            torch.save(lora_state, adapter_ckpt_path)
            adapter_checkpoint_path = adapter_ckpt_path

        # 9. Write metadata.json alongside the checkpoint
        meta = {
            "config": cfg.to_dict(),
            "norm_stats": {"mean": self._norm_stats[0], "std": self._norm_stats[1]},
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "hardware_info": hardware_info,
            "training_history": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "val_accuracy": m.val_accuracy,
                    "duration_seconds": m.duration_seconds,
                    "learning_rate": m.learning_rate,
                }
                for m in history
            ],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
        (self._checkpoint_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        total_dur = time.perf_counter() - t_start
        logger.info("Training complete in %.1f s | best_val_loss=%.4f", total_dur, best_val_loss)

        return TrainingResult(
            training_history=history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_epochs=cfg.epochs,
            duration_seconds=total_dur,
            checkpoint_dir=self._checkpoint_dir,
            model_checkpoint_path=model_ckpt_path,
            adapter_checkpoint_path=adapter_checkpoint_path,
            norm_stats=self._norm_stats,
            total_parameters=total_p,
            trainable_parameters=trainable_p,
            hardware_info=hardware_info,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: AdamW,
    ) -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output.logits, y)
            loss.backward()
            # Gradient clipping to stabilise small-model training
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            output = model(x)
            loss = loss_fn(output.logits, y)
            total_loss += loss.item()
            pred = output.fault_class
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint reload
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_dir: Path, device: str = "cpu") -> tuple[SensorTransformer, dict]:
    """Load a SensorTransformer from a previously saved checkpoint directory.

    Returns (model, metadata) where metadata is the JSON dict written at
    training time.

    Raises FileNotFoundError if the checkpoint is not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "metadata.json"
    model_path = checkpoint_dir / "base" / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg = ModelConfig.from_dict(meta["config"]["model"])
    model = SensorTransformer(model_cfg)
    state = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)   # strict=False: buffers OK
    model.eval()
    return model, meta
