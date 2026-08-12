"""End-to-end training test.

Runs the complete Phase 3 lifecycle without any mocking:
  1. Generate a tiny synthetic dataset
  2. Run LoRA training (1 epoch, tiny model)
  3. Verify checkpoint written
  4. Verify TrainingResult contains real metrics

This test uses DatasetFactory + Trainer directly (not through FastAPI) so
it runs in a few seconds even on CPU and does not require a live DB.

Acceptance criteria (§21 of CLAUDE.md):
  - Training loop runs end-to-end
  - Checkpoint persisted with correct directory structure
  - TrainingResult contains non-trivial metrics (non-NaN losses)
  - LoRA adapter file exists
  - Checkpoint can be reloaded
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory, SensorWindowDataset
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer, TrainingResult, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Tiny synthetic artifact (write directly, no DB needed)
# ---------------------------------------------------------------------------

def _create_tiny_artifact(artifact_dir: Path, n_samples: int = 512, window_size: int = 32) -> Path:
    rng = np.random.default_rng(42)
    timestamps = np.linspace(0.0, 5.0, n_samples)
    values = np.sin(2 * np.pi * 10 * timestamps) + rng.normal(0, 0.05, n_samples)
    labels = np.zeros(n_samples, dtype=np.int64)
    # Inject several fault classes so the model sees all 7
    segment = n_samples // 7
    for fault_label in range(1, 7):
        start = fault_label * segment
        end = start + segment // 2
        labels[start:end] = fault_label

    ds_dir = artifact_dir / "DS-E2E"
    ds_dir.mkdir(parents=True)
    np.savez(str(ds_dir / "data.npz"), timestamps=timestamps, values=values, labels=labels)
    meta = {
        "human_id": "DS-E2E",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 42,
        "configuration": {},
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta))
    return ds_dir


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_result(tmp_path_factory):
    """Run the full tiny training pipeline once; share across tests."""
    tmp = tmp_path_factory.mktemp("e2e")
    ds_dir = _create_tiny_artifact(tmp, n_samples=512, window_size=32)
    ckpt_dir = tmp / "MODEL-E2E"

    cfg = TrainingConfig(
        method="lora",
        epochs=2,
        batch_size=8,
        learning_rate=5e-4,
        seed=0,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        model=ModelConfig(window_size=32, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16),
    )

    splits = DatasetFactory.from_artifact(ds_dir, window_size=32)
    trainer = Trainer(
        train_dataset=splits.train,
        val_dataset=splits.validation,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        norm_stats=splits.norm_stats,
    )
    result = trainer.train()
    return result, ckpt_dir


def test_e2e_training_completes(e2e_result):
    result, _ = e2e_result
    assert isinstance(result, TrainingResult)


def test_e2e_epochs_executed(e2e_result):
    result, _ = e2e_result
    assert result.total_epochs == 2
    assert len(result.training_history) == 2


def test_e2e_losses_are_finite(e2e_result):
    result, _ = e2e_result
    for ep in result.training_history:
        assert not (ep.train_loss != ep.train_loss), f"train_loss is NaN at epoch {ep.epoch}"
        assert not (ep.val_loss != ep.val_loss), f"val_loss is NaN at epoch {ep.epoch}"
        assert ep.train_loss > 0
        assert ep.val_loss > 0


def test_e2e_accuracy_in_range(e2e_result):
    result, _ = e2e_result
    for ep in result.training_history:
        assert 0.0 <= ep.val_accuracy <= 1.0


def test_e2e_checkpoint_exists(e2e_result):
    _, ckpt_dir = e2e_result
    assert (ckpt_dir / "base" / "model.pt").exists()
    assert (ckpt_dir / "metadata.json").exists()


def test_e2e_adapter_checkpoint_exists(e2e_result):
    result, _ = e2e_result
    assert result.adapter_checkpoint_path is not None
    assert result.adapter_checkpoint_path.exists()


def test_e2e_lora_keys_only_in_adapter(e2e_result):
    result, ckpt_dir = e2e_result
    lora_state = torch.load(str(result.adapter_checkpoint_path), map_location="cpu", weights_only=False)
    assert len(lora_state) > 0
    for key in lora_state:
        assert "lora_A" in key or "lora_B" in key, f"Unexpected key in LoRA adapter: {key}"


def test_e2e_checkpoint_reloads(e2e_result):
    _, ckpt_dir = e2e_result
    model, meta = load_model_from_checkpoint(ckpt_dir)
    assert model is not None
    x = torch.randn(1, 32, 1)
    out = model(x)
    assert out.logits.shape == (1, 7)
    assert not torch.isnan(out.logits).any()


def test_e2e_metadata_json_complete(e2e_result):
    _, ckpt_dir = e2e_result
    meta = json.loads((ckpt_dir / "metadata.json").read_text())
    assert "config" in meta
    assert "norm_stats" in meta
    assert "training_history" in meta
    assert "total_parameters" in meta
    assert "trainable_parameters" in meta
    assert meta["trainable_parameters"] < meta["total_parameters"]


def test_e2e_parameter_counts(e2e_result):
    result, _ = e2e_result
    assert result.trainable_parameters > 0
    assert result.total_parameters > result.trainable_parameters


def test_e2e_hardware_info_captured(e2e_result):
    result, _ = e2e_result
    assert "device" in result.hardware_info
    assert "torch_version" in result.hardware_info


def test_e2e_reproducible(tmp_path_factory):
    """Same seed → same training history (loss values)."""
    tmp1 = tmp_path_factory.mktemp("repro1")
    tmp2 = tmp_path_factory.mktemp("repro2")
    ds_dir1 = _create_tiny_artifact(tmp1, n_samples=256, window_size=32)
    ds_dir2 = _create_tiny_artifact(tmp2, n_samples=256, window_size=32)

    cfg = TrainingConfig(
        method="lora", epochs=1, batch_size=4, learning_rate=1e-3, seed=99,
        lora=LoRAConfig(rank=2, target_modules=["q_proj"]),
        model=ModelConfig(window_size=32, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16),
    )

    splits1 = DatasetFactory.from_artifact(ds_dir1, window_size=32)
    splits2 = DatasetFactory.from_artifact(ds_dir2, window_size=32)

    r1 = Trainer(splits1.train, splits1.validation, cfg, tmp1 / "ckpt", splits1.norm_stats).train()
    r2 = Trainer(splits2.train, splits2.validation, cfg, tmp2 / "ckpt", splits2.norm_stats).train()

    # Both runs started from the same seed; loss should be identical
    ep1 = r1.training_history[0]
    ep2 = r2.training_history[0]
    assert abs(ep1.train_loss - ep2.train_loss) < 1e-4, (
        f"Reproducibility failure: {ep1.train_loss} vs {ep2.train_loss}"
    )
