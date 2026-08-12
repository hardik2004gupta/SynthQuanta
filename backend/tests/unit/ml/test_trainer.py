"""Unit tests for the Trainer class.

Uses a tiny synthetic dataset so tests run in seconds on CPU.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.adapters.lora import LoRAConfig
from app.ml.adapters.qlora import QLoRAError
from app.ml.datasets.dataset import SensorWindowDataset
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer, TrainerError, TrainingResult, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_datasets():
    """16-window train, 4-window val — enough to run 1 epoch."""
    rng = np.random.default_rng(0)
    n_train, n_val, window_size = 16, 4, 32
    train_w = rng.random((n_train, window_size)).astype(np.float32)
    val_w = rng.random((n_val, window_size)).astype(np.float32)
    train_l = np.array([i % 7 for i in range(n_train)], dtype=np.int64)
    val_l = np.array([i % 7 for i in range(n_val)], dtype=np.int64)
    norm_stats = (float(train_w.mean()), float(train_w.std()) or 1.0)
    train_ds = SensorWindowDataset(train_w, train_l, norm_stats=norm_stats)
    val_ds = SensorWindowDataset(val_w, val_l, norm_stats=norm_stats)
    return train_ds, val_ds, norm_stats


@pytest.fixture
def tiny_cfg():
    return TrainingConfig(
        method="lora",
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        seed=42,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj"]),
        model=ModelConfig(window_size=32, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trainer_runs_and_returns_result(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=tmp_path / "ckpt",
        norm_stats=norm_stats,
    )
    result = trainer.train()
    assert isinstance(result, TrainingResult)
    assert result.total_epochs == 2
    assert len(result.training_history) == 2


def test_trainer_history_per_epoch(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=tmp_path / "ckpt",
        norm_stats=norm_stats,
    )
    result = trainer.train()
    for ep in result.training_history:
        assert ep.epoch >= 1
        assert isinstance(ep.train_loss, float)
        assert isinstance(ep.val_loss, float)
        assert 0.0 <= ep.val_accuracy <= 1.0


def test_trainer_checkpoint_written(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    ckpt_dir = tmp_path / "ckpt"
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=ckpt_dir,
        norm_stats=norm_stats,
    )
    trainer.train()
    assert (ckpt_dir / "base" / "model.pt").exists()
    assert (ckpt_dir / "metadata.json").exists()


def test_trainer_adapter_checkpoint_for_lora(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    ckpt_dir = tmp_path / "ckpt"
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=ckpt_dir,
        norm_stats=norm_stats,
    )
    result = trainer.train()
    assert result.adapter_checkpoint_path is not None
    assert result.adapter_checkpoint_path.exists()


def test_trainer_metadata_json_content(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    ckpt_dir = tmp_path / "ckpt"
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=ckpt_dir,
        norm_stats=norm_stats,
    )
    trainer.train()
    meta = json.loads((ckpt_dir / "metadata.json").read_text())
    assert "config" in meta
    assert "norm_stats" in meta
    assert "training_history" in meta
    assert len(meta["training_history"]) == 2


def test_trainer_progress_callback_called(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    calls = []
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=tmp_path / "ckpt",
        norm_stats=norm_stats,
        progress_cb=lambda ep: calls.append(ep.epoch),
    )
    result = trainer.train()
    assert calls == list(range(1, result.total_epochs + 1))


def test_trainer_qlora_fails_on_cpu(tiny_datasets, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    qlora_cfg = TrainingConfig(
        method="qlora", epochs=1, batch_size=4,
        model=ModelConfig(window_size=32, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16),
    )
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=qlora_cfg, checkpoint_dir=tmp_path / "ckpt",
        norm_stats=norm_stats,
    )
    with pytest.raises(TrainerError, match="QLoRA"):
        trainer.train()


def test_trainer_full_method(tiny_datasets, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    full_cfg = TrainingConfig(
        method="full", epochs=1, batch_size=4,
        model=ModelConfig(window_size=32, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16),
    )
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=full_cfg, checkpoint_dir=tmp_path / "ckpt",
        norm_stats=norm_stats,
    )
    result = trainer.train()
    assert result.adapter_checkpoint_path is None  # no lora file for full training
    assert result.trainable_parameters == result.total_parameters


def test_load_model_from_checkpoint(tiny_datasets, tiny_cfg, tmp_path):
    train_ds, val_ds, norm_stats = tiny_datasets
    ckpt_dir = tmp_path / "ckpt"
    trainer = Trainer(
        train_dataset=train_ds, val_dataset=val_ds,
        config=tiny_cfg, checkpoint_dir=ckpt_dir,
        norm_stats=norm_stats,
    )
    trainer.train()
    model, meta = load_model_from_checkpoint(ckpt_dir)
    assert model is not None
    x = torch.randn(1, 32, 1)
    out = model(x)
    assert out.logits.shape == (1, 7)
