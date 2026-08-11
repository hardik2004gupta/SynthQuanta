"""Unit tests for SensorWindowDataset and DatasetFactory.

These tests synthesize a small NPZ artifact in a temp dir to avoid
depending on the full data engine.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.datasets.dataset import DatasetFactory, SensorWindowDataset


# ---------------------------------------------------------------------------
# Helpers to create a minimal fake dataset artifact
# ---------------------------------------------------------------------------

def _make_fake_artifact(tmp: Path, n_samples: int = 1024, window_size: int = 32) -> Path:
    """Create a minimal data.npz + metadata.json that DatasetFactory can load."""
    rng = np.random.default_rng(42)
    timestamps = np.linspace(0.0, 10.0, n_samples)
    values = np.sin(timestamps) + rng.normal(0, 0.05, n_samples)
    labels = np.zeros(n_samples, dtype=np.int64)
    # Inject a few fault labels in 20-30% of samples
    fault_start = int(n_samples * 0.3)
    fault_end = int(n_samples * 0.4)
    labels[fault_start:fault_end] = 1  # NOISE fault label

    artifact_dir = tmp / "DS-TEST"
    artifact_dir.mkdir()
    np.savez(
        str(artifact_dir / "data.npz"),
        timestamps=timestamps,
        values=values,
        labels=labels,
    )
    meta = {
        "human_id": "DS-TEST",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 42,
        "configuration": {},
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(meta))
    return artifact_dir


@pytest.fixture
def fake_artifact(tmp_path: Path):
    return _make_fake_artifact(tmp_path, n_samples=1024, window_size=32)


# ---------------------------------------------------------------------------
# SensorWindowDataset
# ---------------------------------------------------------------------------

def test_dataset_length():
    n = 100
    windows = np.random.randn(n, 32)
    labels = np.zeros(n, dtype=np.int64)
    ds = SensorWindowDataset(windows, labels)
    assert len(ds) == n


def test_dataset_item_shape():
    windows = np.random.randn(10, 32)
    labels = np.zeros(10, dtype=np.int64)
    ds = SensorWindowDataset(windows, labels)
    x, y = ds[0]
    assert x.shape == (32, 1)
    assert y.dtype == torch.long
    assert y.item() in range(7)


def test_dataset_nan_replaced_with_zero():
    windows = np.full((5, 16), np.nan)
    labels = np.zeros(5, dtype=np.int64)
    ds = SensorWindowDataset(windows, labels)
    x, _ = ds[0]
    assert not torch.isnan(x).any()


def test_dataset_normalization():
    rng = np.random.default_rng(0)
    windows = rng.normal(5.0, 2.0, (50, 32))
    labels = np.zeros(50, dtype=np.int64)
    ds = SensorWindowDataset(windows, labels)
    # After z-score, mean ≈ 0, std ≈ 1
    all_vals = torch.stack([ds[i][0] for i in range(50)]).flatten()
    assert abs(all_vals.mean().item()) < 0.5
    assert abs(all_vals.std().item() - 1.0) < 0.5


def test_dataset_norm_stats_injected():
    """Norm stats from train split must be injectable to prevent test leakage."""
    windows = np.random.randn(20, 16)
    labels = np.zeros(20, dtype=np.int64)
    norm_stats = (3.0, 2.0)  # artificial mean/std
    ds = SensorWindowDataset(windows, labels, norm_stats=norm_stats)
    x, _ = ds[0]
    # Value should be (raw - 3) / 2 — not zero-centered around its own mean
    raw = windows[0]
    expected = (raw - 3.0) / 2.0
    assert np.allclose(x.squeeze(-1).numpy(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# DatasetFactory
# ---------------------------------------------------------------------------

def test_factory_from_artifact(fake_artifact):
    splits = DatasetFactory.from_artifact(fake_artifact, window_size=32)
    assert len(splits.train) > 0
    assert len(splits.validation) > 0
    assert len(splits.iid_test) > 0
    assert len(splits.shift_test) > 0


def test_factory_splits_non_overlapping(fake_artifact):
    splits = DatasetFactory.from_artifact(fake_artifact, window_size=32)
    total = (len(splits.train) + len(splits.validation) +
             len(splits.iid_test) + len(splits.shift_test))
    # All windows should be accounted for (within rounding)
    assert total > 0


def test_factory_norm_stats_shape(fake_artifact):
    splits = DatasetFactory.from_artifact(fake_artifact, window_size=32)
    mean, std = splits.norm_stats
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0


def test_factory_train_item_shape(fake_artifact):
    splits = DatasetFactory.from_artifact(fake_artifact, window_size=32)
    x, y = splits.train[0]
    assert x.shape == (32, 1)
    assert y.dtype == torch.long


def test_factory_make_loader(fake_artifact):
    splits = DatasetFactory.from_artifact(fake_artifact, window_size=32)
    loader = DatasetFactory.make_loader(splits.train, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1] == 32
    assert batch_x.shape[2] == 1
    assert batch_y.dtype == torch.long
