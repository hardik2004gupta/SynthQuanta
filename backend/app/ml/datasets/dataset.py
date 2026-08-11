"""PyTorch Dataset and factory for loading Phase 2 NPZ artifacts.

Responsibilities:
- Load timestamps/values/labels from NPZ artifact
- Re-apply the deterministic 70/10/10/10 temporal split
- Build SensorWindowDataset for each split
- Compute per-window labels (majority non-NORMAL fault)
- Apply z-score normalization (using training-split statistics only)
- Handle NaN (DROPOUT samples) → replace with 0.0

This module must NOT regenerate synthetic data.  It only reads persisted
Phase 2 artifacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "validation", "iid_test", "shift_test"]


# ---------------------------------------------------------------------------
# Window label helper (mirrors WindowingEngine logic, no circular import)
# ---------------------------------------------------------------------------

def _majority_label(window_labels: np.ndarray) -> int:
    """Return dominant non-NORMAL label, or 0 (NORMAL) if no faults present."""
    NORMAL = 0
    fault_mask = window_labels != NORMAL
    if not np.any(fault_mask):
        return NORMAL
    fault_lbls = window_labels[fault_mask]
    unique, counts = np.unique(fault_lbls, return_counts=True)
    return int(unique[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Temporal split helper (matches WindowingEngine._split exactly)
# ---------------------------------------------------------------------------

def _temporal_splits(total_windows: int) -> dict[SplitName, list[int]]:
    """Re-derive 70/10/10/10 split from window count."""
    if total_windows == 0:
        return {"train": [], "validation": [], "iid_test": [], "shift_test": []}
    shift_start = max(0, int(total_windows * 0.90))
    main = shift_start
    val_start = max(0, int(main * (70 / 90)))
    iid_start = max(0, int(main * (80 / 90)))
    return {
        "train": list(range(0, val_start)),
        "validation": list(range(val_start, iid_start)),
        "iid_test": list(range(iid_start, shift_start)),
        "shift_test": list(range(shift_start, total_windows)),
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SensorWindowDataset(Dataset):
    """Fixed-size windows from a persisted NPZ artifact, ready for PyTorch.

    Args:
        windows:    (n_windows, window_size) float32 — pre-built window array
        labels:     (n_windows,) int64
        norm_stats: (mean, std) to apply z-score normalization.  If None,
                    normalization is computed from this split (use only for
                    the training split; pass the computed stats to others).
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        norm_stats: tuple[float, float] | None = None,
    ) -> None:
        # Replace NaN (DROPOUT) with 0.0 before normalization
        windows = windows.astype(np.float32)
        nan_mask = np.isnan(windows)
        windows[nan_mask] = 0.0

        if norm_stats is None:
            mean = float(windows.mean())
            std = float(windows.std()) + 1e-8
        else:
            mean, std = norm_stats

        self._windows = ((windows - mean) / std).astype(np.float32)
        self._labels = labels.astype(np.int64)
        self.norm_stats: tuple[float, float] = (mean, std)
        self.window_size = windows.shape[1] if windows.ndim == 2 else 0

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._windows[idx]).unsqueeze(-1)  # (window_size, 1)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Factory result
# ---------------------------------------------------------------------------

@dataclass
class SplitDatasets:
    train: SensorWindowDataset
    validation: SensorWindowDataset
    iid_test: SensorWindowDataset
    shift_test: SensorWindowDataset
    window_size: int
    norm_stats: tuple[float, float]  # from training split
    split_counts: dict[SplitName, int]
    artifact_metadata: dict


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class DatasetFactory:
    """Load a persisted Phase 2 dataset artifact into PyTorch Datasets."""

    @classmethod
    def from_artifact(cls, artifact_dir: Path, window_size: int) -> SplitDatasets:
        """
        Args:
            artifact_dir: Path to the dataset directory (contains data.npz + metadata.json)
            window_size:  Window size used during dataset generation (must match)

        Returns:
            SplitDatasets with train/validation/iid_test/shift_test splits
        """
        artifact_dir = Path(artifact_dir)
        npz_path = artifact_dir / "data.npz"
        meta_path = artifact_dir / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset artifact not found: {npz_path}")

        npz = np.load(str(npz_path), allow_pickle=False)
        values: np.ndarray = npz["values"].astype(np.float32)   # (n_samples,)
        labels: np.ndarray = npz["labels"].astype(np.int64)     # (n_samples,)

        metadata: dict = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        # Build windows (non-overlapping, stride = window_size)
        stride = window_size
        n_samples = len(values)
        n_windows = (n_samples - window_size) // stride + 1 if n_samples >= window_size else 0

        if n_windows == 0:
            raise ValueError(
                f"Dataset has {n_samples} samples but window_size={window_size}. "
                "Not enough samples for even one window."
            )

        win_values = np.stack(
            [values[i * stride : i * stride + window_size] for i in range(n_windows)]
        )  # (n_windows, window_size)

        win_labels = np.array(
            [
                _majority_label(labels[i * stride : i * stride + window_size])
                for i in range(n_windows)
            ],
            dtype=np.int64,
        )  # (n_windows,)

        splits = _temporal_splits(n_windows)

        # Build training dataset first to compute normalization stats
        def _subset(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if not indices:
                return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
            idx = np.array(indices)
            return win_values[idx], win_labels[idx]

        train_w, train_l = _subset(splits["train"])
        train_ds = SensorWindowDataset(train_w, train_l, norm_stats=None)
        norm_stats = train_ds.norm_stats  # derived from training split only

        val_w, val_l = _subset(splits["validation"])
        iid_w, iid_l = _subset(splits["iid_test"])
        shift_w, shift_l = _subset(splits["shift_test"])

        return SplitDatasets(
            train=train_ds,
            validation=SensorWindowDataset(val_w, val_l, norm_stats=norm_stats),
            iid_test=SensorWindowDataset(iid_w, iid_l, norm_stats=norm_stats),
            shift_test=SensorWindowDataset(shift_w, shift_l, norm_stats=norm_stats),
            window_size=window_size,
            norm_stats=norm_stats,
            split_counts={
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "iid_test": len(splits["iid_test"]),
                "shift_test": len(splits["shift_test"]),
            },
            artifact_metadata=metadata,
        )

    @staticmethod
    def make_loader(
        dataset: SensorWindowDataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,   # 0 = main process (safe on Windows + all platforms)
            pin_memory=False,
            drop_last=False,
        )
