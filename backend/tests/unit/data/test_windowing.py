"""Unit tests for WindowingEngine."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.models import FAULT_LABEL, FaultType, RawDataset, SignalType
from app.data.windowing import WindowingEngine


def _dataset(n: int = 1000, inject_fault: bool = False) -> RawDataset:
    t = np.arange(n, dtype=np.float64) * 0.01
    v = np.sin(2 * np.pi * 5.0 * t)
    labels = np.zeros(n, dtype=np.int32)
    if inject_fault:
        labels[300:500] = FAULT_LABEL[FaultType.NOISE]
    return RawDataset(
        timestamps=t,
        values=v,
        original_values=v.copy(),
        fault_annotations=[],
        labels=labels,
        signal_type=SignalType.SINUSOIDAL,
        seed=42,
        configuration={},
    )


class TestWindowingEngine:
    def test_window_count(self) -> None:
        ds = _dataset(1000)
        result = WindowingEngine(window_size=100, stride=100).build(ds)
        assert result.total_windows == 10

    def test_window_size_preserved(self) -> None:
        ds = _dataset(1000)
        result = WindowingEngine(window_size=128).build(ds)
        for w in result.windows:
            assert len(w.values) == 128
            assert len(w.timestamps) == 128

    def test_window_ids_sequential(self) -> None:
        ds = _dataset(500)
        result = WindowingEngine(window_size=100).build(ds)
        for i, w in enumerate(result.windows):
            assert w.window_id == i

    def test_normal_window_label(self) -> None:
        ds = _dataset(1000, inject_fault=False)
        result = WindowingEngine(window_size=100).build(ds)
        # All windows should be NORMAL
        for w in result.windows:
            assert w.label == FAULT_LABEL[FaultType.NORMAL]
            assert not w.has_fault

    def test_fault_window_labeled(self) -> None:
        ds = _dataset(1000, inject_fault=True)
        result = WindowingEngine(window_size=100).build(ds)
        # Windows 3-4 (indices 300-499) should have NOISE label
        fault_windows = [w for w in result.windows if w.has_fault]
        assert len(fault_windows) >= 1
        for w in fault_windows:
            assert FaultType.NOISE in w.fault_types

    def test_split_proportions(self) -> None:
        ds = _dataset(2000)
        result = WindowingEngine(window_size=100).build(ds)
        total = result.total_windows
        train_n = len(result.splits.train)
        val_n = len(result.splits.validation)
        iid_n = len(result.splits.iid_test)
        shift_n = len(result.splits.shift_test)
        assert train_n + val_n + iid_n + shift_n == total
        # Train should be ~70%
        assert train_n / total >= 0.60
        # Shift test is last 10%
        assert shift_n >= 1

    def test_splits_non_overlapping(self) -> None:
        ds = _dataset(5000)
        result = WindowingEngine(window_size=100).build(ds)
        all_indices = (
            result.splits.train
            + result.splits.validation
            + result.splits.iid_test
            + result.splits.shift_test
        )
        assert len(all_indices) == len(set(all_indices))

    def test_deterministic(self) -> None:
        ds = _dataset(1000)
        r1 = WindowingEngine(window_size=128).build(ds)
        r2 = WindowingEngine(window_size=128).build(ds)
        assert len(r1.windows) == len(r2.windows)
        for w1, w2 in zip(r1.windows, r2.windows):
            np.testing.assert_array_equal(w1.values, w2.values)
            np.testing.assert_array_equal(w1.timestamps, w2.timestamps)

    def test_empty_dataset(self) -> None:
        ds = _dataset(50)
        result = WindowingEngine(window_size=100).build(ds)  # window > dataset
        assert result.total_windows == 0

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError):
            WindowingEngine(window_size=0)

    def test_properties(self) -> None:
        ds = _dataset(2000, inject_fault=True)
        result = WindowingEngine(window_size=100).build(ds)
        assert isinstance(result.train_windows, list)
        assert isinstance(result.val_windows, list)
        assert isinstance(result.iid_test_windows, list)
        assert isinstance(result.shift_test_windows, list)
