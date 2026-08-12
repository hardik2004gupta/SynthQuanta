"""Deterministic fixed-size windowing and dataset splitting."""
from __future__ import annotations

import numpy as np

from app.data.models import (
    DatasetWindow,
    FaultType,
    FAULT_LABEL,
    LABEL_TO_FAULT,
    RawDataset,
    SplitIndices,
    WindowedDataset,
)


class WindowingEngine:
    """Converts a RawDataset into fixed-size, non-overlapping windows.

    Windowing is fully deterministic — no shuffling before construction.
    The label for each window is the most frequent non-NORMAL fault label
    in that window (majority vote). NORMAL wins only when no fault labels
    are present.

    Split strategy (70 / 10 / 10 / 10):
    - Windows are first split temporally into main (90%) and shift-test (10%).
    - The main portion is split 77.8% / 11.1% / 11.1% → 70/10/10 of total.
    - The shift-test portion is the final 10%.

    This approach preserves temporal order and leaves room for the Phase 4
    distribution-shift evaluation engine to generate genuinely shifted data
    for the shift-test split.
    """

    def __init__(self, window_size: int = 128, stride: int | None = None) -> None:
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size  # non-overlapping by default

    def build(self, dataset: RawDataset) -> WindowedDataset:
        n = len(dataset.timestamps)
        windows: list[DatasetWindow] = []

        idx = 0
        while idx + self.window_size <= n:
            end = idx + self.window_size
            window_labels = dataset.labels[idx:end]
            window_values = dataset.values[idx:end]
            window_ts = dataset.timestamps[idx:end]

            label, fault_types = self._classify_window(window_labels)
            windows.append(
                DatasetWindow(
                    window_id=len(windows),
                    timestamps=window_ts.copy(),
                    values=window_values.copy(),
                    label=label,
                    has_fault=label != FAULT_LABEL[FaultType.NORMAL],
                    fault_types=fault_types,
                )
            )
            idx += self.stride

        splits = self._split(len(windows))
        return WindowedDataset(
            windows=windows,
            window_size=self.window_size,
            stride=self.stride,
            splits=splits,
            total_windows=len(windows),
        )

    # ------------------------------------------------------------------

    def _classify_window(
        self, labels: np.ndarray
    ) -> tuple[int, list[FaultType]]:
        """Return the dominant label and list of distinct fault types in the window."""
        normal_label = FAULT_LABEL[FaultType.NORMAL]
        fault_mask = labels != normal_label
        if not np.any(fault_mask):
            return normal_label, []

        fault_labels = labels[fault_mask]
        unique, counts = np.unique(fault_labels, return_counts=True)
        dominant = int(unique[np.argmax(counts)])
        fault_types = [LABEL_TO_FAULT[int(u)] for u in unique]
        return dominant, fault_types

    def _split(self, total: int) -> SplitIndices:
        """Temporal split: 70/10/10/10.

        Indices are integer window indices (not sample indices).
        The shift_test block comes from the final 10% of temporal order.
        """
        if total == 0:
            return SplitIndices(train=[], validation=[], iid_test=[], shift_test=[])

        shift_start = max(0, int(total * 0.90))
        main_total = shift_start

        val_start = max(0, int(main_total * (70 / 90)))
        iid_start = max(0, int(main_total * (80 / 90)))

        train = list(range(0, val_start))
        validation = list(range(val_start, iid_start))
        iid_test = list(range(iid_start, shift_start))
        shift_test = list(range(shift_start, total))

        return SplitIndices(
            train=train,
            validation=validation,
            iid_test=iid_test,
            shift_test=shift_test,
        )
