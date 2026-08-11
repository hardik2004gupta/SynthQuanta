"""Domain dataclasses for the synthetic data engine.

These are pure Python domain objects — no SQLAlchemy, no Pydantic.
The service layer translates between these and the persistence/API layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class FaultType(StrEnum):
    NORMAL = "NORMAL"
    NOISE = "NOISE"
    DRIFT = "DRIFT"
    DROPOUT = "DROPOUT"
    CLIPPING = "CLIPPING"
    TIMESTAMP_GAP = "TIMESTAMP_GAP"
    SAMPLING_JITTER = "SAMPLING_JITTER"


# Integer label for each fault type (used in labels array)
FAULT_LABEL: dict[FaultType, int] = {
    FaultType.NORMAL: 0,
    FaultType.NOISE: 1,
    FaultType.DRIFT: 2,
    FaultType.DROPOUT: 3,
    FaultType.CLIPPING: 4,
    FaultType.TIMESTAMP_GAP: 5,
    FaultType.SAMPLING_JITTER: 6,
}

LABEL_TO_FAULT: dict[int, FaultType] = {v: k for k, v in FAULT_LABEL.items()}


class SignalType(StrEnum):
    SINUSOIDAL = "sinusoidal"
    COMPOSITE = "composite"
    TREND = "trend"
    PERIODIC = "periodic"


@dataclass
class FaultAnnotation:
    """Ground-truth record for a single injected fault event."""
    fault_id: str
    fault_type: FaultType
    start_index: int
    end_index: int
    severity: float            # normalised [0, 1]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "severity": self.severity,
            "parameters": self.parameters,
        }


@dataclass
class GeneratedSignal:
    """Output of a signal generator before fault injection."""
    timestamps: np.ndarray   # float64, seconds from t=0, shape (N,)
    values: np.ndarray       # float64, shape (N,)
    signal_type: SignalType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawDataset:
    """Complete generated dataset including fault-injected signal and ground truth."""
    timestamps: np.ndarray          # float64, shape (N,)
    values: np.ndarray              # float64, may contain NaN for DROPOUT
    original_values: np.ndarray     # float64, clean signal before any fault injection
    fault_annotations: list[FaultAnnotation]
    labels: np.ndarray              # int32, per-sample, shape (N,) — 0=NORMAL, 1=NOISE, …
    signal_type: SignalType
    seed: int
    configuration: dict[str, Any]   # full generation config for reproducibility


@dataclass
class DatasetWindow:
    """A single fixed-length window extracted from a RawDataset."""
    window_id: int
    timestamps: np.ndarray   # shape (window_size,)
    values: np.ndarray       # shape (window_size,)
    label: int               # dominant fault label (0 = NORMAL)
    has_fault: bool
    fault_types: list[FaultType]


@dataclass
class SplitIndices:
    """Window index ranges for each dataset split."""
    train: list[int]
    validation: list[int]
    iid_test: list[int]
    shift_test: list[int]


@dataclass
class WindowedDataset:
    """Windows extracted from a RawDataset, with split assignments."""
    windows: list[DatasetWindow]
    window_size: int
    stride: int
    splits: SplitIndices
    total_windows: int

    @property
    def train_windows(self) -> list[DatasetWindow]:
        return [self.windows[i] for i in self.splits.train]

    @property
    def val_windows(self) -> list[DatasetWindow]:
        return [self.windows[i] for i in self.splits.validation]

    @property
    def iid_test_windows(self) -> list[DatasetWindow]:
        return [self.windows[i] for i in self.splits.iid_test]

    @property
    def shift_test_windows(self) -> list[DatasetWindow]:
        return [self.windows[i] for i in self.splits.shift_test]
