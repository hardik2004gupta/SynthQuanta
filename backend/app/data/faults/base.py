from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.data.models import FaultAnnotation


@dataclass
class FaultResult:
    """Output of a single fault application."""
    values: np.ndarray          # modified signal values (may contain NaN for dropout)
    timestamps: np.ndarray      # modified timestamps (may differ for gap/jitter)
    annotation: FaultAnnotation


class Fault(ABC):
    """Abstract base for all fault injectors.

    Every implementation must:
    - derive all randomness from the provided rng
    - return a FaultResult containing the modified arrays and ground-truth annotation
    - never modify the input arrays in-place (return copies)
    """

    @abstractmethod
    def apply(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
        fault_id: str,
    ) -> FaultResult:
        """Apply this fault to a signal.

        Args:
            timestamps: Current timestamp array (float64).
            values:     Current value array (float64, may already contain NaN).
            rng:        Seeded generator — the only source of randomness.
            fault_id:   Unique identifier for the resulting annotation.

        Returns:
            FaultResult with copies of arrays (modified where the fault applies)
            and a complete FaultAnnotation for ground-truth tracking.
        """

    @staticmethod
    def _region_indices(
        n: int,
        start_frac: float,
        end_frac: float,
    ) -> tuple[int, int]:
        """Convert fractional region [start_frac, end_frac] to sample indices."""
        start = max(0, int(n * start_frac))
        end = min(n, int(n * end_frac))
        if end <= start:
            end = min(start + 1, n)
        return start, end
