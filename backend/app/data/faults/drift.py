from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class DriftFault(Fault):
    """Gradual baseline shift over a configurable region.

    Applies a linear ramp from 0 to ±magnitude across the affected interval.
    """

    def __init__(
        self,
        magnitude: float = 0.5,
        direction: str = "positive",
        start_frac: float = 0.1,
        end_frac: float = 0.6,
    ) -> None:
        if magnitude <= 0:
            raise ValueError("DriftFault magnitude must be positive")
        if direction not in ("positive", "negative"):
            raise ValueError("DriftFault direction must be 'positive' or 'negative'")
        if not (0.0 <= start_frac < end_frac <= 1.0):
            raise ValueError("DriftFault requires 0 ≤ start_frac < end_frac ≤ 1")
        self.magnitude = magnitude
        self.direction = direction
        self.start_frac = start_frac
        self.end_frac = end_frac

    def apply(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
        fault_id: str,
    ) -> FaultResult:
        n = len(values)
        start, end = self._region_indices(n, self.start_frac, self.end_frac)

        new_values = values.copy()
        ramp = np.linspace(0.0, self.magnitude, end - start)
        if self.direction == "negative":
            ramp = -ramp
        new_values[start:end] += ramp

        severity = min(1.0, self.magnitude)
        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.DRIFT,
            start_index=start,
            end_index=end,
            severity=severity,
            parameters={"magnitude": self.magnitude, "direction": self.direction},
        )
        return FaultResult(values=new_values, timestamps=timestamps.copy(), annotation=annotation)
