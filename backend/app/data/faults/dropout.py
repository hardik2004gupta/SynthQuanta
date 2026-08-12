from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class DropoutFault(Fault):
    """Signal dropout: missing observations represented as NaN.

    Using NaN preserves the ground-truth distinction between a genuine zero
    reading and a missing/corrupted sample. The downstream model or
    validator must handle NaN explicitly — it will not be silently treated
    as a valid measurement.
    """

    def __init__(
        self,
        start_frac: float = 0.4,
        end_frac: float = 0.5,
        severity: float = 1.0,
    ) -> None:
        if not (0.0 <= start_frac < end_frac <= 1.0):
            raise ValueError("DropoutFault requires 0 ≤ start_frac < end_frac ≤ 1")
        if not (0.0 < severity <= 1.0):
            raise ValueError("DropoutFault severity must be in (0, 1]")
        self.start_frac = start_frac
        self.end_frac = end_frac
        self.severity = severity

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
        # severity < 1: randomly zero out only a fraction of samples in the region
        if self.severity >= 1.0:
            new_values[start:end] = np.nan
        else:
            mask = rng.random(size=end - start) < self.severity
            new_values[start:end] = np.where(mask, np.nan, new_values[start:end])

        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.DROPOUT,
            start_index=start,
            end_index=end,
            severity=self.severity,
            parameters={"dropout_fraction": self.severity},
        )
        return FaultResult(values=new_values, timestamps=timestamps.copy(), annotation=annotation)
