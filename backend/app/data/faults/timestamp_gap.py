from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class TimestampGapFault(Fault):
    """Temporal discontinuity: a gap is injected into the timestamp sequence.

    The timestamp array is modified to include an abnormally large interval
    at the specified position. Signal values are preserved but the temporal
    continuity is broken — validators will detect the unexpected interval.

    This is distinct from Dropout: dropout affects values; timestamp gap
    affects temporal continuity without corrupting sensor values.
    """

    def __init__(
        self,
        position_frac: float = 0.5,
        gap_seconds: float = 1.0,
    ) -> None:
        if not (0.0 < position_frac < 1.0):
            raise ValueError("TimestampGapFault position_frac must be in (0, 1)")
        if gap_seconds <= 0:
            raise ValueError("TimestampGapFault gap_seconds must be positive")
        self.position_frac = position_frac
        self.gap_seconds = gap_seconds

    def apply(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
        fault_id: str,
    ) -> FaultResult:
        n = len(timestamps)
        gap_idx = max(1, int(n * self.position_frac))

        new_timestamps = timestamps.copy()
        # Shift all timestamps after the gap point by gap_seconds
        new_timestamps[gap_idx:] += self.gap_seconds

        # Annotation: single-sample event at gap insertion point
        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.TIMESTAMP_GAP,
            start_index=gap_idx,
            end_index=gap_idx + 1,
            severity=min(1.0, self.gap_seconds),
            parameters={
                "gap_seconds": self.gap_seconds,
                "position_index": gap_idx,
            },
        )
        return FaultResult(values=values.copy(), timestamps=new_timestamps, annotation=annotation)
