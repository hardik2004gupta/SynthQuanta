from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class SamplingJitterFault(Fault):
    """Irregular sampling: timestamps receive small random perturbations.

    Models clock jitter, buffering delays, or irregular polling.

    Unlike TimestampGap (which introduces a single large discontinuity),
    jitter applies many small perturbations throughout the signal.
    The values themselves are not changed — only timing is affected.
    """

    def __init__(
        self,
        jitter_std_seconds: float = 0.001,
        start_frac: float = 0.0,
        end_frac: float = 1.0,
    ) -> None:
        if jitter_std_seconds <= 0:
            raise ValueError("SamplingJitterFault jitter_std_seconds must be positive")
        if not (0.0 <= start_frac < end_frac <= 1.0):
            raise ValueError("SamplingJitterFault requires 0 ≤ start_frac < end_frac ≤ 1")
        self.jitter_std_seconds = jitter_std_seconds
        self.start_frac = start_frac
        self.end_frac = end_frac

    def apply(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
        fault_id: str,
    ) -> FaultResult:
        n = len(timestamps)
        start, end = self._region_indices(n, self.start_frac, self.end_frac)

        new_timestamps = timestamps.copy()
        jitter = rng.normal(0.0, self.jitter_std_seconds, size=end - start)
        new_timestamps[start:end] += jitter

        # Expected sample interval from the first two timestamps
        if n >= 2:
            nominal_interval = float(timestamps[1] - timestamps[0])
        else:
            nominal_interval = 1.0
        severity = min(1.0, self.jitter_std_seconds / max(nominal_interval, 1e-9))

        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.SAMPLING_JITTER,
            start_index=start,
            end_index=end,
            severity=severity,
            parameters={"jitter_std_seconds": self.jitter_std_seconds},
        )
        return FaultResult(values=values.copy(), timestamps=new_timestamps, annotation=annotation)
