from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class ClippingFault(Fault):
    """Sensor saturation / clipping: values exceeding bounds are hard-limited.

    Models ADC saturation, sensor range limits, or amplifier clipping.
    The annotation region covers samples that actually hit the bounds.
    """

    def __init__(
        self,
        lower: float = -1.0,
        upper: float = 1.0,
    ) -> None:
        if lower >= upper:
            raise ValueError("ClippingFault lower must be less than upper")
        self.lower = lower
        self.upper = upper

    def apply(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
        fault_id: str,
    ) -> FaultResult:
        new_values = np.clip(values, self.lower, self.upper)

        # Annotation covers samples that were actually clipped
        clipped_mask = (values < self.lower) | (values > self.upper)
        clipped_indices = np.where(clipped_mask)[0]

        if clipped_indices.size > 0:
            start = int(clipped_indices[0])
            end = int(clipped_indices[-1]) + 1
        else:
            # Nothing was clipped — bounds are wider than the signal range
            start = 0
            end = len(values)

        # Severity: fraction of samples clipped, clamped to [0, 1]
        severity = float(np.sum(clipped_mask)) / len(values)
        severity = max(severity, 0.01)  # at least minimal severity when enabled

        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.CLIPPING,
            start_index=start,
            end_index=end,
            severity=min(1.0, severity),
            parameters={"lower": self.lower, "upper": self.upper},
        )
        return FaultResult(values=new_values, timestamps=timestamps.copy(), annotation=annotation)
