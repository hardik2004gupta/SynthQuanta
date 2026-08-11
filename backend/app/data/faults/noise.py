from __future__ import annotations

import numpy as np

from app.data.models import FaultAnnotation, FaultType
from app.data.faults.base import Fault, FaultResult


class NoiseFault(Fault):
    """Additive Gaussian noise injection over a configurable signal region.

    Severity is defined as the noise std relative to expected signal amplitude.
    """

    def __init__(
        self,
        std: float = 0.1,
        start_frac: float = 0.2,
        end_frac: float = 0.8,
    ) -> None:
        if std <= 0:
            raise ValueError("NoiseFault std must be positive")
        if not (0.0 <= start_frac < end_frac <= 1.0):
            raise ValueError("NoiseFault requires 0 ≤ start_frac < end_frac ≤ 1")
        self.std = std
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
        noise = rng.normal(0.0, self.std, size=end - start)
        new_values[start:end] += noise

        # Severity: clamp std to [0, 1] for normalisation (std > 1 → severity 1)
        severity = min(1.0, self.std)

        annotation = FaultAnnotation(
            fault_id=fault_id,
            fault_type=FaultType.NOISE,
            start_index=start,
            end_index=end,
            severity=severity,
            parameters={"std": self.std},
        )
        return FaultResult(values=new_values, timestamps=timestamps.copy(), annotation=annotation)
