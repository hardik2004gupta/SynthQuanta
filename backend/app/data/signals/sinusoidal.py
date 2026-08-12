from __future__ import annotations

import numpy as np

from app.data.models import GeneratedSignal, SignalType
from app.data.signals.base import SignalGenerator


class SinusoidalSignal(SignalGenerator):
    """Pure sinusoidal signal: A·sin(2πft + φ) + baseline."""

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 10.0,
        phase: float = 0.0,
        baseline: float = 0.0,
    ) -> None:
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.baseline = baseline

    def generate(
        self,
        duration: float,
        sampling_rate: float,
        rng: np.random.Generator,
    ) -> GeneratedSignal:
        t = self.make_timestamps(duration, sampling_rate)
        values = (
            self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)
            + self.baseline
        )
        return GeneratedSignal(
            timestamps=t,
            values=values,
            signal_type=SignalType.SINUSOIDAL,
            metadata={
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "phase": self.phase,
                "baseline": self.baseline,
            },
        )
