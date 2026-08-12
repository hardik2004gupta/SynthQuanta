from __future__ import annotations

import numpy as np

from app.data.models import GeneratedSignal, SignalType
from app.data.signals.base import SignalGenerator


class TrendSignal(SignalGenerator):
    """Linear trend + optional sinusoidal overlay + optional background noise.

    Models a slowly drifting physical measurement (e.g. temperature rise).
    """

    def __init__(
        self,
        slope: float = 0.05,
        baseline: float = 0.0,
        overlay_amplitude: float = 0.2,
        overlay_frequency: float = 2.0,
        background_noise_std: float = 0.0,
    ) -> None:
        self.slope = slope
        self.baseline = baseline
        self.overlay_amplitude = overlay_amplitude
        self.overlay_frequency = overlay_frequency
        self.background_noise_std = background_noise_std

    def generate(
        self,
        duration: float,
        sampling_rate: float,
        rng: np.random.Generator,
    ) -> GeneratedSignal:
        t = self.make_timestamps(duration, sampling_rate)
        values = self.baseline + self.slope * t
        if self.overlay_amplitude > 0:
            values += self.overlay_amplitude * np.sin(
                2.0 * np.pi * self.overlay_frequency * t
            )
        if self.background_noise_std > 0:
            values += rng.normal(0.0, self.background_noise_std, size=len(t))
        return GeneratedSignal(
            timestamps=t,
            values=values,
            signal_type=SignalType.TREND,
            metadata={
                "slope": self.slope,
                "baseline": self.baseline,
                "overlay_amplitude": self.overlay_amplitude,
                "overlay_frequency": self.overlay_frequency,
                "background_noise_std": self.background_noise_std,
            },
        )
