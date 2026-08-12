from __future__ import annotations

from enum import StrEnum

import numpy as np

from app.data.models import GeneratedSignal, SignalType
from app.data.signals.base import SignalGenerator


class WaveformType(StrEnum):
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"


class PeriodicSignal(SignalGenerator):
    """Non-sinusoidal periodic signals: sawtooth, square, or triangle wave.

    Useful for modelling repetitive mechanical processes.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 5.0,
        waveform: WaveformType = WaveformType.SAWTOOTH,
        baseline: float = 0.0,
    ) -> None:
        self.amplitude = amplitude
        self.frequency = frequency
        self.waveform = WaveformType(waveform)
        self.baseline = baseline

    def generate(
        self,
        duration: float,
        sampling_rate: float,
        rng: np.random.Generator,
    ) -> GeneratedSignal:
        t = self.make_timestamps(duration, sampling_rate)
        phase_t = (t * self.frequency) % 1.0  # normalised phase in [0, 1)

        if self.waveform == WaveformType.SAWTOOTH:
            wave = 2.0 * phase_t - 1.0  # -1 → +1
        elif self.waveform == WaveformType.SQUARE:
            wave = np.where(phase_t < 0.5, 1.0, -1.0)
        else:  # triangle
            wave = 1.0 - 4.0 * np.abs(phase_t - 0.5)  # -1 → +1 → -1

        values = self.amplitude * wave + self.baseline
        return GeneratedSignal(
            timestamps=t,
            values=values,
            signal_type=SignalType.PERIODIC,
            metadata={
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "waveform": self.waveform.value,
                "baseline": self.baseline,
            },
        )
