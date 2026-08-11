from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.data.models import GeneratedSignal, SignalType
from app.data.signals.base import SignalGenerator


@dataclass
class SineComponent:
    amplitude: float = 1.0
    frequency: float = 10.0
    phase: float = 0.0


class CompositeSignal(SignalGenerator):
    """Multi-frequency composite: Σ Aᵢ·sin(2πfᵢt + φᵢ) + baseline.

    Mimics a physical sensor stream with multiple spectral components.
    """

    def __init__(
        self,
        components: list[SineComponent] | None = None,
        baseline: float = 0.0,
    ) -> None:
        if components is None:
            components = [
                SineComponent(amplitude=1.0, frequency=10.0, phase=0.0),
                SineComponent(amplitude=0.3, frequency=25.0, phase=0.5),
                SineComponent(amplitude=0.1, frequency=50.0, phase=1.0),
            ]
        self.components = components
        self.baseline = baseline

    def generate(
        self,
        duration: float,
        sampling_rate: float,
        rng: np.random.Generator,
    ) -> GeneratedSignal:
        t = self.make_timestamps(duration, sampling_rate)
        values = np.full_like(t, self.baseline)
        for comp in self.components:
            values += comp.amplitude * np.sin(
                2.0 * np.pi * comp.frequency * t + comp.phase
            )
        return GeneratedSignal(
            timestamps=t,
            values=values,
            signal_type=SignalType.COMPOSITE,
            metadata={
                "components": [
                    {"amplitude": c.amplitude, "frequency": c.frequency, "phase": c.phase}
                    for c in self.components
                ],
                "baseline": self.baseline,
            },
        )
