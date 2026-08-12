from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.data.models import GeneratedSignal


class SignalGenerator(ABC):
    """Abstract base for all synthetic signal generators.

    Implementations must be deterministic given the same rng state.
    """

    @abstractmethod
    def generate(
        self,
        duration: float,
        sampling_rate: float,
        rng: np.random.Generator,
    ) -> GeneratedSignal:
        """Generate a clean signal.

        Args:
            duration:      Total duration in seconds.
            sampling_rate: Samples per second (Hz).
            rng:           Seeded NumPy random generator — all stochastic draws
                           must use this generator to preserve reproducibility.

        Returns:
            GeneratedSignal with timestamps and values arrays of length
            round(duration * sampling_rate).
        """

    @staticmethod
    def make_timestamps(duration: float, sampling_rate: float) -> np.ndarray:
        """Uniform timestamp grid from 0 to duration (exclusive of end)."""
        n = round(duration * sampling_rate)
        return np.arange(n, dtype=np.float64) / sampling_rate
