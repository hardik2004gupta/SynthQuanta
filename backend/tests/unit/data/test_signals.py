"""Unit tests for signal generators."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.models import SignalType
from app.data.signals import (
    CompositeSignal,
    PeriodicSignal,
    SinusoidalSignal,
    TrendSignal,
)
from app.data.signals.composite import SineComponent
from app.data.signals.periodic import WaveformType


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Sinusoidal
# ---------------------------------------------------------------------------

class TestSinusoidalSignal:
    def test_output_length(self) -> None:
        sig = SinusoidalSignal()
        result = sig.generate(duration=1.0, sampling_rate=100.0, rng=_rng())
        assert len(result.timestamps) == 100
        assert len(result.values) == 100

    def test_signal_type(self) -> None:
        result = SinusoidalSignal().generate(1.0, 100.0, _rng())
        assert result.signal_type == SignalType.SINUSOIDAL

    def test_amplitude_respected(self) -> None:
        sig = SinusoidalSignal(amplitude=2.0, frequency=1.0)
        result = sig.generate(1.0, 1000.0, _rng())
        assert abs(np.max(result.values) - 2.0) < 0.01

    def test_deterministic(self) -> None:
        sig = SinusoidalSignal(amplitude=1.5, frequency=5.0)
        r1 = sig.generate(1.0, 100.0, _rng(42))
        r2 = sig.generate(1.0, 100.0, _rng(42))
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_timestamps_start_at_zero(self) -> None:
        result = SinusoidalSignal().generate(5.0, 50.0, _rng())
        assert result.timestamps[0] == pytest.approx(0.0)

    def test_baseline_offset(self) -> None:
        sig = SinusoidalSignal(amplitude=0.5, baseline=3.0)
        result = sig.generate(10.0, 100.0, _rng())
        # mean should be near baseline
        assert abs(np.mean(result.values) - 3.0) < 0.05


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestCompositeSignal:
    def test_output_length(self) -> None:
        result = CompositeSignal().generate(2.0, 100.0, _rng())
        assert len(result.values) == 200

    def test_signal_type(self) -> None:
        result = CompositeSignal().generate(1.0, 100.0, _rng())
        assert result.signal_type == SignalType.COMPOSITE

    def test_custom_components(self) -> None:
        comps = [
            SineComponent(amplitude=1.0, frequency=5.0, phase=0.0),
            SineComponent(amplitude=0.5, frequency=15.0, phase=1.0),
        ]
        sig = CompositeSignal(components=comps)
        result = sig.generate(1.0, 100.0, _rng())
        # Amplitude bounded by sum of component amplitudes
        assert np.max(np.abs(result.values)) <= 1.5 + 0.01

    def test_deterministic(self) -> None:
        sig = CompositeSignal()
        r1 = sig.generate(1.0, 100.0, _rng(7))
        r2 = sig.generate(1.0, 100.0, _rng(7))
        np.testing.assert_array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class TestTrendSignal:
    def test_output_length(self) -> None:
        result = TrendSignal().generate(3.0, 50.0, _rng())
        assert len(result.values) == 150

    def test_signal_type(self) -> None:
        result = TrendSignal().generate(1.0, 100.0, _rng())
        assert result.signal_type == SignalType.TREND

    def test_positive_slope(self) -> None:
        sig = TrendSignal(slope=1.0, overlay_amplitude=0.0, background_noise_std=0.0)
        result = sig.generate(10.0, 10.0, _rng())
        # Last value should be much larger than first
        assert result.values[-1] > result.values[0] + 5.0

    def test_deterministic_without_noise(self) -> None:
        sig = TrendSignal(background_noise_std=0.0)
        r1 = sig.generate(2.0, 100.0, _rng(1))
        r2 = sig.generate(2.0, 100.0, _rng(1))
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_noise_uses_rng(self) -> None:
        sig = TrendSignal(background_noise_std=0.1)
        r1 = sig.generate(1.0, 100.0, _rng(1))
        r2 = sig.generate(1.0, 100.0, _rng(2))
        # Different seeds → different noise
        assert not np.array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# Periodic
# ---------------------------------------------------------------------------

class TestPeriodicSignal:
    def test_output_length(self) -> None:
        result = PeriodicSignal().generate(2.0, 100.0, _rng())
        assert len(result.values) == 200

    def test_signal_type(self) -> None:
        result = PeriodicSignal().generate(1.0, 100.0, _rng())
        assert result.signal_type == SignalType.PERIODIC

    def test_sawtooth_range(self) -> None:
        sig = PeriodicSignal(amplitude=1.0, waveform=WaveformType.SAWTOOTH)
        result = sig.generate(1.0, 1000.0, _rng())
        assert np.min(result.values) >= -1.0 - 1e-9
        assert np.max(result.values) <= 1.0 + 1e-9

    def test_square_range(self) -> None:
        sig = PeriodicSignal(amplitude=1.0, waveform=WaveformType.SQUARE)
        result = sig.generate(1.0, 1000.0, _rng())
        unique = set(np.round(np.unique(result.values), 6))
        assert unique == {-1.0, 1.0}

    def test_triangle_range(self) -> None:
        sig = PeriodicSignal(amplitude=1.0, waveform=WaveformType.TRIANGLE)
        result = sig.generate(1.0, 1000.0, _rng())
        assert np.min(result.values) >= -1.0 - 1e-9
        assert np.max(result.values) <= 1.0 + 1e-9

    def test_deterministic(self) -> None:
        sig = PeriodicSignal(amplitude=2.0, frequency=3.0)
        r1 = sig.generate(1.0, 100.0, _rng(0))
        r2 = sig.generate(1.0, 100.0, _rng(0))
        np.testing.assert_array_equal(r1.values, r2.values)
