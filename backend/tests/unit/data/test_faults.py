"""Unit tests for fault injectors."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.faults import (
    ClippingFault,
    DriftFault,
    DropoutFault,
    NoiseFault,
    SamplingJitterFault,
    TimestampGapFault,
)
from app.data.models import FaultType


def _signal(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) * 0.01
    v = np.sin(2 * np.pi * 5.0 * t)
    return t, v


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# NoiseFault
# ---------------------------------------------------------------------------

class TestNoiseFault:
    def test_annotation_type(self) -> None:
        t, v = _signal()
        result = NoiseFault(std=0.1).apply(t, v, _rng(), "f0")
        assert result.annotation.fault_type == FaultType.NOISE

    def test_only_region_modified(self) -> None:
        t, v = _signal(200)
        fault = NoiseFault(std=0.5, start_frac=0.4, end_frac=0.6)
        result = fault.apply(t, v, _rng(), "f0")
        start, end = result.annotation.start_index, result.annotation.end_index
        # Outside region: untouched
        np.testing.assert_array_equal(result.values[:start], v[:start])
        np.testing.assert_array_equal(result.values[end:], v[end:])
        # Inside region: modified
        assert not np.array_equal(result.values[start:end], v[start:end])

    def test_deterministic(self) -> None:
        t, v = _signal()
        fault = NoiseFault(std=0.1)
        r1 = fault.apply(t.copy(), v.copy(), _rng(1), "f0")
        r2 = fault.apply(t.copy(), v.copy(), _rng(1), "f0")
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_different_seeds_differ(self) -> None:
        t, v = _signal()
        fault = NoiseFault(std=0.1)
        r1 = fault.apply(t.copy(), v.copy(), _rng(1), "f0")
        r2 = fault.apply(t.copy(), v.copy(), _rng(2), "f0")
        assert not np.array_equal(r1.values, r2.values)

    def test_input_not_mutated(self) -> None:
        t, v = _signal()
        v_orig = v.copy()
        NoiseFault(std=0.1).apply(t, v, _rng(), "f0")
        np.testing.assert_array_equal(v, v_orig)

    def test_severity_bounded(self) -> None:
        result = NoiseFault(std=0.05).apply(*_signal(), _rng(), "f0")
        assert 0.0 <= result.annotation.severity <= 1.0

    def test_invalid_std(self) -> None:
        with pytest.raises(ValueError):
            NoiseFault(std=0.0)

    def test_invalid_fracs(self) -> None:
        with pytest.raises(ValueError):
            NoiseFault(start_frac=0.8, end_frac=0.2)


# ---------------------------------------------------------------------------
# DriftFault
# ---------------------------------------------------------------------------

class TestDriftFault:
    def test_annotation_type(self) -> None:
        result = DriftFault().apply(*_signal(), _rng(), "f1")
        assert result.annotation.fault_type == FaultType.DRIFT

    def test_positive_drift(self) -> None:
        t, v = _signal(1000)
        fault = DriftFault(magnitude=1.0, direction="positive", start_frac=0.0, end_frac=1.0)
        result = fault.apply(t, v, _rng(), "f1")
        # Last drifted sample should be ~1.0 higher than the original
        last_orig = v[-1]
        last_drifted = result.values[-1]
        assert last_drifted - last_orig > 0.9

    def test_negative_drift(self) -> None:
        t, v = _signal(1000)
        fault = DriftFault(magnitude=1.0, direction="negative", start_frac=0.0, end_frac=1.0)
        result = fault.apply(t, v, _rng(), "f1")
        last_drifted = result.values[-1]
        assert last_drifted < v[-1] - 0.9

    def test_input_not_mutated(self) -> None:
        t, v = _signal()
        v_orig = v.copy()
        DriftFault().apply(t, v, _rng(), "f1")
        np.testing.assert_array_equal(v, v_orig)

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValueError):
            DriftFault(direction="sideways")


# ---------------------------------------------------------------------------
# DropoutFault
# ---------------------------------------------------------------------------

class TestDropoutFault:
    def test_annotation_type(self) -> None:
        result = DropoutFault().apply(*_signal(), _rng(), "f2")
        assert result.annotation.fault_type == FaultType.DROPOUT

    def test_nan_in_region(self) -> None:
        t, v = _signal(1000)
        fault = DropoutFault(start_frac=0.3, end_frac=0.5, severity=1.0)
        result = fault.apply(t, v, _rng(), "f2")
        start, end = result.annotation.start_index, result.annotation.end_index
        assert np.all(np.isnan(result.values[start:end]))

    def test_outside_region_clean(self) -> None:
        t, v = _signal(1000)
        fault = DropoutFault(start_frac=0.3, end_frac=0.5)
        result = fault.apply(t, v, _rng(), "f2")
        start = result.annotation.start_index
        end = result.annotation.end_index
        assert not np.any(np.isnan(result.values[:start]))
        assert not np.any(np.isnan(result.values[end:]))

    def test_input_not_mutated(self) -> None:
        t, v = _signal()
        v_orig = v.copy()
        DropoutFault().apply(t, v, _rng(), "f2")
        np.testing.assert_array_equal(v, v_orig)

    def test_invalid_fracs(self) -> None:
        with pytest.raises(ValueError):
            DropoutFault(start_frac=0.6, end_frac=0.4)


# ---------------------------------------------------------------------------
# ClippingFault
# ---------------------------------------------------------------------------

class TestClippingFault:
    def test_annotation_type(self) -> None:
        result = ClippingFault().apply(*_signal(), _rng(), "f3")
        assert result.annotation.fault_type == FaultType.CLIPPING

    def test_values_clipped(self) -> None:
        t, v = _signal()
        v_scaled = v * 2.0  # will exceed ±1
        result = ClippingFault(lower=-1.0, upper=1.0).apply(t, v_scaled, _rng(), "f3")
        assert np.all(result.values >= -1.0)
        assert np.all(result.values <= 1.0)

    def test_clipping_preserves_in_range(self) -> None:
        t = np.arange(100, dtype=np.float64) * 0.01
        v = np.full(100, 0.5)  # all in range
        result = ClippingFault(lower=-1.0, upper=1.0).apply(t, v, _rng(), "f3")
        np.testing.assert_array_equal(result.values, v)

    def test_input_not_mutated(self) -> None:
        t, v = _signal()
        v_orig = v.copy()
        ClippingFault().apply(t, v, _rng(), "f3")
        np.testing.assert_array_equal(v, v_orig)

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            ClippingFault(lower=1.0, upper=-1.0)


# ---------------------------------------------------------------------------
# TimestampGapFault
# ---------------------------------------------------------------------------

class TestTimestampGapFault:
    def test_annotation_type(self) -> None:
        result = TimestampGapFault().apply(*_signal(), _rng(), "f4")
        assert result.annotation.fault_type == FaultType.TIMESTAMP_GAP

    def test_gap_inserted(self) -> None:
        t, v = _signal(1000)
        gap_seconds = 2.0
        fault = TimestampGapFault(position_frac=0.5, gap_seconds=gap_seconds)
        result = fault.apply(t, v, _rng(), "f4")
        # Maximum interval should be much larger than nominal
        intervals = np.diff(result.timestamps)
        nominal = float(t[1] - t[0])
        assert float(np.max(intervals)) > nominal * 10

    def test_values_unchanged(self) -> None:
        t, v = _signal()
        result = TimestampGapFault().apply(t, v, _rng(), "f4")
        np.testing.assert_array_equal(result.values, v)

    def test_input_timestamps_not_mutated(self) -> None:
        t, v = _signal()
        t_orig = t.copy()
        TimestampGapFault().apply(t, v, _rng(), "f4")
        np.testing.assert_array_equal(t, t_orig)

    def test_invalid_position(self) -> None:
        with pytest.raises(ValueError):
            TimestampGapFault(position_frac=0.0)


# ---------------------------------------------------------------------------
# SamplingJitterFault
# ---------------------------------------------------------------------------

class TestSamplingJitterFault:
    def test_annotation_type(self) -> None:
        result = SamplingJitterFault().apply(*_signal(), _rng(), "f5")
        assert result.annotation.fault_type == FaultType.SAMPLING_JITTER

    def test_values_unchanged(self) -> None:
        t, v = _signal()
        result = SamplingJitterFault().apply(t, v, _rng(), "f5")
        np.testing.assert_array_equal(result.values, v)

    def test_timestamps_differ_from_input(self) -> None:
        t, v = _signal()
        result = SamplingJitterFault(jitter_std_seconds=0.005).apply(t, v, _rng(1), "f5")
        assert not np.array_equal(result.timestamps, t)

    def test_deterministic(self) -> None:
        t, v = _signal()
        fault = SamplingJitterFault(jitter_std_seconds=0.001)
        r1 = fault.apply(t.copy(), v.copy(), _rng(99), "f5")
        r2 = fault.apply(t.copy(), v.copy(), _rng(99), "f5")
        np.testing.assert_array_equal(r1.timestamps, r2.timestamps)

    def test_input_not_mutated(self) -> None:
        t, v = _signal()
        t_orig = t.copy()
        SamplingJitterFault().apply(t, v, _rng(), "f5")
        np.testing.assert_array_equal(t, t_orig)

    def test_invalid_std(self) -> None:
        with pytest.raises(ValueError):
            SamplingJitterFault(jitter_std_seconds=0.0)
