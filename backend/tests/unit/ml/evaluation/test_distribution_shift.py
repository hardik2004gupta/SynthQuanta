"""Unit tests for distribution-shift config builders (no I/O, no model)."""
import copy

import pytest

from app.ml.evaluation.distribution_shift import (
    SHIFT_SCENARIOS,
    DistributionShiftEngine,
)

# ---------------------------------------------------------------------------
# Minimal base config for config-builder tests
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "seed": 42,
    "signal": {
        "type": "sinusoidal",
        "duration": 2.0,
        "sampling_rate": 100.0,
        "amplitude": 1.0,
        "frequency": 10.0,
    },
    "faults": {
        "noise": {"enabled": True, "std": 0.10, "start_frac": 0.2, "end_frac": 0.8},
        "drift": {"enabled": True, "magnitude": 0.5, "direction": "positive"},
    },
}


def _build_shifted(scenario: str) -> dict:
    """Call the private config-builder directly without running the full engine."""
    import types
    eng = object.__new__(DistributionShiftEngine)
    eng._base_config = copy.deepcopy(BASE_CONFIG)
    eng._seed_offset = 10000
    # Use a dummy model to avoid instantiation
    seed = BASE_CONFIG["seed"] + eng._seed_offset + SHIFT_SCENARIOS.index(scenario)
    return eng._build_shifted_config(scenario, seed)


class TestShiftScenarios:
    def test_all_five_scenarios_defined(self):
        assert len(SHIFT_SCENARIOS) == 5
        expected = {
            "noise_shift", "amplitude_shift", "frequency_shift",
            "severity_shift", "compound_shift",
        }
        assert set(SHIFT_SCENARIOS) == expected

    def test_noise_shift_increases_std(self):
        cfg = _build_shifted("noise_shift")
        base_std = 0.10
        shifted_std = cfg["faults"]["noise"]["std"]
        assert shifted_std > base_std
        assert shifted_std >= 0.30

    def test_amplitude_shift_doubles_amplitude(self):
        cfg = _build_shifted("amplitude_shift")
        original_amp = float(BASE_CONFIG["signal"]["amplitude"])
        shifted_amp = float(cfg["signal"]["amplitude"])
        assert shifted_amp == pytest.approx(original_amp * 2.0)

    def test_frequency_shift_increases_frequency(self):
        cfg = _build_shifted("frequency_shift")
        original_freq = float(BASE_CONFIG["signal"]["frequency"])
        shifted_freq = float(cfg["signal"]["frequency"])
        assert shifted_freq == pytest.approx(original_freq * 1.5)

    def test_severity_shift_increases_fault_params(self):
        cfg = _build_shifted("severity_shift")
        # noise std should have increased
        shifted_noise_std = float(cfg["faults"]["noise"]["std"])
        assert shifted_noise_std > 0.10
        # drift magnitude should have increased
        shifted_drift_mag = float(cfg["faults"]["drift"]["magnitude"])
        assert shifted_drift_mag > 0.5

    def test_compound_shift_enables_multiple_faults(self):
        cfg = _build_shifted("compound_shift")
        enabled_faults = [
            k for k, v in cfg["faults"].items()
            if isinstance(v, dict) and v.get("enabled")
        ]
        assert len(enabled_faults) >= 2
        assert "noise" in enabled_faults
        assert "drift" in enabled_faults

    def test_seed_is_different_per_scenario(self):
        seeds = set()
        for scenario in SHIFT_SCENARIOS:
            cfg = _build_shifted(scenario)
            seeds.add(cfg["seed"])
        assert len(seeds) == len(SHIFT_SCENARIOS), "Each scenario must use a unique seed"

    def test_base_config_not_mutated(self):
        original = copy.deepcopy(BASE_CONFIG)
        for scenario in SHIFT_SCENARIOS:
            _build_shifted(scenario)
        # BASE_CONFIG should be unchanged
        assert BASE_CONFIG == original

    def test_shifted_config_retains_signal_type(self):
        for scenario in SHIFT_SCENARIOS:
            cfg = _build_shifted(scenario)
            assert "signal" in cfg
            assert "type" in cfg["signal"]
