"""Unit tests for SyntheticDataEngine — including the critical determinism test."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.engine import DataEngineError, SyntheticDataEngine


def _base_config(**overrides) -> dict:
    cfg = {
        "seed": 42,
        "signal": {
            "type": "sinusoidal",
            "duration": 5.0,
            "sampling_rate": 100.0,
            "amplitude": 1.0,
            "frequency": 10.0,
        },
        "faults": {},
        "window_size": 128,
    }
    cfg.update(overrides)
    return cfg


class TestDeterminism:
    """Critical: same config + same seed must produce identical output."""

    def test_clean_signal_reproducible(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config()
        raw_a, _, _ = engine.generate(cfg)
        raw_b, _, _ = engine.generate(cfg)
        np.testing.assert_array_equal(raw_a.timestamps, raw_b.timestamps)
        np.testing.assert_array_equal(raw_a.values, raw_b.values)
        np.testing.assert_array_equal(raw_a.labels, raw_b.labels)

    def test_fault_injected_reproducible(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(faults={
            "noise": {"enabled": True, "std": 0.1},
            "drift": {"enabled": True, "magnitude": 0.3},
        })
        raw_a, _, _ = engine.generate(cfg)
        raw_b, _, _ = engine.generate(cfg)
        np.testing.assert_array_equal(raw_a.values, raw_b.values)
        np.testing.assert_array_equal(raw_a.timestamps, raw_b.timestamps)
        assert len(raw_a.fault_annotations) == len(raw_b.fault_annotations)

    def test_different_seeds_differ(self) -> None:
        engine = SyntheticDataEngine()
        cfg_42 = _base_config(seed=42, faults={"noise": {"enabled": True, "std": 0.1}})
        cfg_43 = _base_config(seed=43, faults={"noise": {"enabled": True, "std": 0.1}})
        raw_42, _, _ = engine.generate(cfg_42)
        raw_43, _, _ = engine.generate(cfg_43)
        assert not np.array_equal(raw_42.values, raw_43.values)

    def test_different_seeds_same_signal_differ(self) -> None:
        """Even without faults, TrendSignal with noise uses the rng."""
        engine = SyntheticDataEngine()
        cfg_a = _base_config(seed=1, signal={
            "type": "trend", "duration": 5.0, "sampling_rate": 100.0,
            "amplitude": 1.0, "frequency": 5.0,
            "background_noise_std": 0.05,
        })
        cfg_b = {**cfg_a, "seed": 2}
        raw_a, _, _ = engine.generate(cfg_a)
        raw_b, _, _ = engine.generate(cfg_b)
        assert not np.array_equal(raw_a.values, raw_b.values)


class TestSignalTypes:
    @pytest.mark.parametrize("sig_type", ["sinusoidal", "composite", "trend", "periodic"])
    def test_all_signal_types_generate(self, sig_type: str) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(signal={
            "type": sig_type,
            "duration": 2.0,
            "sampling_rate": 100.0,
            "amplitude": 1.0,
            "frequency": 5.0,
        })
        raw, windowed, validation = engine.generate(cfg)
        assert len(raw.timestamps) == 200
        assert len(raw.values) == 200
        assert windowed.total_windows > 0


class TestFaultCombinations:
    def test_single_fault(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(faults={"noise": {"enabled": True, "std": 0.2}})
        raw, _, _ = engine.generate(cfg)
        assert len(raw.fault_annotations) == 1

    def test_all_six_faults(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(
            signal={"type": "sinusoidal", "duration": 30.0, "sampling_rate": 100.0,
                    "amplitude": 2.0, "frequency": 5.0},
            faults={
                "noise": {"enabled": True, "std": 0.1},
                "drift": {"enabled": True, "magnitude": 0.3, "direction": "positive"},
                "dropout": {"enabled": True, "start_frac": 0.4, "end_frac": 0.45},
                "clipping": {"enabled": True, "lower": -1.5, "upper": 1.5},
                "timestamp_gap": {"enabled": True, "position_frac": 0.6, "gap_seconds": 0.5},
                "sampling_jitter": {"enabled": True, "jitter_std_seconds": 0.0005},
            },
        )
        raw, windowed, validation = engine.generate(cfg)
        assert len(raw.fault_annotations) == 6
        fault_types = {a.fault_type.value for a in raw.fault_annotations}
        assert "NOISE" in fault_types
        assert "DRIFT" in fault_types
        assert "DROPOUT" in fault_types
        assert "CLIPPING" in fault_types
        assert "TIMESTAMP_GAP" in fault_types
        assert "SAMPLING_JITTER" in fault_types

    def test_no_faults(self) -> None:
        engine = SyntheticDataEngine()
        raw, _, _ = engine.generate(_base_config())
        assert len(raw.fault_annotations) == 0
        assert np.all(raw.labels == 0)  # all NORMAL

    def test_fault_order_deterministic(self) -> None:
        """Fault order in annotations must be fixed: noise, drift, dropout, clipping, gap, jitter."""
        engine = SyntheticDataEngine()
        cfg = _base_config(
            signal={"type": "sinusoidal", "duration": 10.0, "sampling_rate": 100.0,
                    "amplitude": 1.0, "frequency": 5.0},
            faults={
                "noise": {"enabled": True, "std": 0.05},
                "drift": {"enabled": True, "magnitude": 0.2},
            },
        )
        raw, _, _ = engine.generate(cfg)
        types = [a.fault_type.value for a in raw.fault_annotations]
        assert types == ["NOISE", "DRIFT"]


class TestWindowing:
    def test_window_count_matches_config(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(window_size=64)
        _, windowed, _ = engine.generate(cfg)
        assert windowed.window_size == 64

    def test_splits_sum_to_total(self) -> None:
        engine = SyntheticDataEngine()
        _, windowed, _ = engine.generate(_base_config())
        total = windowed.total_windows
        split_sum = (
            len(windowed.splits.train)
            + len(windowed.splits.validation)
            + len(windowed.splits.iid_test)
            + len(windowed.splits.shift_test)
        )
        assert split_sum == total


class TestValidation:
    def test_clean_dataset_valid(self) -> None:
        engine = SyntheticDataEngine()
        _, _, validation = engine.generate(_base_config())
        assert validation.valid

    def test_dataset_with_faults_still_valid(self) -> None:
        engine = SyntheticDataEngine()
        cfg = _base_config(faults={
            "noise": {"enabled": True, "std": 0.1},
            "drift": {"enabled": True, "magnitude": 0.3},
        })
        _, _, validation = engine.generate(cfg)
        # Dataset with legitimate faults should still pass validation
        assert validation.structural.valid
        assert validation.ground_truth.valid


class TestConfigMetadata:
    def test_seed_stored(self) -> None:
        engine = SyntheticDataEngine()
        raw, _, _ = engine.generate(_base_config(seed=99))
        assert raw.seed == 99

    def test_generator_version_in_config(self) -> None:
        from app.data.engine import GENERATOR_VERSION
        engine = SyntheticDataEngine()
        raw, _, _ = engine.generate(_base_config())
        assert raw.configuration.get("generator_version") == GENERATOR_VERSION
