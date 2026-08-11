"""Unit tests for DatasetValidator."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.models import FaultAnnotation, FaultType, RawDataset, SignalType
from app.data.validation import DatasetValidator


def _make_dataset(
    n: int = 1000,
    with_nan: bool = False,
    with_gap: bool = False,
    annotations: list[FaultAnnotation] | None = None,
) -> RawDataset:
    t = np.arange(n, dtype=np.float64) * 0.01
    v = np.sin(2 * np.pi * 5.0 * t)
    if with_nan:
        v[300:350] = np.nan
    if with_gap:
        t[500:] += 5.0  # large gap at midpoint
    labels = np.zeros(n, dtype=np.int32)
    return RawDataset(
        timestamps=t,
        values=v,
        original_values=v.copy(),
        fault_annotations=annotations or [],
        labels=labels,
        signal_type=SignalType.SINUSOIDAL,
        seed=42,
        configuration={},
    )


class TestStructural:
    def test_valid_dataset_passes(self) -> None:
        result = DatasetValidator().validate(_make_dataset())
        assert result.structural.valid

    def test_nan_detected(self) -> None:
        ds = _make_dataset(with_nan=True)
        result = DatasetValidator().validate(ds)
        assert result.structural.nan_count == 50

    def test_inf_fails(self) -> None:
        ds = _make_dataset()
        ds.values[100] = np.inf
        result = DatasetValidator().validate(ds)
        assert not result.structural.valid
        assert result.structural.inf_count == 1

    def test_length_mismatch(self) -> None:
        ds = _make_dataset()
        ds.values = ds.values[:900]  # mismatch
        result = DatasetValidator().validate(ds)
        assert not result.structural.valid


class TestTemporal:
    def test_monotonic_timestamps_pass(self) -> None:
        result = DatasetValidator().validate(_make_dataset())
        assert result.temporal.is_monotonic

    def test_gap_detected(self) -> None:
        ds = _make_dataset(with_gap=True)
        result = DatasetValidator().validate(ds)
        assert result.temporal.gap_count >= 1

    def test_non_monotonic_fails(self) -> None:
        ds = _make_dataset()
        ds.timestamps[200] = ds.timestamps[199] - 0.001  # backward
        result = DatasetValidator().validate(ds)
        assert not result.temporal.is_monotonic
        assert not result.temporal.valid


class TestStatistical:
    def test_statistics_computed(self) -> None:
        result = DatasetValidator().validate(_make_dataset())
        stat = result.statistical
        assert stat.valid
        assert stat.non_nan_count == 1000
        assert abs(stat.mean) < 0.1  # sinusoid mean ≈ 0
        assert stat.std > 0

    def test_all_nan_fails(self) -> None:
        ds = _make_dataset()
        ds.values[:] = np.nan
        result = DatasetValidator().validate(ds)
        assert not result.statistical.valid


class TestGroundTruth:
    def test_valid_annotation_passes(self) -> None:
        ann = FaultAnnotation(
            fault_id="f0",
            fault_type=FaultType.NOISE,
            start_index=100,
            end_index=200,
            severity=0.5,
        )
        result = DatasetValidator().validate(_make_dataset(annotations=[ann]))
        assert result.ground_truth.valid
        assert result.ground_truth.annotation_count == 1

    def test_invalid_interval_fails(self) -> None:
        ann = FaultAnnotation(
            fault_id="f0",
            fault_type=FaultType.DRIFT,
            start_index=300,
            end_index=200,  # end < start
            severity=0.5,
        )
        result = DatasetValidator().validate(_make_dataset(annotations=[ann]))
        assert not result.ground_truth.valid
        assert result.ground_truth.invalid_intervals == 1

    def test_invalid_severity_fails(self) -> None:
        ann = FaultAnnotation(
            fault_id="f0",
            fault_type=FaultType.NOISE,
            start_index=10,
            end_index=20,
            severity=1.5,  # > 1
        )
        result = DatasetValidator().validate(_make_dataset(annotations=[ann]))
        assert not result.ground_truth.valid
        assert result.ground_truth.invalid_severities == 1

    def test_fault_types_collected(self) -> None:
        anns = [
            FaultAnnotation("f0", FaultType.NOISE, 10, 50, 0.2),
            FaultAnnotation("f1", FaultType.DRIFT, 60, 100, 0.4),
        ]
        result = DatasetValidator().validate(_make_dataset(annotations=anns))
        assert "NOISE" in result.ground_truth.fault_types_present
        assert "DRIFT" in result.ground_truth.fault_types_present


class TestOverallValidity:
    def test_clean_dataset_is_valid(self) -> None:
        result = DatasetValidator().validate(_make_dataset())
        assert result.valid

    def test_issues_aggregated(self) -> None:
        ds = _make_dataset()
        ds.values[:] = np.nan
        result = DatasetValidator().validate(ds)
        assert not result.valid
        assert len(result.issues) > 0

    def test_to_dict_roundtrip(self) -> None:
        result = DatasetValidator().validate(_make_dataset())
        d = result.to_dict()
        assert d["valid"] is True
        assert "structural" in d
        assert "temporal" in d
        assert "statistical" in d
        assert "ground_truth" in d
