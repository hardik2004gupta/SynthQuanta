"""Dataset validation — structural, temporal, statistical, and ground-truth checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.data.models import FaultAnnotation, RawDataset


@dataclass
class StructuralResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    sample_count: int = 0
    has_timestamps: bool = False
    has_values: bool = False
    lengths_match: bool = False
    nan_count: int = 0
    inf_count: int = 0


@dataclass
class TemporalResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    is_monotonic: bool = False
    min_interval: float = 0.0
    max_interval: float = 0.0
    mean_interval: float = 0.0
    gap_count: int = 0       # intervals > 3× mean interval
    jitter_std: float = 0.0  # std of interval deviations from mean


@dataclass
class StatisticalResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    non_nan_count: int = 0


@dataclass
class GroundTruthResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    annotation_count: int = 0
    fault_types_present: list[str] = field(default_factory=list)
    invalid_intervals: int = 0
    invalid_severities: int = 0


@dataclass
class ValidationResult:
    valid: bool                          # True only when all sub-results are valid
    structural: StructuralResult
    temporal: TemporalResult
    statistical: StatisticalResult
    ground_truth: GroundTruthResult
    issues: list[str] = field(default_factory=list)  # aggregated from all checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": self.issues,
            "structural": {
                "valid": self.structural.valid,
                "issues": self.structural.issues,
                "sample_count": self.structural.sample_count,
                "has_timestamps": self.structural.has_timestamps,
                "has_values": self.structural.has_values,
                "lengths_match": self.structural.lengths_match,
                "nan_count": self.structural.nan_count,
                "inf_count": self.structural.inf_count,
            },
            "temporal": {
                "valid": self.temporal.valid,
                "issues": self.temporal.issues,
                "is_monotonic": self.temporal.is_monotonic,
                "min_interval": self.temporal.min_interval,
                "max_interval": self.temporal.max_interval,
                "mean_interval": self.temporal.mean_interval,
                "gap_count": self.temporal.gap_count,
                "jitter_std": self.temporal.jitter_std,
            },
            "statistical": {
                "valid": self.statistical.valid,
                "issues": self.statistical.issues,
                "mean": self.statistical.mean,
                "std": self.statistical.std,
                "min": self.statistical.min,
                "max": self.statistical.max,
                "non_nan_count": self.statistical.non_nan_count,
            },
            "ground_truth": {
                "valid": self.ground_truth.valid,
                "issues": self.ground_truth.issues,
                "annotation_count": self.ground_truth.annotation_count,
                "fault_types_present": self.ground_truth.fault_types_present,
                "invalid_intervals": self.ground_truth.invalid_intervals,
                "invalid_severities": self.ground_truth.invalid_severities,
            },
        }


class DatasetValidator:
    """Validates a RawDataset across four dimensions.

    Does not raise exceptions — returns a structured ValidationResult
    so callers can decide how to handle issues.
    """

    # Intervals larger than this multiple of the mean are considered gaps
    GAP_THRESHOLD_FACTOR = 3.0

    def validate(self, dataset: RawDataset) -> ValidationResult:
        structural = self._check_structural(dataset)
        temporal = self._check_temporal(dataset)
        statistical = self._check_statistical(dataset)
        ground_truth = self._check_ground_truth(dataset)

        all_issues = (
            structural.issues
            + temporal.issues
            + statistical.issues
            + ground_truth.issues
        )
        overall_valid = (
            structural.valid
            and temporal.valid
            and statistical.valid
            and ground_truth.valid
        )
        return ValidationResult(
            valid=overall_valid,
            structural=structural,
            temporal=temporal,
            statistical=statistical,
            ground_truth=ground_truth,
            issues=all_issues,
        )

    # ------------------------------------------------------------------
    # Structural
    # ------------------------------------------------------------------

    def _check_structural(self, dataset: RawDataset) -> StructuralResult:
        issues: list[str] = []

        has_ts = dataset.timestamps is not None and len(dataset.timestamps) > 0
        has_vals = dataset.values is not None and len(dataset.values) > 0

        if not has_ts:
            issues.append("timestamps array is missing or empty")
        if not has_vals:
            issues.append("values array is missing or empty")

        lengths_match = has_ts and has_vals and len(dataset.timestamps) == len(dataset.values)
        if has_ts and has_vals and not lengths_match:
            issues.append(
                f"timestamps length ({len(dataset.timestamps)}) != "
                f"values length ({len(dataset.values)})"
            )

        nan_count = int(np.sum(np.isnan(dataset.values))) if has_vals else 0
        inf_count = int(np.sum(np.isinf(dataset.values))) if has_vals else 0

        if inf_count > 0:
            issues.append(f"{inf_count} infinite value(s) in values array")

        sample_count = len(dataset.timestamps) if has_ts else 0

        return StructuralResult(
            valid=len(issues) == 0,
            issues=issues,
            sample_count=sample_count,
            has_timestamps=has_ts,
            has_values=has_vals,
            lengths_match=lengths_match,
            nan_count=nan_count,
            inf_count=inf_count,
        )

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------

    def _check_temporal(self, dataset: RawDataset) -> TemporalResult:
        issues: list[str] = []
        ts = dataset.timestamps

        if ts is None or len(ts) < 2:
            return TemporalResult(
                valid=False,
                issues=["insufficient timestamps for temporal validation (need ≥ 2)"],
                is_monotonic=False,
            )

        intervals = np.diff(ts)
        is_monotonic = bool(np.all(intervals > 0))
        if not is_monotonic:
            non_mono = int(np.sum(intervals <= 0))
            issues.append(f"{non_mono} non-monotonic timestamp interval(s)")

        mean_interval = float(np.mean(intervals))
        min_interval = float(np.min(intervals))
        max_interval = float(np.max(intervals))

        gap_count = 0
        jitter_std = 0.0
        if mean_interval > 0:
            gap_threshold = self.GAP_THRESHOLD_FACTOR * mean_interval
            gap_count = int(np.sum(intervals > gap_threshold))
            deviations = intervals - mean_interval
            jitter_std = float(np.std(deviations))

        return TemporalResult(
            valid=len(issues) == 0,
            issues=issues,
            is_monotonic=is_monotonic,
            min_interval=min_interval,
            max_interval=max_interval,
            mean_interval=mean_interval,
            gap_count=gap_count,
            jitter_std=jitter_std,
        )

    # ------------------------------------------------------------------
    # Statistical
    # ------------------------------------------------------------------

    def _check_statistical(self, dataset: RawDataset) -> StatisticalResult:
        issues: list[str] = []
        vals = dataset.values

        if vals is None or len(vals) == 0:
            return StatisticalResult(
                valid=False,
                issues=["no values to compute statistics"],
            )

        valid_mask = ~np.isnan(vals) & ~np.isinf(vals)
        non_nan_count = int(np.sum(valid_mask))

        if non_nan_count == 0:
            return StatisticalResult(
                valid=False,
                issues=["all values are NaN or Inf — no valid samples for statistics"],
                non_nan_count=0,
            )

        valid_vals = vals[valid_mask]
        return StatisticalResult(
            valid=len(issues) == 0,
            issues=issues,
            mean=float(np.mean(valid_vals)),
            std=float(np.std(valid_vals)),
            min=float(np.min(valid_vals)),
            max=float(np.max(valid_vals)),
            non_nan_count=non_nan_count,
        )

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _check_ground_truth(self, dataset: RawDataset) -> GroundTruthResult:
        issues: list[str] = []
        annotations: list[FaultAnnotation] = dataset.fault_annotations or []
        n = len(dataset.timestamps) if dataset.timestamps is not None else 0

        invalid_intervals = 0
        invalid_severities = 0
        fault_types: set[str] = set()

        for ann in annotations:
            if ann.start_index >= ann.end_index:
                issues.append(
                    f"Annotation {ann.fault_id}: start_index ({ann.start_index}) "
                    f">= end_index ({ann.end_index})"
                )
                invalid_intervals += 1
            if n > 0 and ann.end_index > n:
                issues.append(
                    f"Annotation {ann.fault_id}: end_index ({ann.end_index}) "
                    f"exceeds dataset length ({n})"
                )
                invalid_intervals += 1
            if not (0.0 <= ann.severity <= 1.0):
                issues.append(
                    f"Annotation {ann.fault_id}: severity ({ann.severity}) not in [0, 1]"
                )
                invalid_severities += 1
            fault_types.add(ann.fault_type.value)

        return GroundTruthResult(
            valid=len(issues) == 0,
            issues=issues,
            annotation_count=len(annotations),
            fault_types_present=sorted(fault_types),
            invalid_intervals=invalid_intervals,
            invalid_severities=invalid_severities,
        )
