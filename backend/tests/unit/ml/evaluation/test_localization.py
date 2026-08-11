"""Unit tests for app.ml.evaluation.localization (Interval IoU)."""
import numpy as np
import pytest

from app.ml.evaluation.localization import (
    Interval,
    LocalizationResult,
    _extract_intervals,
    _interval_iou,
    compute_localization,
)

NORMAL = 0
NOISE = 1
DRIFT = 2


# ---------------------------------------------------------------------------
# _extract_intervals
# ---------------------------------------------------------------------------

class TestExtractIntervals:
    def test_no_faults_empty(self):
        labels = np.zeros(10, dtype=np.int64)
        assert _extract_intervals(labels) == []

    def test_single_fault_run(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
        intervals = _extract_intervals(labels)
        assert len(intervals) == 1
        assert intervals[0].start == 2
        assert intervals[0].end == 5
        assert intervals[0].cls == NOISE

    def test_two_separate_fault_runs(self):
        labels = np.array([0, 1, 1, 0, 2, 2, 2, 0], dtype=np.int64)
        intervals = _extract_intervals(labels)
        assert len(intervals) == 2
        assert intervals[0].cls == NOISE and intervals[0].start == 1 and intervals[0].end == 3
        assert intervals[1].cls == DRIFT and intervals[1].start == 4 and intervals[1].end == 7

    def test_fault_at_start_and_end(self):
        labels = np.array([1, 1, 0, 0, 2, 2], dtype=np.int64)
        intervals = _extract_intervals(labels)
        assert len(intervals) == 2
        assert intervals[0].start == 0 and intervals[0].end == 2
        assert intervals[1].start == 4 and intervals[1].end == 6

    def test_all_fault_same_class(self):
        labels = np.ones(5, dtype=np.int64)
        intervals = _extract_intervals(labels)
        assert len(intervals) == 1
        assert intervals[0].start == 0 and intervals[0].end == 5

    def test_empty_array(self):
        assert _extract_intervals(np.array([], dtype=np.int64)) == []


# ---------------------------------------------------------------------------
# _interval_iou
# ---------------------------------------------------------------------------

class TestIntervalIoU:
    def test_perfect_overlap(self):
        a = Interval(2, 6, NOISE)
        b = Interval(2, 6, NOISE)
        assert _interval_iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = Interval(0, 3, NOISE)
        b = Interval(5, 8, NOISE)
        assert _interval_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap_50_percent(self):
        a = Interval(0, 4, NOISE)
        b = Interval(2, 6, NOISE)
        # intersection = [2,4) = 2, union = [0,6) = 6, iou = 2/6 ≈ 0.333
        assert _interval_iou(a, b) == pytest.approx(2 / 6)

    def test_one_contains_other(self):
        a = Interval(1, 9, NOISE)
        b = Interval(3, 6, NOISE)
        # intersection = 3, union = 8, iou = 3/8
        assert _interval_iou(a, b) == pytest.approx(3 / 8)

    def test_adjacent_no_overlap(self):
        a = Interval(0, 3, NOISE)
        b = Interval(3, 6, NOISE)
        assert _interval_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_localization
# ---------------------------------------------------------------------------

class TestComputeLocalization:
    def test_no_gt_intervals_returns_zeros(self):
        y_true = np.zeros(20, dtype=np.int64)
        y_pred = np.zeros(20, dtype=np.int64)
        loc = compute_localization(y_true, y_pred)
        assert isinstance(loc, LocalizationResult)
        assert loc.n_gt_intervals == 0
        assert loc.mean_iou == 0.0
        assert loc.per_interval_iou == []

    def test_perfect_localization(self):
        # GT and pred are identical
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0], dtype=np.int64)
        loc = compute_localization(labels, labels.copy())
        assert loc.n_gt_intervals == 2
        assert loc.mean_iou == pytest.approx(1.0)
        assert loc.iou_at_50 == pytest.approx(1.0)
        assert loc.iou_at_75 == pytest.approx(1.0)

    def test_no_predicted_fault_gives_zero_iou(self):
        y_true = np.array([0, 1, 1, 0], dtype=np.int64)
        y_pred = np.zeros(4, dtype=np.int64)   # model predicts all NORMAL
        loc = compute_localization(y_true, y_pred)
        assert loc.n_gt_intervals == 1
        assert loc.mean_iou == pytest.approx(0.0)
        assert loc.iou_at_50 == pytest.approx(0.0)

    def test_wrong_class_prediction_gives_zero_iou(self):
        y_true = np.array([0, 1, 1, 0], dtype=np.int64)
        y_pred = np.array([0, 2, 2, 0], dtype=np.int64)   # DRIFT instead of NOISE
        loc = compute_localization(y_true, y_pred)
        # GT class=NOISE, pred class=DRIFT → no match → iou=0
        assert loc.mean_iou == pytest.approx(0.0)

    def test_iou_threshold_counts(self):
        # GT interval at [2,8) = 6 windows, pred at [6,8) = 2 windows
        # intersection=2, union=6, iou=2/6 ≈ 0.33 < 0.5
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.int64)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int64)
        loc = compute_localization(y_true, y_pred)
        assert loc.n_gt_intervals == 1
        assert loc.iou_at_50 == pytest.approx(0.0)   # 2/6 ≈ 0.33 < 0.5
        assert loc.iou_at_75 == pytest.approx(0.0)

    def test_to_dict_has_required_keys(self):
        labels = np.array([0, 1, 0], dtype=np.int64)
        loc = compute_localization(labels, labels.copy())
        d = loc.to_dict()
        for key in ("mean_iou", "median_iou", "iou_at_50", "iou_at_75",
                    "n_gt_intervals", "n_pred_intervals", "per_interval_iou"):
            assert key in d
