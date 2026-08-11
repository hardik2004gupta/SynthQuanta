"""Unit tests for app.ml.evaluation.metrics (pure-numpy, no DB, no I/O)."""
import numpy as np
import pytest

from app.ml.evaluation.metrics import (
    CLASS_LABELS,
    NUM_CLASSES,
    NORMAL_CLASS,
    compute_classification_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect(n_per_class: int = 10):
    labels = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    return labels, labels.copy()


def _all_normal(n: int = 50):
    y_true = np.zeros(n, dtype=np.int64)
    y_pred = np.zeros(n, dtype=np.int64)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_diagonal(self):
        y_true, y_pred = _perfect(5)
        m = compute_classification_metrics(y_true, y_pred)
        cm = np.array(m.confusion_matrix)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert int(np.trace(cm)) == len(y_true)
        assert int((cm - np.diag(np.diag(cm))).sum()) == 0

    def test_all_same_class(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_pred = np.array([0, 1, 0, 1], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        cm = np.array(m.confusion_matrix)
        assert cm[0, 0] == 1   # TP for class 0
        assert cm[0, 1] == 1   # FP: predicted 1 but true 0
        assert cm[1, 0] == 1   # FP: predicted 0 but true 1
        assert cm[1, 1] == 1   # TP for class 1

    def test_shape_always_7x7(self):
        y_true = np.array([0], dtype=np.int64)
        y_pred = np.array([0], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        assert len(m.confusion_matrix) == NUM_CLASSES
        assert all(len(row) == NUM_CLASSES for row in m.confusion_matrix)


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

class TestPerClassMetrics:
    def test_perfect_f1_is_one(self):
        y_true, y_pred = _perfect(10)
        m = compute_classification_metrics(y_true, y_pred)
        for cls, pcm in m.per_class.items():
            assert pcm.precision == pytest.approx(1.0)
            assert pcm.recall == pytest.approx(1.0)
            assert pcm.f1 == pytest.approx(1.0)

    def test_zero_support_class_has_zero_f1(self):
        y_true = np.zeros(20, dtype=np.int64)
        y_pred = np.zeros(20, dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        # Classes 1-6 have no support → f1 = 0.0
        for cls in CLASS_LABELS[1:]:
            assert m.per_class[cls].f1 == 0.0
            assert m.per_class[cls].support == 0

    def test_class_labels_in_result(self):
        y_true, y_pred = _perfect(3)
        m = compute_classification_metrics(y_true, y_pred)
        assert set(m.per_class.keys()) == set(CLASS_LABELS)

    def test_support_sums_to_n(self):
        y_true = np.array([0, 0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 2], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        total_support = sum(pcm.support for pcm in m.per_class.values())
        assert total_support == len(y_true)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_macro_f1_perfect(self):
        y_true, y_pred = _perfect(10)
        m = compute_classification_metrics(y_true, y_pred)
        assert m.macro_f1 == pytest.approx(1.0)

    def test_macro_f1_unweighted(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_pred = np.array([0, 0, 1, 1], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        # Only class 0 and 1 have support with F1=1.0; rest have 0 support so F1=0.0
        # macro = mean over ALL 7 classes
        expected_macro = 2 / 7  # 2 classes with F1=1.0, 5 with F1=0.0
        assert m.macro_f1 == pytest.approx(expected_macro)

    def test_weighted_f1_dominated_by_majority(self):
        # 90 NORMAL (perfect), 10 NOISE (perfect)
        y_true = np.concatenate([np.zeros(90, dtype=np.int64), np.ones(10, dtype=np.int64)])
        y_pred = y_true.copy()
        m = compute_classification_metrics(y_true, y_pred)
        # weighted_f1 should be 1.0 since both classes are perfect
        assert m.weighted_f1 == pytest.approx(1.0)

    def test_n_samples(self):
        y_true = np.arange(7, dtype=np.int64)
        y_pred = np.arange(7, dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        assert m.n_samples == 7


# ---------------------------------------------------------------------------
# False alarm rate
# ---------------------------------------------------------------------------

class TestFalseAlarmRate:
    def test_zero_when_all_normal_correctly_classified(self):
        y_true, y_pred = _all_normal(30)
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_alarm_rate == pytest.approx(0.0)
        assert m.false_alarm_count == 0

    def test_one_when_all_normal_misclassified(self):
        n = 20
        y_true = np.zeros(n, dtype=np.int64)
        y_pred = np.ones(n, dtype=np.int64)   # predict NOISE for all
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_alarm_rate == pytest.approx(1.0)
        assert m.false_alarm_count == n

    def test_partial_false_alarm(self):
        y_true = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        y_pred = np.array([0, 0, 1, 1, 1], dtype=np.int64)   # 2 of 4 NORMAL misclassified
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_alarm_rate == pytest.approx(0.5)
        assert m.false_alarm_count == 2
        assert m.normal_window_count == 4

    def test_no_normal_windows_rate_is_zero(self):
        y_true = np.array([1, 2, 3], dtype=np.int64)
        y_pred = np.array([1, 2, 3], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_alarm_rate == 0.0
        assert m.normal_window_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample(self):
        y_true = np.array([0], dtype=np.int64)
        y_pred = np.array([0], dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        assert m.n_samples == 1
        assert m.macro_f1 >= 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            compute_classification_metrics(np.array([0, 1]), np.array([0]))

    def test_all_classes_present(self):
        y_true = np.arange(NUM_CLASSES, dtype=np.int64)
        y_pred = np.arange(NUM_CLASSES, dtype=np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        assert len(m.per_class) == NUM_CLASSES

    def test_to_dict_serializable(self):
        import json
        y_true, y_pred = _perfect(3)
        m = compute_classification_metrics(y_true, y_pred)
        d = m.to_dict()
        # Should not raise
        json.dumps(d)
        assert "macro_f1" in d
        assert "per_class" in d
        assert "confusion_matrix" in d
