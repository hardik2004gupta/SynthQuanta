"""Classification metrics for SynthQuanta Phase 4 evaluation.

Pure-numpy implementation — no scikit-learn dependency.  All calculations
are deterministic given the same prediction/ground-truth arrays.

Class ordering (canonical, from app.data.models.FAULT_LABEL):
    0  NORMAL
    1  NOISE
    2  DRIFT
    3  DROPOUT
    4  CLIPPING
    5  TIMESTAMP_GAP
    6  SAMPLING_JITTER
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_LABELS = [
    "NORMAL",
    "NOISE",
    "DRIFT",
    "DROPOUT",
    "CLIPPING",
    "TIMESTAMP_GAP",
    "SAMPLING_JITTER",
]
NUM_CLASSES = len(CLASS_LABELS)
NORMAL_CLASS = 0


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # number of true examples for this class


@dataclass
class ClassificationMetrics:
    """Complete classification result for one evaluation run."""
    # Aggregate
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    weighted_precision: float
    macro_recall: float
    weighted_recall: float

    # Per-class breakdown (key = class label string)
    per_class: dict[str, PerClassMetrics]

    # Confusion matrix (NUM_CLASSES × NUM_CLASSES, row=true, col=pred)
    confusion_matrix: list[list[int]]
    class_labels: list[str]

    # False alarm rate (NORMAL windows predicted as non-NORMAL)
    false_alarm_rate: float
    false_alarm_count: int      # FP for NORMAL class
    normal_window_count: int    # total windows truly NORMAL

    # Sample counts
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "weighted_precision": self.weighted_precision,
            "macro_recall": self.macro_recall,
            "weighted_recall": self.weighted_recall,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_count": self.false_alarm_count,
            "normal_window_count": self.normal_window_count,
            "n_samples": self.n_samples,
            "class_labels": self.class_labels,
            "confusion_matrix": self.confusion_matrix,
            "per_class": {
                cls: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": m.support,
                }
                for cls, m in self.per_class.items()
            },
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassificationMetrics:
    """Compute all classification metrics from true and predicted label arrays.

    Args:
        y_true: int64 array of ground-truth class indices (0–6)
        y_pred: int64 array of predicted class indices (0–6)

    Returns:
        ClassificationMetrics dataclass with all aggregate and per-class stats.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    assert len(y_pred) == n, "y_true and y_pred must have the same length"

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = int(np.clip(t, 0, NUM_CLASSES - 1))
        pi = int(np.clip(p, 0, NUM_CLASSES - 1))
        cm[ti, pi] += 1

    # Per-class metrics
    per_class: dict[str, PerClassMetrics] = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp   # predicted c but not true c
        fn = int(cm[c, :].sum()) - tp   # true c but not predicted c
        support = int(cm[c, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[CLASS_LABELS[c]] = PerClassMetrics(
            precision=prec, recall=rec, f1=f1, support=support
        )
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    supports_arr = np.array(supports, dtype=np.float64)
    total_support = supports_arr.sum()

    macro_f1        = float(np.mean(f1s))
    macro_precision = float(np.mean(precisions))
    macro_recall    = float(np.mean(recalls))

    if total_support > 0:
        weighted_f1        = float(np.dot(f1s, supports_arr) / total_support)
        weighted_precision = float(np.dot(precisions, supports_arr) / total_support)
        weighted_recall    = float(np.dot(recalls, supports_arr) / total_support)
    else:
        weighted_f1 = weighted_precision = weighted_recall = 0.0

    # False alarm rate: normal windows incorrectly predicted as any fault
    normal_mask = y_true == NORMAL_CLASS
    normal_count = int(normal_mask.sum())
    false_alarm_count = int((y_pred[normal_mask] != NORMAL_CLASS).sum()) if normal_count > 0 else 0
    false_alarm_rate = false_alarm_count / normal_count if normal_count > 0 else 0.0

    return ClassificationMetrics(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        weighted_precision=weighted_precision,
        macro_recall=macro_recall,
        weighted_recall=weighted_recall,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_labels=CLASS_LABELS,
        false_alarm_rate=false_alarm_rate,
        false_alarm_count=false_alarm_count,
        normal_window_count=normal_count,
        n_samples=n,
    )
