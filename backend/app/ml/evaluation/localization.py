"""Fault localization and Interval IoU for SynthQuanta Phase 4.

Model predictions are per-window (each window → one class label).
A "fault interval" is a contiguous run of windows with the same non-NORMAL class.

IoU is computed at the window-index level:
    intersection = overlap in window indices
    union        = total span covered by either interval

We match each ground-truth interval to the highest-IoU predicted interval of
the same class (or 0 if none exists).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMAL_CLASS = 0


@dataclass
class Interval:
    start: int   # inclusive window index
    end: int     # exclusive window index
    cls: int     # fault class label


@dataclass
class LocalizationResult:
    mean_iou: float
    median_iou: float
    iou_at_50: float   # fraction of GT intervals with IoU ≥ 0.5
    iou_at_75: float   # fraction of GT intervals with IoU ≥ 0.75
    n_gt_intervals: int
    n_pred_intervals: int
    per_interval_iou: list[float]   # one entry per GT fault interval

    def to_dict(self) -> dict:
        return {
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "iou_at_50": self.iou_at_50,
            "iou_at_75": self.iou_at_75,
            "n_gt_intervals": self.n_gt_intervals,
            "n_pred_intervals": self.n_pred_intervals,
            "per_interval_iou": self.per_interval_iou,
        }


def _extract_intervals(labels: np.ndarray) -> list[Interval]:
    """Extract contiguous non-NORMAL fault intervals from a label sequence."""
    intervals: list[Interval] = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != NORMAL_CLASS:
            start = i
            cls = int(labels[i])
            while i < n and labels[i] == cls:
                i += 1
            intervals.append(Interval(start=start, end=i, cls=cls))
        else:
            i += 1
    return intervals


def _interval_iou(a: Interval, b: Interval) -> float:
    """1D IoU for two intervals [a.start, a.end) and [b.start, b.end)."""
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    intersection = max(0, inter_end - inter_start)
    union_start = min(a.start, b.start)
    union_end = max(a.end, b.end)
    union = union_end - union_start
    if union <= 0:
        return 0.0
    return intersection / union


def compute_localization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> LocalizationResult:
    """Compute fault localization metrics.

    Args:
        y_true: int64 array of ground-truth window labels
        y_pred: int64 array of predicted window labels

    Returns:
        LocalizationResult with mean/median IoU and threshold metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    gt_intervals = _extract_intervals(y_true)
    pred_intervals = _extract_intervals(y_pred)

    if not gt_intervals:
        # No fault intervals in ground truth — localization is undefined but not an error
        return LocalizationResult(
            mean_iou=0.0, median_iou=0.0, iou_at_50=0.0, iou_at_75=0.0,
            n_gt_intervals=0, n_pred_intervals=len(pred_intervals),
            per_interval_iou=[],
        )

    per_iou: list[float] = []
    for gt in gt_intervals:
        # Find best matching predicted interval of the SAME class
        best_iou = 0.0
        for pred in pred_intervals:
            if pred.cls != gt.cls:
                continue
            iou = _interval_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
        per_iou.append(best_iou)

    ious = np.array(per_iou, dtype=np.float64)
    n_gt = len(gt_intervals)

    return LocalizationResult(
        mean_iou=float(ious.mean()),
        median_iou=float(np.median(ious)),
        iou_at_50=float((ious >= 0.5).sum() / n_gt),
        iou_at_75=float((ious >= 0.75).sum() / n_gt),
        n_gt_intervals=n_gt,
        n_pred_intervals=len(pred_intervals),
        per_interval_iou=per_iou,
    )
