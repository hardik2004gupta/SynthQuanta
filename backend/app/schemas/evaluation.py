"""Pydantic schemas for the Evaluation API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class EvaluationRunRequest(BaseModel):
    experiment_id: str = Field(description="UUID of the experiment/model to evaluate")
    batch_size: int = Field(default=64, ge=1, le=512)
    include_shift: bool = Field(
        default=True,
        description="Whether to run the 5 distribution-shift scenarios after IID evaluation",
    )


# ---------------------------------------------------------------------------
# Start response (returned immediately)
# ---------------------------------------------------------------------------

class EvaluationStartResponse(BaseModel):
    evaluation_id: str
    human_id: str
    experiment_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Per-class metrics (embedded in response)
# ---------------------------------------------------------------------------

class PerClassMetricsSchema(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


# ---------------------------------------------------------------------------
# Aggregate classification metrics
# ---------------------------------------------------------------------------

class ClassificationMetricsSchema(BaseModel):
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    weighted_precision: float
    macro_recall: float
    weighted_recall: float
    false_alarm_rate: float
    false_alarm_count: int
    normal_window_count: int
    n_samples: int
    class_labels: list[str]
    confusion_matrix: list[list[int]]
    per_class: dict[str, PerClassMetricsSchema]


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

class LocalizationMetricsSchema(BaseModel):
    mean_iou: float
    median_iou: float
    iou_at_50: float
    iou_at_75: float
    n_gt_intervals: int
    n_pred_intervals: int
    per_interval_iou: list[float]


# ---------------------------------------------------------------------------
# Distribution-shift scenario result
# ---------------------------------------------------------------------------

class ShiftScenarioSchema(BaseModel):
    scenario: str
    iid_macro_f1: float
    shifted_macro_f1: float
    absolute_degradation: float
    relative_degradation: float
    robustness_ratio: float
    n_shifted_windows: int
    seed: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Full evaluation response
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    evaluation_id: str
    human_id: str
    experiment_id: str
    model_id: str | None
    dataset_id: str
    status: str
    evaluation_type: str
    metrics: dict[str, Any] | None
    results: dict[str, Any] | None
    duration_seconds: float | None
    hardware_info: dict[str, Any] | None
    artifact_path: str | None
    created_at: datetime
