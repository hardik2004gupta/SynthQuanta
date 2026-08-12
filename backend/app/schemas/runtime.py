"""Pydantic schemas for the SQRuntime API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Load request
# ---------------------------------------------------------------------------

class RuntimeLoadRequest(BaseModel):
    """Load a model artifact into SQRuntime.

    Exactly one of model_id or quantization_id must be provided.
    The backend resolves the artifact from the DB — no filesystem paths accepted.
    """
    model_id: Optional[str] = Field(None, description="UUID of an FP32 MLModel record")
    quantization_id: Optional[str] = Field(None, description="UUID of a completed Quantization record")

    @field_validator("quantization_id")
    @classmethod
    def check_at_least_one(cls, v, info):
        if not v and not info.data.get("model_id"):
            raise ValueError("Either model_id or quantization_id must be provided")
        return v


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

class RuntimeHealthResponse(BaseModel):
    """Observable state of the SQRuntime."""
    status: str                          # uninitialized | loading | ready | failed
    model_id: Optional[str] = None
    artifact_path: Optional[str] = None
    precision: Optional[str] = None
    runtime_variant: Optional[str] = None
    device: str = "cpu"
    backend: Optional[str] = None
    loaded_at: Optional[str] = None      # ISO datetime string
    request_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-window inference request."""
    values: list[float] = Field(..., min_length=1, description="Sensor window values")


class BatchPredictRequest(BaseModel):
    """Batch inference request."""
    windows: list[list[float]] = Field(..., min_length=1, description="List of sensor windows")


class PredictionResponse(BaseModel):
    """Result of a single-window inference call."""
    predicted_class: str
    predicted_class_index: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetryResponse(BaseModel):
    """Runtime telemetry snapshot."""
    request_count: int
    success_count: int
    error_count: int
    last_latency_ms: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
