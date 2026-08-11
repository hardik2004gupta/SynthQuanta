"""Pydantic schemas for the Quantization API (Phase 5)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QuantizationRunRequest(BaseModel):
    """POST /api/v1/quantization/run"""

    source_model_id: str = Field(
        ...,
        description="UUID of the FP32 MLModel record to quantize",
    )
    dataset_id: Optional[str] = Field(
        None,
        description="Dataset UUID for quality comparison. If omitted, uses the model's source experiment dataset.",
    )
    benchmark_iterations: int = Field(
        50,
        ge=1,
        le=1000,
        description="Number of timed forward passes for latency comparison",
    )
    benchmark_warmup: int = Field(
        10,
        ge=0,
        le=100,
        description="Warmup passes excluded from latency stats",
    )


class QuantizationStartResponse(BaseModel):
    """Returned immediately after POST /api/v1/quantization/run (202 Accepted)."""

    quantization_id: str
    human_id: str
    source_model_id: str
    status: str
    message: str


class ComparisonMetrics(BaseModel):
    """FP32 vs INT8 comparison metrics — all from actual measurements."""

    fp32_macro_f1: float
    int8_macro_f1: float
    f1_delta: float
    fp32_size_bytes: int
    int8_size_bytes: int
    size_reduction_ratio: float
    fp32_latency_ms: float
    int8_latency_ms: float
    latency_speedup: float
    n_test_windows: int


class QuantizationResponse(BaseModel):
    """GET /api/v1/quantization/{quantization_id}"""

    quantization_id: str
    human_id: str
    source_model_id: str
    quantized_model_id: Optional[str]
    dataset_id: Optional[str]
    status: str
    method: str
    backend: Optional[str]
    comparison: Optional[ComparisonMetrics]
    artifact_path: Optional[str]
    duration_seconds: Optional[float]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
