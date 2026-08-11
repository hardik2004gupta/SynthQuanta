"""Pydantic schemas for the Benchmark API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run request
# ---------------------------------------------------------------------------

class BenchmarkRunRequest(BaseModel):
    """Start a benchmark job.

    Exactly one of model_id or quantization_id must be provided.
    """
    model_id: Optional[str] = Field(None, description="UUID of an FP32 MLModel record")
    quantization_id: Optional[str] = Field(None, description="UUID of a completed Quantization record")
    batch_sizes: list[int] = Field(default=[1, 4, 8, 16], description="Batch sizes to benchmark")
    iterations: int = Field(default=50, ge=1, le=1000, description="Measured calls per batch size")
    warmup: int = Field(default=10, ge=0, le=500, description="Warmup calls per batch size")
    seed: int = Field(default=42, description="RNG seed for input generation")


# ---------------------------------------------------------------------------
# Start response
# ---------------------------------------------------------------------------

class BenchmarkStartResponse(BaseModel):
    benchmark_id: str
    human_id: str
    model_id: str
    runtime_variant: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Per-batch-size result
# ---------------------------------------------------------------------------

class BatchResult(BaseModel):
    batch_size: int
    iterations: int
    warmup_count: int
    latency_stats: dict                       # {p50, p90, p95, p99, min, max, mean}
    throughput_rps: float
    throughput_sps: float
    memory_peak_mb: Optional[float] = None
    duration_seconds: float
    status: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Full benchmark response
# ---------------------------------------------------------------------------

class BenchmarkResponse(BaseModel):
    benchmark_id: str
    human_id: str
    model_id: str
    runtime_variant: str
    device: str
    status: str
    iterations: int
    warmup_count: int
    batch_results: Optional[list[BatchResult]] = None
    latency_metrics: Optional[dict] = None   # primary batch_size stats
    throughput: Optional[float] = None
    memory: Optional[dict] = None
    hardware_info: Optional[dict] = None
    artifact_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    created_at: str
