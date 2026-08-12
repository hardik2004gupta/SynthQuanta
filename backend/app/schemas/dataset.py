"""Pydantic schemas for the Dataset API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Signal configuration
# ---------------------------------------------------------------------------

class SignalConfig(BaseModel):
    type: str = Field(default="sinusoidal", description="Signal family")
    duration: float = Field(default=10.0, gt=0, le=3600, description="Duration in seconds")
    sampling_rate: float = Field(default=100.0, gt=0, le=10000, description="Samples per second")
    amplitude: float = Field(default=1.0, gt=0, le=100, description="Signal amplitude")
    frequency: float = Field(default=10.0, gt=0, le=5000, description="Primary frequency (Hz)")
    phase: float = Field(default=0.0, description="Phase offset (radians)")
    baseline: float = Field(default=0.0, ge=-1000, le=1000, description="Baseline offset")
    # Composite-specific
    components: list[dict[str, float]] | None = Field(
        default=None,
        description="Explicit frequency components for composite signal"
    )
    # Trend-specific
    slope: float = Field(default=0.05, description="Linear trend slope (units/second)")
    # Periodic-specific
    waveform: str = Field(default="sawtooth", description="Waveform type for periodic signal")

    @field_validator("type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        allowed = {"sinusoidal", "composite", "trend", "periodic"}
        if v not in allowed:
            raise ValueError(f"Signal type must be one of {sorted(allowed)}")
        return v

    @field_validator("waveform")
    @classmethod
    def validate_waveform(cls, v: str) -> str:
        allowed = {"sawtooth", "square", "triangle"}
        if v not in allowed:
            raise ValueError(f"Waveform must be one of {sorted(allowed)}")
        return v


# ---------------------------------------------------------------------------
# Fault configuration
# ---------------------------------------------------------------------------

class NoiseFaultConfig(BaseModel):
    enabled: bool = False
    std: float = Field(default=0.1, gt=0, le=100)
    start_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    end_frac: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def fracs_valid(self) -> "NoiseFaultConfig":
        if self.start_frac >= self.end_frac:
            raise ValueError("start_frac must be less than end_frac")
        return self


class DriftFaultConfig(BaseModel):
    enabled: bool = False
    magnitude: float = Field(default=0.5, gt=0, le=1000)
    direction: str = Field(default="positive")
    start_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    end_frac: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("positive", "negative"):
            raise ValueError("direction must be 'positive' or 'negative'")
        return v

    @model_validator(mode="after")
    def fracs_valid(self) -> "DriftFaultConfig":
        if self.start_frac >= self.end_frac:
            raise ValueError("start_frac must be less than end_frac")
        return self


class DropoutFaultConfig(BaseModel):
    enabled: bool = False
    start_frac: float = Field(default=0.4, ge=0.0, le=1.0)
    end_frac: float = Field(default=0.5, ge=0.0, le=1.0)
    severity: float = Field(default=1.0, gt=0, le=1.0)

    @model_validator(mode="after")
    def fracs_valid(self) -> "DropoutFaultConfig":
        if self.start_frac >= self.end_frac:
            raise ValueError("start_frac must be less than end_frac")
        return self


class ClippingFaultConfig(BaseModel):
    enabled: bool = False
    lower: float = Field(default=-1.0)
    upper: float = Field(default=1.0)

    @model_validator(mode="after")
    def bounds_valid(self) -> "ClippingFaultConfig":
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        return self


class TimestampGapFaultConfig(BaseModel):
    enabled: bool = False
    position_frac: float = Field(default=0.5, gt=0.0, lt=1.0)
    gap_seconds: float = Field(default=1.0, gt=0, le=3600)


class SamplingJitterFaultConfig(BaseModel):
    enabled: bool = False
    jitter_std_seconds: float = Field(default=0.001, gt=0)
    start_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    end_frac: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def fracs_valid(self) -> "SamplingJitterFaultConfig":
        if self.start_frac >= self.end_frac:
            raise ValueError("start_frac must be less than end_frac")
        return self


class FaultConfig(BaseModel):
    noise: NoiseFaultConfig = Field(default_factory=NoiseFaultConfig)
    drift: DriftFaultConfig = Field(default_factory=DriftFaultConfig)
    dropout: DropoutFaultConfig = Field(default_factory=DropoutFaultConfig)
    clipping: ClippingFaultConfig = Field(default_factory=ClippingFaultConfig)
    timestamp_gap: TimestampGapFaultConfig = Field(default_factory=TimestampGapFaultConfig)
    sampling_jitter: SamplingJitterFaultConfig = Field(default_factory=SamplingJitterFaultConfig)


# ---------------------------------------------------------------------------
# Dataset generation request
# ---------------------------------------------------------------------------

class DatasetGenerateRequest(BaseModel):
    name: str = Field(
        default="sensor-dataset",
        min_length=1,
        max_length=255,
        description="Human-readable dataset name",
    )
    seed: int = Field(default=42, ge=0, le=2**31 - 1, description="Reproducibility seed")
    signal: SignalConfig = Field(default_factory=SignalConfig)
    faults: FaultConfig = Field(default_factory=FaultConfig)
    window_size: int = Field(default=128, ge=8, le=4096, description="Samples per window")

    def to_engine_config(self) -> dict[str, Any]:
        """Convert to the dict expected by SyntheticDataEngine.generate()."""
        return {
            "seed": self.seed,
            "signal": self.signal.model_dump(),
            "faults": self.faults.model_dump(),
            "window_size": self.window_size,
        }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FaultAnnotationSchema(BaseModel):
    fault_id: str
    fault_type: str
    start_index: int
    end_index: int
    severity: float
    parameters: dict[str, Any]


class ValidationSummary(BaseModel):
    valid: bool
    issue_count: int
    nan_count: int
    gap_count: int
    annotation_count: int
    fault_types_present: list[str]
    statistics: dict[str, float]


class SignalPreviewPoint(BaseModel):
    t: float   # timestamp
    v: float   # value (NaN represented as null via JSON serialisation)


class DatasetSummary(BaseModel):
    """Lightweight listing record — returned by GET /datasets."""
    dataset_id: str
    human_id: str
    name: str
    status: str
    seed: int
    sample_count: int
    window_count: int
    fault_count: int
    signal_type: str
    duration: float
    sampling_rate: float
    created_at: datetime


class DatasetResponse(BaseModel):
    """Full dataset record — returned by POST /datasets and GET /datasets/{id}."""
    dataset_id: str
    human_id: str
    name: str
    status: str
    seed: int
    sample_count: int
    window_count: int
    fault_count: int
    signal_type: str
    duration: float
    sampling_rate: float
    fault_annotations: list[FaultAnnotationSchema]
    validation: ValidationSummary
    # Downsampled preview for frontend waveform (max 500 points)
    signal_preview: list[list[float]]
    # Split window counts
    split_counts: dict[str, int]
    configuration: dict[str, Any]
    artifact_path: str
    created_at: datetime
