"""Pydantic schemas for the Training API.

These are the external-facing contract.  Internal dataclasses (TrainingConfig,
EpochMetrics, etc.) live in app.ml.training — schemas are the thin API layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# LoRA hyperparameters (inline in request)
# ---------------------------------------------------------------------------

class LoRAConfigSchema(BaseModel):
    rank: int = Field(default=4, ge=1, le=64, description="LoRA rank")
    alpha: float = Field(default=8.0, gt=0, description="LoRA scaling factor")
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0, description="LoRA dropout")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "out_proj"],
        description="Which linear layers to replace with LoRA",
    )


# ---------------------------------------------------------------------------
# Model architecture overrides
# ---------------------------------------------------------------------------

class ModelConfigSchema(BaseModel):
    architecture: str = Field(default="sensor-transformer-small")
    window_size: int = Field(default=128, ge=8, le=4096)
    embedding_dim: int = Field(default=32, ge=8, le=512)
    num_layers: int = Field(default=2, ge=1, le=12)
    num_heads: int = Field(default=4, ge=1, le=16)
    ffn_dim: int = Field(default=64, ge=8, le=2048)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


# ---------------------------------------------------------------------------
# Training run request
# ---------------------------------------------------------------------------

class TrainingRunRequest(BaseModel):
    dataset_id: str = Field(description="UUID of the dataset record to train on")
    name: str = Field(
        default="experiment",
        min_length=1,
        max_length=255,
        description="Human-readable experiment name",
    )
    method: str = Field(
        default="lora",
        description="Training method: 'full' | 'lora' | 'qlora'",
    )
    epochs: int = Field(default=5, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    lora: LoRAConfigSchema = Field(default_factory=LoRAConfigSchema)
    model: ModelConfigSchema = Field(default_factory=ModelConfigSchema)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"full", "lora", "qlora"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v


# ---------------------------------------------------------------------------
# Start response (returned immediately — job has started)
# ---------------------------------------------------------------------------

class TrainingStartResponse(BaseModel):
    experiment_id: str
    human_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Per-epoch metrics (embedded in experiment response)
# ---------------------------------------------------------------------------

class EpochMetricsSchema(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    duration_seconds: float
    learning_rate: float


# ---------------------------------------------------------------------------
# Experiment response (returned by GET /training/{experiment_id})
# ---------------------------------------------------------------------------

class ExperimentResponse(BaseModel):
    experiment_id: str
    human_id: str
    name: str
    dataset_id: str
    method: str
    status: str
    configuration: dict[str, Any]
    metrics: dict[str, Any] | None
    training_history: list[EpochMetricsSchema] | None
    artifact_path: str | None
    duration_seconds: float | None
    hardware_info: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    model_id: str | None
    model_human_id: str | None


# ---------------------------------------------------------------------------
# Model record (abbreviated — for embedding in experiment response)
# ---------------------------------------------------------------------------

class ModelSummary(BaseModel):
    model_id: str
    human_id: str
    name: str
    precision: str
    status: str
    total_parameters: int | None
    trainable_parameters: int | None
    artifact_path: str
    created_at: datetime
