"""Quantization DB model — tracks the FP32 → INT8 job and comparison results."""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Quantization(Base):
    """Tracks a quantization job and its FP32 vs INT8 comparison metrics.

    Lineage: Experiment → MLModel(fp32) → Quantization → MLModel(int8)

    status:  PENDING → RUNNING → COMPLETED | FAILED
    method:  always "dynamic_int8" for MVP
    """

    __tablename__ = "quantizations"

    # Human-readable ID, e.g. "QUANT-0001"
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="QUANT-0000")

    # Source FP32 model
    source_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )
    # Resulting INT8 model (null until COMPLETED)
    quantized_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    # Dataset used for quality comparison
    dataset_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # PENDING | RUNNING | COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    method: Mapped[str] = mapped_column(String(64), nullable=False, default="dynamic_int8")
    backend: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Quality comparison (set on COMPLETED)
    fp32_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    int8_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_delta: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Size comparison
    fp32_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    int8_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_reduction_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Latency comparison
    fp32_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    int8_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_speedup: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Job metadata
    n_test_windows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Path to the INT8 artifact directory (relative to artifact root)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
