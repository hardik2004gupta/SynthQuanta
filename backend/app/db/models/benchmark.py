from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Benchmark(Base):
    """Benchmark result for a model across one or more batch sizes.

    latency_metrics: primary batch-size stats {p50, p90, p95, p99, min, max, mean}
    batch_results:   full per-batch-size comparison as JSON list
    memory:          {peak_mb, method} from tracemalloc
    hardware_info:   device, torch_version, platform, etc.
    """

    __tablename__ = "benchmarks"

    # Human-readable ID, e.g. "BENCH-0001"
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="BENCH-0000")

    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )
    # "fp32" | "int8"
    runtime_variant: Mapped[str] = mapped_column(String(16), nullable=False, default="fp32")
    device: Mapped[str] = mapped_column(String(64), nullable=False)
    # Primary batch_size tested (1 for the default latency measurement)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    warmup_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Primary batch_size latency stats {p50, p90, p95, p99, min, max, mean} — ms
    latency_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Requests per second for primary batch_size
    throughput: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Full per-batch-size results list
    batch_results: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # {peak_mb, method}
    memory: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Full hardware context
    hardware_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Relative filesystem path, e.g. "benchmarks/BENCH-0001"
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # PENDING → RUNNING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    # Total benchmark duration in seconds
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Error message on FAILED
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="benchmarks")
