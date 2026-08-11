from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Benchmark(Base):
    """Benchmark result metadata for a specific model × device × batch_size run.

    All latency values (P50, P90, P95, P99, min, max, mean) are stored in
    latency_metrics as a JSON object rather than individual columns so the
    schema remains stable as new percentiles are added.

    Raw per-request latency arrays are stored in the filesystem artifact.
    """

    __tablename__ = "benchmarks"

    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )
    device: Mapped[str] = mapped_column(String(64), nullable=False)  # "cpu" | "cuda"
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    warmup_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # {p50, p90, p95, p99, min, max, mean} — all in milliseconds
    latency_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Requests per second
    throughput: Mapped[float | None] = mapped_column(Float, nullable=True)
    # {baseline_mb, peak_mb, model_mb}
    memory: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Full hardware context for result interpretation
    hardware_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Filesystem path for raw latency data, e.g. "benchmarks/BENCH-0001"
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # PENDING → RUNNING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="benchmarks")
