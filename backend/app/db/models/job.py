from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Job(Base):
    """Lightweight background-job tracking record.

    Long-running operations (dataset generation, training, evaluation,
    quantization, benchmarking) create a Job so the frontend can poll
    for status without blocking on the HTTP request.

    Job state machine: QUEUED → RUNNING → COMPLETED
                                        ↘ FAILED
                                        ↘ CANCELLED

    entity_id / entity_type reference the domain record produced by this job
    (e.g., entity_type="dataset", entity_id="DS-0001").
    """

    __tablename__ = "jobs"

    # JOB type: DATASET_GENERATION | TRAINING | EVALUATION | QUANTIZATION | BENCHMARKING
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    # QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    # ID of the entity this job is producing (nullable until the entity is created)
    entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Human-readable error message if the job failed
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
