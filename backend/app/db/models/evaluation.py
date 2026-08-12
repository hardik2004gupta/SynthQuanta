from __future__ import annotations

from sqlalchemy import Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Evaluation(Base):
    """Evaluation result for a trained model against a dataset split.

    evaluation_type values: "iid" | "noise_shift" | "amplitude_shift" |
    "frequency_shift" | "severity_shift" | "compound_shift"
    """

    __tablename__ = "evaluations"

    # Human-readable ID, e.g. "EVAL-0001"
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="EVAL-0000")
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="RESTRICT"), nullable=False
    )
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False
    )
    evaluation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # Aggregate metrics: precision, recall, f1, false_alarm_rate, interval_iou, …
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Per-class breakdown, confusion matrix, shift degradation, etc.
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # PENDING → RUNNING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    # Wall-clock evaluation duration
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Hardware/software context
    hardware_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Filesystem path relative to artifact root
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="evaluations"
    )
