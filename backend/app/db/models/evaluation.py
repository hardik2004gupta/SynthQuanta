from __future__ import annotations

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Evaluation(Base):
    """Evaluation result for a trained model against a dataset split.

    evaluation_type values: "iid" | "noise_shift" | "amplitude_shift" |
    "frequency_shift" | "severity_shift" | "compound_shift"
    """

    __tablename__ = "evaluations"

    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="RESTRICT"), nullable=False
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

    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="evaluations"
    )
