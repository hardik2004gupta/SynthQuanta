from __future__ import annotations

from sqlalchemy import Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Experiment(Base):
    """Training run metadata.

    Lineage: Dataset → Experiment → MLModel (checkpoint)
    """

    __tablename__ = "experiments"

    # Human-readable ID, e.g. "EXP-0001"
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="EXP-0000")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False
    )
    # Training method: "lora" | "qlora" | "full"
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    # Full training configuration (hyperparams, lora rank, alpha, etc.)
    configuration: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Metrics populated after training completes
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Per-epoch training history: list of {epoch, train_loss, val_loss, val_accuracy, ...}
    training_history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Path to checkpoint artifact, e.g. "models/MODEL-0001"
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # PENDING → RUNNING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    # Wall-clock training duration in seconds
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Hardware/software context captured at training start
    hardware_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="experiments")
    models: Mapped[list["MLModel"]] = relationship(
        "MLModel", back_populates="experiment", lazy="select"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation", back_populates="experiment", lazy="select"
    )
