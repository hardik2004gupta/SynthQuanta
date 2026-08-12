from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class MLModel(Base):
    """Registered model artifact metadata.

    precision: "fp32" | "int8"
    Artifacts are immutable after registration. A configuration change or
    quantization step creates a new MLModel record.

    Lineage: Experiment → MLModel (fp32) → MLModel (int8, via quantization)
    """

    __tablename__ = "models"

    # Human-readable ID, e.g. "MODEL-0001"
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="MODEL-0000")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    # FK to the experiment that produced this model (nullable for externally loaded)
    experiment_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True
    )
    # "fp32" | "int8"
    precision: Mapped[str] = mapped_column(String(16), nullable=False, default="fp32")
    # Quantization method if applicable, e.g. "dynamic_int8"
    quantization: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Relative artifact path, e.g. "models/MODEL-0001"
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # Latest evaluation metrics attached to this model
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # PENDING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    # Parameter counts
    total_parameters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trainable_parameters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # LoRA configuration (if LoRA/QLoRA was used)
    lora_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    experiment: Mapped["Experiment | None"] = relationship(
        "Experiment", back_populates="models"
    )
    benchmarks: Mapped[list["Benchmark"]] = relationship(
        "Benchmark", back_populates="model", lazy="select"
    )
