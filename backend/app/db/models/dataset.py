from __future__ import annotations

from sqlalchemy import Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Dataset(Base):
    """Metadata record for a generated synthetic sensor dataset.

    The actual data lives at artifact_path on the filesystem (data.npz + metadata.json).
    PostgreSQL holds only metadata and the path reference.
    """

    __tablename__ = "datasets"

    # Human-readable ID, e.g. "DS-0001" (displayed in the UI)
    human_id: Mapped[str] = mapped_column(String(16), nullable=False, default="DS-0000")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    # Full configuration that produced this dataset (enables reproduction)
    configuration: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Relative path from artifact_root, e.g. "datasets/DS-0001"
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fault_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Validation summary stored as JSON for quick retrieval without loading artifacts
    validation_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # PENDING → COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")

    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="dataset", lazy="select"
    )
