"""Dataset repository — all SQLAlchemy operations for the datasets table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.dataset import Dataset


class DatasetRepository:
    """Thin persistence layer for Dataset metadata records.

    All business logic lives in DatasetService; this class only touches
    the database session.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_id(self, dataset_id: str) -> Dataset | None:
        return self._db.query(Dataset).filter(Dataset.id == dataset_id).first()

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[Dataset], int]:
        total = self._db.query(Dataset).count()
        rows = (
            self._db.query(Dataset)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(Dataset).count()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Dataset:
        dataset = Dataset(**kwargs)
        self._db.add(dataset)
        self._db.flush()   # get the id without committing
        return dataset

    def update(self, dataset: Dataset, **kwargs: Any) -> Dataset:
        for key, value in kwargs.items():
            setattr(dataset, key, value)
        self._db.flush()
        return dataset

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
