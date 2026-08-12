"""Model repository — all SQLAlchemy access for the models table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.ml_model import MLModel


class ModelRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_id(self, model_id: str) -> MLModel | None:
        return self._db.query(MLModel).filter(MLModel.id == model_id).first()

    def get_by_experiment(self, experiment_id: str) -> list[MLModel]:
        return (
            self._db.query(MLModel)
            .filter(MLModel.experiment_id == experiment_id)
            .order_by(MLModel.created_at.desc())
            .all()
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[MLModel], int]:
        total = self._db.query(MLModel).count()
        rows = (
            self._db.query(MLModel)
            .order_by(MLModel.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(MLModel).count()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> MLModel:
        record = MLModel(**kwargs)
        self._db.add(record)
        self._db.flush()
        return record

    def update(self, record: MLModel, **kwargs: Any) -> MLModel:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._db.flush()
        return record

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
