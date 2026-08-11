"""Experiment repository — all SQLAlchemy access for the experiments table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.experiment import Experiment


class ExperimentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_id(self, experiment_id: str) -> Experiment | None:
        return self._db.query(Experiment).filter(Experiment.id == experiment_id).first()

    def get_by_human_id(self, human_id: str) -> Experiment | None:
        return self._db.query(Experiment).filter(Experiment.human_id == human_id).first()

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[Experiment], int]:
        total = self._db.query(Experiment).count()
        rows = (
            self._db.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(Experiment).count()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Experiment:
        record = Experiment(**kwargs)
        self._db.add(record)
        self._db.flush()
        return record

    def update(self, record: Experiment, **kwargs: Any) -> Experiment:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._db.flush()
        return record

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
