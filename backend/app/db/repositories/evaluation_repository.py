"""Evaluation repository — all SQLAlchemy access for the evaluations table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.evaluation import Evaluation


class EvaluationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, evaluation_id: str) -> Evaluation | None:
        return self._db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()

    def get_by_human_id(self, human_id: str) -> Evaluation | None:
        return self._db.query(Evaluation).filter(Evaluation.human_id == human_id).first()

    def list_by_experiment(self, experiment_id: str) -> list[Evaluation]:
        return (
            self._db.query(Evaluation)
            .filter(Evaluation.experiment_id == experiment_id)
            .order_by(Evaluation.created_at.desc())
            .all()
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[Evaluation], int]:
        total = self._db.query(Evaluation).count()
        rows = (
            self._db.query(Evaluation)
            .order_by(Evaluation.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(Evaluation).count()

    def create(self, **kwargs: Any) -> Evaluation:
        record = Evaluation(**kwargs)
        self._db.add(record)
        self._db.flush()
        return record

    def update(self, record: Evaluation, **kwargs: Any) -> Evaluation:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._db.flush()
        return record

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
