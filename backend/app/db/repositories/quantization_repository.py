"""QuantizationRepository — all SQLAlchemy access for the quantizations table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.quantization import Quantization


class QuantizationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, quantization_id: str) -> Quantization | None:
        return (
            self._db.query(Quantization)
            .filter(Quantization.id == quantization_id)
            .first()
        )

    def get_by_human_id(self, human_id: str) -> Quantization | None:
        return (
            self._db.query(Quantization)
            .filter(Quantization.human_id == human_id)
            .first()
        )

    def list_by_source_model(self, source_model_id: str) -> list[Quantization]:
        return (
            self._db.query(Quantization)
            .filter(Quantization.source_model_id == source_model_id)
            .order_by(Quantization.created_at.desc())
            .all()
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[Quantization], int]:
        total = self._db.query(Quantization).count()
        rows = (
            self._db.query(Quantization)
            .order_by(Quantization.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(Quantization).count()

    def create(self, **kwargs: Any) -> Quantization:
        record = Quantization(**kwargs)
        self._db.add(record)
        self._db.flush()
        return record

    def update(self, record: Quantization, **kwargs: Any) -> Quantization:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._db.flush()
        return record

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
