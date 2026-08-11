"""Benchmark repository — all SQLAlchemy access for the benchmarks table."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models.benchmark import Benchmark


class BenchmarkRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, benchmark_id: str) -> Benchmark | None:
        return self._db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()

    def get_by_human_id(self, human_id: str) -> Benchmark | None:
        return self._db.query(Benchmark).filter(Benchmark.human_id == human_id).first()

    def list_by_model(self, model_id: str) -> list[Benchmark]:
        return (
            self._db.query(Benchmark)
            .filter(Benchmark.model_id == model_id)
            .order_by(Benchmark.created_at.desc())
            .all()
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[list[Benchmark], int]:
        total = self._db.query(Benchmark).count()
        rows = (
            self._db.query(Benchmark)
            .order_by(Benchmark.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return rows, total

    def count(self) -> int:
        return self._db.query(Benchmark).count()

    def create(self, **kwargs: Any) -> Benchmark:
        record = Benchmark(**kwargs)
        self._db.add(record)
        self._db.flush()
        return record

    def update(self, record: Benchmark, **kwargs: Any) -> Benchmark:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._db.flush()
        return record

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()
