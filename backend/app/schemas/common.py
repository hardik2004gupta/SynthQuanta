from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Status enumerations
# ---------------------------------------------------------------------------

class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EntityStatus(StrEnum):
    """Generic lifecycle status for persisted entities (datasets, models, …)."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RuntimeStatus(StrEnum):
    OFFLINE = "offline"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Common response bodies
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    env: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Generic paginated response
# ---------------------------------------------------------------------------

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
