"""Dataset API — thin route layer that delegates to DatasetService."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.common import ErrorDetail, ErrorResponse, Page
from app.schemas.dataset import (
    DatasetGenerateRequest,
    DatasetResponse,
    DatasetSummary,
)
from app.services.dataset_service import DatasetService, DatasetServiceError

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = logging.getLogger(__name__)


def _service(db: Session = Depends(get_db)) -> DatasetService:
    return DatasetService(db)


# ---------------------------------------------------------------------------
# POST /datasets — generate a new dataset
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a synthetic sensor dataset",
)
def generate_dataset(
    request: DatasetGenerateRequest,
    svc: DatasetService = Depends(_service),
) -> DatasetResponse:
    try:
        return svc.generate(request)
    except DatasetServiceError as exc:
        logger.warning("Dataset generation rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /datasets — list datasets
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=Page[DatasetSummary],
    summary="List generated datasets",
)
def list_datasets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    svc: DatasetService = Depends(_service),
) -> Page[DatasetSummary]:
    items, total = svc.list_datasets(limit=limit, offset=offset)
    return Page(items=items, total=total, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id} — retrieve a single dataset
# ---------------------------------------------------------------------------

@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset by ID",
)
def get_dataset(
    dataset_id: str,
    svc: DatasetService = Depends(_service),
) -> DatasetResponse:
    try:
        return svc.get_dataset(dataset_id)
    except DatasetServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
