"""Benchmark API routes — thin delegation to BenchmarkService.

Routes:
    POST /benchmarks/run       → start a benchmark job
    GET  /benchmarks/{id}      → get benchmark result
    GET  /benchmarks           → list all benchmarks
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.benchmark import BenchmarkResponse, BenchmarkRunRequest, BenchmarkStartResponse
from app.schemas.common import ErrorDetail, ErrorResponse
from app.services.benchmark_service import BenchmarkService, BenchmarkServiceError

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _svc(db: Session = Depends(get_db)) -> BenchmarkService:
    return BenchmarkService(db)


@router.post(
    "/run",
    response_model=BenchmarkStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_benchmark(request: BenchmarkRunRequest, svc: BenchmarkService = Depends(_svc)):
    """Start a benchmark job. Returns immediately with benchmark_id; poll GET /benchmarks/{id}."""
    try:
        return svc.start_benchmark_job(request)
    except BenchmarkServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(code="BENCHMARK_START_ERROR", message=str(exc))
            ).model_dump(),
        )


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
def get_benchmark(benchmark_id: str, svc: BenchmarkService = Depends(_svc)):
    """Get a benchmark result by ID."""
    try:
        return svc.get_benchmark(benchmark_id)
    except BenchmarkServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error=ErrorDetail(code="BENCHMARK_NOT_FOUND", message=str(exc))
            ).model_dump(),
        )


@router.get("", response_model=list[BenchmarkResponse])
def list_benchmarks(
    limit: int = 100,
    offset: int = 0,
    svc: BenchmarkService = Depends(_svc),
):
    """List all benchmark results."""
    return svc.list_benchmarks(limit=limit, offset=offset)
