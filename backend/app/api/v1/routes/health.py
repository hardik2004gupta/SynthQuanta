from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])
_settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Application liveness")
async def health() -> HealthResponse:
    """Return application liveness status.

    This endpoint confirms the FastAPI process is running and routing requests.
    It does NOT check database connectivity or model state — those are
    operational concerns reported by their own endpoints once implemented.
    """
    return HealthResponse(
        status="ok",
        version=_settings.app_version,
        env=_settings.app_env,
    )
