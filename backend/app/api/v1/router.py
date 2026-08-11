from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import health
from app.api.v1.routes import datasets
from app.api.v1.routes import training

# Central v1 router — sub-routers for each domain are registered here.
api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
