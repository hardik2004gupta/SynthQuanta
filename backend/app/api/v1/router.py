from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import health
from app.api.v1.routes import datasets
from app.api.v1.routes import training
from app.api.v1.routes import evaluation
from app.api.v1.routes import quantization
from app.api.v1.routes import runtime
from app.api.v1.routes import benchmark

# Central v1 router — sub-routers for each domain are registered here.
api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(evaluation.router)
api_router.include_router(quantization.router)
api_router.include_router(runtime.router)
api_router.include_router(benchmark.router)
