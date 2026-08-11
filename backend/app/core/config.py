from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "SynthQuanta"
    app_version: str = "0.1.0"
    app_env: Literal["development", "production", "test"] = "development"

    # API
    api_prefix: str = "/api/v1"

    # Database — PostgreSQL metadata store
    database_url: str = "postgresql://synthquanta:synthquanta@localhost:5432/synthquanta"

    # Filesystem artifact root
    artifact_root: str = "../artifacts"

    # CORS — list of allowed origins for the frontend
    cors_origins: list[str] = ["http://localhost:3000"]

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def artifact_root_path(self) -> Path:
        """Resolved absolute path for artifact storage."""
        path = Path(self.artifact_root)
        if not path.is_absolute():
            # Resolve relative to the backend directory
            backend_dir = Path(__file__).parent.parent.parent
            path = (backend_dir / path).resolve()
        return path

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()
