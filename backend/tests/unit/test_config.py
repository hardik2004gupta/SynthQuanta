from __future__ import annotations

import os

import pytest


def test_settings_defaults() -> None:
    """Settings should have sensible defaults when no .env is present."""
    from app.core.config import Settings
    s = Settings()
    assert s.app_name == "SynthQuanta"
    assert s.api_prefix == "/api/v1"
    assert s.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should override defaults."""
    monkeypatch.setenv("APP_NAME", "SynthQuantaTest")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    from app.core.config import Settings
    s = Settings()
    assert s.app_name == "SynthQuantaTest"
    assert s.log_level == "DEBUG"


def test_cors_origins_parsed_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORS_ORIGINS env var must be a JSON list (pydantic-settings v2 requirement)."""
    monkeypatch.setenv(
        "CORS_ORIGINS", '["http://localhost:3000","http://localhost:3001"]'
    )
    from app.core.config import Settings
    s = Settings()
    assert "http://localhost:3000" in s.cors_origins
    assert "http://localhost:3001" in s.cors_origins


def test_artifact_root_path_is_absolute() -> None:
    """artifact_root_path property must always return an absolute Path."""
    from app.core.config import Settings
    s = Settings(artifact_root="../artifacts")
    assert s.artifact_root_path.is_absolute()
