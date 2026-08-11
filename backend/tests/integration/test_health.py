from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_health_returns_version(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    body = response.json()
    assert "version" in body
    assert body["version"]  # non-empty


def test_health_returns_env(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    body = response.json()
    assert "env" in body


def test_health_content_type_is_json(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert "application/json" in response.headers["content-type"]


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
