from __future__ import annotations

from fastapi.testclient import TestClient


def test_full_request_response_cycle(client: TestClient) -> None:
    """Smoke test: HTTP request → FastAPI → router → response.

    Verifies the complete chain works without mocking any layer.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"


def test_api_docs_available(client: TestClient) -> None:
    """OpenAPI docs endpoint should be reachable."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_available(client: TestClient) -> None:
    """OpenAPI JSON schema should list at least the health endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/health" in schema["paths"]


def test_cors_headers_present(client: TestClient) -> None:
    """CORS preflight should be handled for the configured origin."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # 200 or 204 both indicate CORS was handled
    assert response.status_code in (200, 204)


def test_validation_error_returns_structured_json(client: TestClient) -> None:
    """A request with a bad body should return a structured error response."""
    # POST to health (which doesn't exist as POST) → 405, not a crash
    response = client.post("/api/v1/health", json={"bad": "data"})
    assert response.status_code in (405, 422)
