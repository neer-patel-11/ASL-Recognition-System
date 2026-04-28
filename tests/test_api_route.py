"""
Unit Tests — FastAPI Route Handlers (no live MLflow/model server required)
Uses FastAPI TestClient to test routes in isolation.

Test Plan Reference: TC-U-016 through TC-U-030
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-016  GET / returns 200
# ─────────────────────────────────────────────────────────────────────────────
def test_root_returns_200(client):
    """Root endpoint must return HTTP 200."""
    res = client.get("/")
    assert res.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-017  GET /health returns healthy status
# ─────────────────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    """Health endpoint must return 200 with status=healthy."""
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert "timestamp" in body


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-018  GET /ready returns ready status
# ─────────────────────────────────────────────────────────────────────────────
def test_ready_endpoint(client):
    """Ready endpoint must return 200 with status=ready."""
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-019  GET /metrics returns Prometheus scrape data
# ─────────────────────────────────────────────────────────────────────────────
def test_metrics_endpoint(client):
    """Prometheus /metrics endpoint must return 200 and text content."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers.get("content-type", "")


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-020  GET /admin/pipeline requires auth
# ─────────────────────────────────────────────────────────────────────────────
def test_admin_pipeline_requires_auth(client):
    """Admin pipeline route must reject unauthenticated requests with 401."""
    res = client.get("/admin/pipeline")
    assert res.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-021  GET /admin/pipeline wrong credentials → 401
# ─────────────────────────────────────────────────────────────────────────────
def test_admin_pipeline_wrong_credentials(client, bad_auth):
    """Wrong admin credentials must return 401."""
    res = client.get("/admin/pipeline", auth=bad_auth)
    assert res.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-022  GET /admin/pipeline correct credentials → 200
# ─────────────────────────────────────────────────────────────────────────────
def test_admin_pipeline_correct_credentials(client, admin_auth):
    """Correct admin credentials must return 200 for admin dashboard."""
    res = client.get("/admin/pipeline", auth=admin_auth)
    assert res.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-023  GET /admin/pipeline/log requires auth
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_log_requires_auth(client):
    """Pipeline log endpoint must require admin auth."""
    res = client.get("/admin/pipeline/log")
    assert res.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-024  GET /admin/pipeline/log authenticated → valid JSON
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_log_authenticated(client, admin_auth):
    """Authenticated pipeline log must return valid JSON with expected keys."""
    res = client.get("/admin/pipeline/log", auth=admin_auth)
    assert res.status_code == 200
    body = res.json()
    assert "events" in body
    assert "total_events" in body
    assert "success_rate" in body


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-025  GET /admin/pipeline/stats → valid JSON
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_stats_authenticated(client, admin_auth):
    """Pipeline stats endpoint must return 200 with JSON."""
    res = client.get("/admin/pipeline/stats", auth=admin_auth)
    assert res.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-026  POST /predict without file → 422 Unprocessable Entity
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_without_file(client):
    """POST /predict without a file must return 422."""
    res = client.post("/predict")
    assert res.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-027  POST /predict with non-image bytes → error response (not 200)
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_invalid_image(client):
    """Sending garbage bytes must return a non-200 error response."""
    res = client.post(
        "/predict",
        files={"file": ("test.png", b"not-an-image", "image/png")},
    )
    assert res.status_code != 200


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-028  GET /admin/tests/history requires auth
# ─────────────────────────────────────────────────────────────────────────────
def test_test_history_requires_auth(client):
    """Test history endpoint must require admin auth."""
    res = client.get("/admin/tests/history")
    assert res.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-029  GET /admin/tests/history authenticated → valid JSON
# ─────────────────────────────────────────────────────────────────────────────
def test_test_history_authenticated(client, admin_auth):
    """Authenticated test history must return JSON with runs list."""
    res = client.get("/admin/tests/history", auth=admin_auth)
    assert res.status_code == 200
    body = res.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-030  GET /admin/mlflow/metrics authenticated → valid JSON
# ─────────────────────────────────────────────────────────────────────────────
def test_model_metrics_endpoint(client, admin_auth):
    """Model metrics endpoint must return 200 with metrics list."""
    res = client.get("/admin/mlflow/metrics", auth=admin_auth)
    assert res.status_code == 200
    body = res.json()
    assert "metrics" in body
    assert isinstance(body["metrics"], list)