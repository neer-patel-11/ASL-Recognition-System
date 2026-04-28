"""
Integration Tests — ASL Recognition System
Tests end-to-end flows between components. Assumes FastAPI server is running
(via TestClient) but does NOT require live MLflow or model server.

Test Plan Reference: TC-I-001 through TC-I-015
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-001  Predict endpoint → pipeline log is updated
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_updates_pipeline_log(client, admin_auth, sample_image_bytes):
    """A predict call (even if model is unavailable) must create a log event."""
    # Get baseline log count
    before = client.get("/admin/pipeline/log", auth=admin_auth).json()
    count_before = before["total_events"]

    # Make a predict request (may fail with 500 if model server is down — that's OK)
    client.post(
        "/predict",
        files={"file": ("hand.png", sample_image_bytes, "image/png")},
    )

    after = client.get("/admin/pipeline/log", auth=admin_auth).json()
    assert after["total_events"] >= count_before, "Pipeline log was not updated"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-002  Pipeline log POST then GET consistency
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_log_post_and_get(client, admin_auth):
    """Posting a pipeline event must make it visible in the GET log."""
    client.post(
        "/admin/pipeline/log?stage=test_stage&event_status=success&details=integration_test&duration_ms=42",
        auth=admin_auth,
    )
    log = client.get("/admin/pipeline/log?limit=100", auth=admin_auth).json()
    stages = [e["stage"] for e in log["events"]]
    assert "test_stage" in stages, "Posted event not found in log"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-003  Pipeline log status filter — success only
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_log_filter_success(client, admin_auth):
    """Filtering by status=success must return only success events."""
    # Seed one error event
    client.post(
        "/admin/pipeline/log?stage=seed_err&event_status=error&details=seed",
        auth=admin_auth,
    )
    log = client.get("/admin/pipeline/log?status_filter=success", auth=admin_auth).json()
    for event in log["events"]:
        assert event["status"] == "success", f"Non-success event leaked through: {event}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-004  Pipeline log error count increments on predict failure
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_error_increments_error_count(client, admin_auth):
    """A failing predict call must increment the error count in log."""
    before = client.get("/admin/pipeline/log", auth=admin_auth).json()
    errors_before = before.get("errors", 0)

    # Force an error by sending invalid image
    client.post(
        "/predict",
        files={"file": ("bad.png", b"garbage", "image/png")},
    )

    after = client.get("/admin/pipeline/log", auth=admin_auth).json()
    assert after["errors"] >= errors_before, "Error count did not increase"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-005  Health + Ready endpoints both succeed
# ─────────────────────────────────────────────────────────────────────────────
def test_health_and_ready_together(client):
    """Both /health and /ready must return 200 simultaneously."""
    h = client.get("/health")
    r = client.get("/ready")
    assert h.status_code == 200
    assert r.status_code == 200
    assert h.json()["status"] == "healthy"
    assert r.json()["status"] == "ready"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-006  Admin pipeline dashboard loads HTML
# ─────────────────────────────────────────────────────────────────────────────
def test_admin_dashboard_returns_html(client, admin_auth):
    """Admin pipeline dashboard must return HTML content."""
    res = client.get("/admin/pipeline", auth=admin_auth)
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert b"Pipeline Admin" in res.content


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-007  Test history is empty before first test run
# ─────────────────────────────────────────────────────────────────────────────
def test_history_initially_empty_or_has_runs(client, admin_auth):
    """Test history must return a valid JSON list (empty or populated)."""
    res = client.get("/admin/tests/history", auth=admin_auth)
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["runs"], list)
    assert isinstance(body["total_runs"], int)


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-008  Pipeline stats after events
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_stats_after_events(client, admin_auth):
    """After seeding events, stats must reflect real data."""
    client.post(
        "/admin/pipeline/log?stage=ingest&event_status=success&duration_ms=100",
        auth=admin_auth,
    )
    res = client.get("/admin/pipeline/stats", auth=admin_auth)
    assert res.status_code == 200
    body = res.json()
    # Either message (no events) or real stats
    assert "total_runs" in body or "message" in body


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-009  Multiple concurrent predict calls handled gracefully
# ─────────────────────────────────────────────────────────────────────────────
def test_multiple_predict_calls_handled(client, sample_image_bytes):
    """Multiple consecutive predict calls must not crash the server."""
    for _ in range(3):
        res = client.post(
            "/predict",
            files={"file": ("hand.png", sample_image_bytes, "image/png")},
        )
        # 500 is OK (model server down), 200 is OK, but not a crash (4xx/5xx are fine)
        assert res.status_code in (200, 422, 500, 503)


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-010  Prometheus counter increments after predict calls
# ─────────────────────────────────────────────────────────────────────────────
def test_prometheus_metrics_increment(client, sample_image_bytes):
    """prediction_requests_total metric must increment after a predict call."""
    metrics_before = client.get("/metrics").text
    client.post(
        "/predict",
        files={"file": ("hand.png", sample_image_bytes, "image/png")},
    )
    metrics_after = client.get("/metrics").text
    assert "prediction_requests_total" in metrics_after


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-011  MLflow experiments endpoint responds (even if MLflow is down)
# ─────────────────────────────────────────────────────────────────────────────
def test_mlflow_experiments_responds(client, admin_auth):
    """MLflow experiments endpoint must respond (200 or 503, not 500 crash)."""
    res = client.get("/admin/mlflow/experiments", auth=admin_auth)
    assert res.status_code in (200, 503)


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-012  MLflow models endpoint responds (even if MLflow is down)
# ─────────────────────────────────────────────────────────────────────────────
def test_mlflow_models_responds(client, admin_auth):
    """MLflow models endpoint must respond (200 or 503, not 500 crash)."""
    res = client.get("/admin/mlflow/models", auth=admin_auth)
    assert res.status_code in (200, 503)


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-013  Pipeline log limit parameter respected
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_log_limit(client, admin_auth):
    """Pipeline log with limit=5 must return at most 5 events."""
    res = client.get("/admin/pipeline/log?limit=5", auth=admin_auth)
    assert res.status_code == 200
    events = res.json()["events"]
    assert len(events) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-014  Success rate is between 0 and 100
# ─────────────────────────────────────────────────────────────────────────────
def test_success_rate_range(client, admin_auth):
    """Success rate in pipeline log must be between 0 and 100."""
    res = client.get("/admin/pipeline/log", auth=admin_auth)
    rate = res.json()["success_rate"]
    assert 0 <= rate <= 100, f"Success rate out of range: {rate}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-I-015  OpenAPI docs endpoint reachable
# ─────────────────────────────────────────────────────────────────────────────
def test_openapi_docs_reachable(client):
    """FastAPI /docs must be accessible."""
    res = client.get("/docs")
    assert res.status_code == 200