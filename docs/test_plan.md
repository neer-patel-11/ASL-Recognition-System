# Test Plan — ASL Recognition System
**Version:** 1.0  
**Date:** 2025-01  
**Project:** ASL Recognition System with MLOps Pipeline  

---

## 1. Introduction

### 1.1 Purpose
This document defines the test strategy, test cases, acceptance criteria, and reporting format for the ASL Recognition System. It covers all software layers: preprocessing utilities, FastAPI REST endpoints, admin pipeline dashboard, MLflow integration, and the test runner itself.

### 1.2 Scope
| In Scope | Out of Scope |
|---|---|
| FastAPI application (`app/main.py`) | Airflow DAG execution |
| Image preprocessing pipeline | MLflow model training |
| Admin authentication (HTTP Basic) | Grafana dashboard rendering |
| Pipeline event logging | Docker networking between containers |
| MLflow API integration (error handling) | Load/stress testing |
| Test runner trigger API | Browser cross-compatibility |

### 1.3 Test Environment
- **Runtime:** Python 3.10+
- **Test Framework:** pytest ≥ 7.0 with `pytest-json-report`
- **HTTP Client:** `fastapi.testclient.TestClient` (no live server required for unit tests)
- **Dependencies:** See `requirements.api.txt`
- **Model Server:** Not required for unit/integration tests (mocked via request failure)

---

## 2. Test Strategy

### 2.1 Test Levels

| Level | Folder | Description |
|---|---|---|
| **Unit** | `tests/unit/` | Test individual functions in isolation. No network I/O. |
| **Integration** | `tests/integration/` | Test component interactions via TestClient. No live model server required. |
| **End-to-End** | Manual (see Section 7) | Full Docker stack with live model server. |

### 2.2 Test Execution
Tests are triggered via:
1. **Admin Dashboard** → `POST /admin/tests/run?suite={all|unit|integration}` (results shown in UI)
2. **CLI:** `pytest tests/ --json-report --json-report-file=report.json`
3. **CI/CD:** DVC pipeline stage or GitHub Actions workflow

### 2.3 Test Tooling
| Tool | Purpose |
|---|---|
| `pytest` | Test runner |
| `pytest-json-report` | Machine-readable JSON output for API consumption |
| `fastapi.testclient` | In-process HTTP testing |
| `Pillow` + `numpy` | Synthetic test image generation |

---

## 3. Test Cases — Unit Tests

### Module: `app/main.py` — `preprocess_image()`

| ID | Test Case | Input | Expected Output | Status |
|---|---|---|---|---|
| TC-U-001 | Output shape correct | 224×224 PNG | shape `(1, 3, 224, 224)` | ⬜ |
| TC-U-002 | Output dtype float32 | 224×224 PNG | `dtype == float32` | ⬜ |
| TC-U-003 | Grayscale auto-converted | Grayscale PNG | shape `(1, 3, 224, 224)` | ⬜ |
| TC-U-004 | Tiny image resized | 8×8 PNG | shape `(1, 3, 224, 224)` | ⬜ |
| TC-U-005 | Pixel normalisation range | 224×224 PNG | values in `[-5, 5]` | ⬜ |
| TC-U-006 | Invalid bytes raise exception | `b"garbage"` | `Exception` raised | ⬜ |
| TC-U-012 | Batch dimension = 1 | 224×224 PNG | `shape[0] == 1` | ⬜ |
| TC-U-013 | Channels-first layout | 224×224 PNG | `shape[1] == 3` | ⬜ |
| TC-U-014 | Spatial dimensions 224×224 | Any PNG | `shape[2,3] == 224` | ⬜ |

### Module: `app/main.py` — `compute_psi()`

| ID | Test Case | Input | Expected Output | Status |
|---|---|---|---|---|
| TC-U-007 | Identical distributions → PSI ≈ 0 | `(0.45, 0.45)` | `< 0.01` | ⬜ |
| TC-U-008 | Large drift → PSI > 0.2 | `(0.9, 0.45)` | `> 0.2` | ⬜ |
| TC-U-009 | PSI always non-negative | Various pairs | `≥ 0` | ⬜ |
| TC-U-010 | Zero inputs no crash | `(0.0, 0.0)` | `float` returned | ⬜ |

### Module: Application Setup

| ID | Test Case | Input | Expected Output | Status |
|---|---|---|---|---|
| TC-U-011 | App imports cleanly | `from app.main import app` | No `ImportError` | ⬜ |
| TC-U-015 | Env var override works | `ADMIN_USERNAME=admin` | Matches env var | ⬜ |

### Module: API Route Handlers

| ID | Test Case | Input | Expected Output | Status |
|---|---|---|---|---|
| TC-U-016 | `GET /` returns 200 | No auth | HTTP 200 | ⬜ |
| TC-U-017 | `GET /health` returns healthy | No auth | `{"status":"healthy"}` | ⬜ |
| TC-U-018 | `GET /ready` returns ready | No auth | `{"status":"ready"}` | ⬜ |
| TC-U-019 | `GET /metrics` returns Prometheus data | No auth | HTTP 200, `text/plain` | ⬜ |
| TC-U-020 | Admin route requires auth | No auth | HTTP 401 | ⬜ |
| TC-U-021 | Admin wrong password → 401 | Bad credentials | HTTP 401 | ⬜ |
| TC-U-022 | Admin correct credentials → 200 | `admin:asl_admin_2024` | HTTP 200 | ⬜ |
| TC-U-023 | Pipeline log requires auth | No auth | HTTP 401 | ⬜ |
| TC-U-024 | Pipeline log returns JSON | Admin auth | Keys: `events`, `total_events` | ⬜ |
| TC-U-025 | Pipeline stats returns JSON | Admin auth | HTTP 200 | ⬜ |
| TC-U-026 | Predict without file → 422 | No file | HTTP 422 | ⬜ |
| TC-U-027 | Predict with invalid image → error | Garbage bytes | Non-200 response | ⬜ |
| TC-U-028 | Test history requires auth | No auth | HTTP 401 | ⬜ |
| TC-U-029 | Test history returns runs list | Admin auth | `{"runs":[...]}` | ⬜ |
| TC-U-030 | Model metrics returns list | Admin auth | `{"metrics":[...]}` | ⬜ |

---

## 4. Test Cases — Integration Tests

| ID | Test Case | Components | Expected Outcome | Status |
|---|---|---|---|---|
| TC-I-001 | Predict call updates pipeline log | `/predict` + `/admin/pipeline/log` | `total_events` increases | ⬜ |
| TC-I-002 | POST event visible in GET log | Pipeline log POST + GET | Event appears in log | ⬜ |
| TC-I-003 | Log filter=success returns only successes | Pipeline log with filter | All events have `status=success` | ⬜ |
| TC-I-004 | Predict failure increments error count | `/predict` (bad image) + log | `errors` count ≥ before | ⬜ |
| TC-I-005 | Health + Ready both succeed simultaneously | `/health` + `/ready` | Both return 200 | ⬜ |
| TC-I-006 | Admin dashboard returns HTML | `GET /admin/pipeline` with auth | HTML with "Pipeline Admin" | ⬜ |
| TC-I-007 | Test history returns valid structure | `GET /admin/tests/history` | `{"runs":[], "total_runs":0}` | ⬜ |
| TC-I-008 | Pipeline stats reflect seeded events | POST log + GET stats | `total_runs` ≥ 1 | ⬜ |
| TC-I-009 | Multiple predict calls handled gracefully | 3× `/predict` | All return valid status codes | ⬜ |
| TC-I-010 | Prometheus counter increments | `/predict` + `/metrics` | `prediction_requests_total` present | ⬜ |
| TC-I-011 | MLflow experiments responds gracefully | `GET /admin/mlflow/experiments` | 200 or 503, not crash | ⬜ |
| TC-I-012 | MLflow models responds gracefully | `GET /admin/mlflow/models` | 200 or 503, not crash | ⬜ |
| TC-I-013 | Log limit parameter respected | `?limit=5` | `len(events) ≤ 5` | ⬜ |
| TC-I-014 | Success rate in valid range | `GET /admin/pipeline/log` | `0 ≤ success_rate ≤ 100` | ⬜ |
| TC-I-015 | OpenAPI docs accessible | `GET /docs` | HTTP 200 | ⬜ |

---

## 5. Acceptance Criteria

The software is considered to **meet acceptance** when all of the following are satisfied:

| # | Criterion | Threshold | Measurement |
|---|---|---|---|
| AC-01 | All unit tests pass | 100% (30/30) | `pytest tests/unit/` exit code 0 |
| AC-02 | All integration tests pass | ≥ 93% (14/15 — MLflow tests may fail if server is down) | `pytest tests/integration/` |
| AC-03 | No import errors | Zero `ImportError` | Checked in TC-U-011 |
| AC-04 | Authentication rejects invalid credentials | 100% of bad-auth tests pass | TC-U-020, TC-U-021, TC-U-023, TC-U-028 |
| AC-05 | Health/Ready endpoints functional | Both return 200 | TC-U-017, TC-U-018, TC-I-005 |
| AC-06 | Pipeline log records all events | `total_events` grows on each predict | TC-I-001 |
| AC-07 | Preprocessing output correct shape/dtype | All shape/dtype tests pass | TC-U-001 to TC-U-005 |
| AC-08 | Test suite completes within 120 seconds | Duration < 120,000ms | Measured by test runner |
| AC-09 | No unhandled 500 errors on expected paths | All intentional-error paths return defined codes | TC-U-026, TC-U-027 |
| AC-10 | Prometheus metrics exposed | `/metrics` returns 200 | TC-U-019, TC-I-010 |

---

## 6. Test Report Template

After each test run, the system automatically generates a report available via `GET /admin/tests/history`. Manual report format:

```
Test Report — ASL Recognition System
Run Date:       <timestamp>
Suite:          <all | unit | integration>
Total Tests:    <N>
Passed:         <N>
Failed:         <N>
Errors:         <N>
Duration:       <Xms>
Pass Rate:      <X%>

Acceptance Criteria Met:  <YES / NO — list failed criteria if NO>

Failed Tests:
  - <nodeid>: <failure summary>

Notes:
  <Any observations, environment issues, flaky tests>
```

---

## 7. Manual / End-to-End Test Cases

These require the full Docker Compose stack (`docker-compose.api.yml`) to be running.

| ID | Test | Steps | Pass Condition |
|---|---|---|---|
| TC-E2E-001 | Full predict flow | Upload ASL hand image via UI → Click Identify | Predicted letter displayed with confidence % |
| TC-E2E-002 | Drift detection shown in UI | Upload very bright/dark image | Drift PSI shown in result panel |
| TC-E2E-003 | Admin dashboard loads pipeline data | Login to `/admin/pipeline` → click Pipeline Console → Refresh | Log events visible in console |
| TC-E2E-004 | Test runner from UI | Click "Run All Tests" in admin panel | Results show passed/failed counts |
| TC-E2E-005 | Grafana shows metrics | Open Grafana → ASL dashboard | Prediction count graph updates |
| TC-E2E-006 | Prometheus scrape working | Open `localhost:9090/targets` | `fastapi` target shows `UP` |

---

## 8. Known Limitations & Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| MLflow server not running during tests | TC-I-011, TC-I-012 may return 503 | Tests are written to accept 503 gracefully |
| Model server (port 5001) not running | `/predict` returns 500 | Integration tests accept 500 for predict path |
| `pytest-json-report` not installed | Test runner falls back to stdout parsing | Install via `pip install pytest-json-report` |
| Tests run inside Docker with no write access to `/tmp` | JSON report save fails | Fall back to stdout; test results still returned |

---

## 9. How to Run Tests

### Via Admin Dashboard
1. Navigate to `http://localhost:8000/admin/pipeline` (credentials: `admin` / `asl_admin_2024`)
2. Click the **🧪 Test Runner** tab
3. Click **▶ Run All Tests**, **▶ Unit Tests**, or **▶ Integration Tests**
4. Results appear inline with per-test pass/fail status

### Via CLI
```bash
# All tests
pytest tests/ -v --tb=short

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With JSON report
pytest tests/ --json-report --json-report-file=test_report.json -q
```

### Install test dependencies
```bash
pip install pytest pytest-json-report fastapi[all] httpx pillow numpy mlflow
```