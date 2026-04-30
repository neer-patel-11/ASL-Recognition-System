"""
ASL Recognition System - FastAPI Backend
Includes: prediction, MLflow pipeline dashboard (admin-protected), test runner
"""

import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import requests
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
import secrets

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("asl_api")

# ── Admin credentials (change via env vars in production) ─────────────────────
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
MLFLOW_INVOCATION_URL = os.getenv("MLFLOW_URL", "http://host.docker.internal:5001/invocations")
BASELINE_MEAN = 0.45

# ── Prometheus metrics ────────────────────────────────────────────────────────
prediction_errors = Counter("prediction_errors_total", "Total prediction errors")
prediction_requests = Counter("prediction_requests_total", "Total prediction requests")
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
drift_score = Gauge("model_drift_score", "Current data drift score (PSI)")
pipeline_runs_total = Counter("pipeline_runs_total", "Total pipeline stage runs", ["stage", "status"])
test_runs_total = Counter("test_runs_total", "Total test suite runs", ["status"])

# ── Pipeline run log (in-memory, survives container lifetime) ─────────────────
pipeline_run_log: list[dict] = []
test_run_log: list[dict] = []

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ASL Recognition API",
    description="American Sign Language recognition with MLflow pipeline management",
    version="1.0.0",
)

security = HTTPBasic()
Instrumentator().instrument(app).expose(app)

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Validate admin credentials using constant-time comparison."""
    valid_user = secrets.compare_digest(credentials.username.encode(), ADMIN_USERNAME.encode())
    valid_pass = secrets.compare_digest(credentials.password.encode(), ADMIN_PASSWORD.encode())
    if not (valid_user and valid_pass):
        logger.warning("Failed admin login attempt for user: %s", credentials.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ── Image helpers ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((224, 224))

    arr = np.array(image).astype(np.float32) / 255.0

    # replicate grayscale to 3 channels
    arr = np.stack([arr, arr, arr], axis=0)

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    arr = (arr - mean[:, None, None]) / std[:, None, None]

    return np.expand_dims(arr, axis=0).astype(np.float32)
def compute_psi(sample_mean: float, baseline_mean: float) -> float:
    import numpy as np

    eps = 1e-6

    try:
        ratio = (sample_mean + eps) / (baseline_mean + eps)

        # avoid invalid log
        if ratio <= 0 or not np.isfinite(ratio):
            return 0.0

        psi = (sample_mean - baseline_mean) * np.log(ratio)

        if not np.isfinite(psi):
            return 0.0

        return float(abs(psi))

    except Exception:
        return 0.0
    
def safe_float(x):
    import math
    if x is None or not math.isfinite(x):
        return 0.0
    return float(x)
# ── MLflow helpers ────────────────────────────────────────────────────────────
def _mlflow_client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()


def _log_pipeline_event(stage: str, status: str, details: str = "", duration_ms: float = 0):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stage": stage,
        "status": status,
        "details": details,
        "duration_ms": round(duration_ms, 2),
    }
    pipeline_run_log.append(entry)
    pipeline_runs_total.labels(stage=stage, status=status).inc()
    logger.info("Pipeline event: stage=%s status=%s duration=%.1fms", stage, status, duration_ms)


# ── Routes: public ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
def ready():
    return {"status": "ready", "model_url": MLFLOW_INVOCATION_URL}


@app.get("/user_manual")
def ready():
    return FileResponse(str(STATIC_DIR / "user_manual.html"))
    



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run ASL sign inference on an uploaded image."""
    prediction_requests.inc()
    start = time.time()
    logger.info("Prediction request received: filename=%s", file.filename)
    try:
        image_bytes = await file.read()
        data = preprocess_image(image_bytes)

        sample_mean = float(data.mean())
        psi = compute_psi(sample_mean, BASELINE_MEAN)
        drift_score.set(psi)

        payload = {"inputs": data.tolist()}
        response = requests.post(MLFLOW_INVOCATION_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if isinstance(result.get("predictions"), list):
            preds = result["predictions"]

            if isinstance(preds[0], str):
                label = preds[0]
                result["predictions"] = [{label: 1.0}]
                
        # print("Getting result")
        # print(result)
        elapsed_ms = (time.time() - start) * 1000
        logger.info("Prediction success: latency=%.1fms drift_psi=%.4f", elapsed_ms, psi)
        _log_pipeline_event("inference", "success", f"file={file.filename}", elapsed_ms)

        # return {**result, "drift_psi": round(psi, 5), "latency_ms": round(elapsed_ms, 2)}
        return {
        **result,
        "drift_psi": safe_float(round(psi, 5)),
        "latency_ms": safe_float(round(elapsed_ms, 2))
    }
    except Exception as exc:
        prediction_errors.inc()
        elapsed_ms = (time.time() - start) * 1000
        logger.error("Prediction failed: %s", exc, exc_info=True)
        _log_pipeline_event("inference", "error", str(exc), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        prediction_latency.observe(time.time() - start)


# ── Routes: Admin — Pipeline Dashboard ───────────────────────────────────────
@app.get("/admin/pipeline", include_in_schema=False)
def admin_pipeline_ui(_: str = Depends(require_admin)):
    """Serve the admin pipeline dashboard HTML."""
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/admin/mlflow/experiments")
def list_experiments(_: str = Depends(require_admin)):
    """List all MLflow experiments with run summary."""
    try:
        client = _mlflow_client()
        # print("Got the client")
        # print(client)
        experiments = client.search_experiments()
        result = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=5,
                order_by=["start_time DESC"],
            )
            # print("runs ")
            # print(runs)
            result.append(
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "recent_runs": [
                        {
                            "run_id": r.info.run_id,
                            "status": r.info.status,
                            "start_time": r.info.start_time,
                            "end_time": r.info.end_time,
                            "metrics": r.data.metrics,
                            "params": r.data.params,
                            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                        }
                        for r in runs
                    ],
                }
            )
        print("printing the result")
        print(result)
        return {"experiments": result, "total": len(result)}
    except Exception as exc:
        logger.error("MLflow experiments fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")


@app.get("/admin/mlflow/models")
def list_registered_models(_: str = Depends(require_admin)):
    """List registered MLflow models and their versions."""
    try:
        client = _mlflow_client()
        models = client.search_registered_models()
        result = []
        for m in models:
            versions = client.get_latest_versions(m.name)
            result.append(
                {
                    "name": m.name,
                    "description": m.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status,
                            "run_id": v.run_id,
                            "creation_timestamp": v.creation_timestamp,
                        }
                        for v in versions
                    ],
                }
            )
        return {"models": result, "total": len(result)}
    except Exception as exc:
        logger.error("MLflow models fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")


@app.get("/admin/mlflow/metrics")
def get_model_metrics(_: str = Depends(require_admin)):
    """Aggregate metrics from local JSON files (works offline)."""
    metrics_root = Path("models")
    all_metrics = []
    print("metrics")
    print(all_metrics)
    for json_file in sorted(metrics_root.glob("**/*_metrics.json")):
        print("there is a file")
        try:
            with open(json_file) as f:
                data = json.load(f)
            model_name = json_file.stem.replace("_metrics", "")
            all_metrics.append({"model": model_name, "file": str(json_file), **data})
        except Exception as exc:
            logger.warning("Could not read metrics %s: %s", json_file, exc)
    return {"metrics": all_metrics}


@app.get("/admin/pipeline/log")
def get_pipeline_log(
    limit: int = Query(50, le=500),
    status_filter: Optional[str] = None,
    _: str = Depends(require_admin),
):
    """Return recent pipeline run events, optionally filtered by status."""
    log = pipeline_run_log[-limit:]
    if status_filter:
        log = [e for e in log if e["status"] == status_filter]
    success = sum(1 for e in pipeline_run_log if e["status"] == "success")
    errors = sum(1 for e in pipeline_run_log if e["status"] == "error")
    return {
        "total_events": len(pipeline_run_log),
        "success": success,
        "errors": errors,
        "success_rate": round(success / max(len(pipeline_run_log), 1) * 100, 1),
        "events": list(reversed(log)),
    }


@app.post("/admin/pipeline/log")
def add_pipeline_event(
    stage: str,
    event_status: str,
    details: str = "",
    duration_ms: float = 0,
    _: str = Depends(require_admin),
):
    """Manually record a pipeline event (for Airflow/DVC callbacks)."""
    _log_pipeline_event(stage, event_status, details, duration_ms)
    return {"ok": True}


@app.get("/admin/pipeline/stats")
def pipeline_stats(_: str = Depends(require_admin)):
    """Throughput and latency stats for the inference pipeline."""
    if not pipeline_run_log:
        return {"message": "No pipeline events yet"}
    durations = [e["duration_ms"] for e in pipeline_run_log if e["duration_ms"] > 0]
    return {
        "total_runs": len(pipeline_run_log),
        "avg_latency_ms": round(sum(durations) / max(len(durations), 1), 2),
        "min_latency_ms": round(min(durations), 2) if durations else 0,
        "max_latency_ms": round(max(durations), 2) if durations else 0,
        "throughput_rps": round(len(pipeline_run_log) / max((time.time() - 1), 1), 4),
        "stages": list({e["stage"] for e in pipeline_run_log}),
    }


# ── Routes: Admin — Test Runner ───────────────────────────────────────────────
@app.post("/admin/tests/run")
def run_tests(
    suite: str = Query("all", description="all | unit | integration"),
    _: str = Depends(require_admin),
):
    """Trigger pytest for the requested test suite and return results."""
    logger.info("Test run triggered: suite=%s", suite)
    test_map = {
        "all": "tests/",
        "unit": "tests/unit/",
        "integration": "tests/integration/",
    }
    test_path = test_map.get(suite, "tests/")

    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                test_path,
                "--tb=short",
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        duration_ms = (time.time() - start) * 1000

        # Parse the JSON report if available
        report_data = {}
        try:
            with open("/tmp/pytest_report.json") as f:
                report_data = json.load(f)
        except Exception:
            pass
        print("Got the report data")
        print(result)
        summary = report_data.get("summary", {})
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        errors = summary.get("error", 0)
        total = summary.get("total", passed + failed + errors)

        status_str = "passed" if result.returncode == 0 else "failed"
        test_runs_total.labels(status=status_str).inc()

        # Build per-test details
        tests_detail = []
        for test in report_data.get("tests", []):
            tests_detail.append(
                {
                    "nodeid": test.get("nodeid", ""),
                    "outcome": test.get("outcome", "unknown"),
                    "duration": round(test.get("duration", 0) * 1000, 2),
                    "message": (test.get("call", {}) or {}).get("longrepr", ""),
                }
            )

        run_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "suite": suite,
            "status": status_str,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "duration_ms": round(duration_ms, 2),
            "stdout": result.stdout[-3000:],
            "tests": tests_detail,
        }
        test_run_log.append(run_entry)
        _log_pipeline_event("test_runner", status_str, f"suite={suite} passed={passed}/{total}", duration_ms)

        return run_entry

    except subprocess.TimeoutExpired:
        logger.error("Test run timed out after 120s")
        raise HTTPException(status_code=504, detail="Test run timed out (>120s)")
    except Exception as exc:
        logger.error("Test run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/tests/history")
def test_history(limit: int = Query(20, le=100), _: str = Depends(require_admin)):
    """Return the last N test run results."""
    return {"runs": list(reversed(test_run_log[-limit:])), "total_runs": len(test_run_log)}