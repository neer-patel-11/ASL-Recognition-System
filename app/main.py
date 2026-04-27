from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import requests, numpy as np
from PIL import Image
import io, time

app = FastAPI()

# ── custom metrics ──────────────────────────────────────────
prediction_errors   = Counter("prediction_errors_total", "Total prediction errors")
prediction_requests = Counter("prediction_requests_total", "Total prediction requests")
prediction_latency  = Histogram("prediction_latency_seconds", "Prediction latency",
                                buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
drift_score         = Gauge("model_drift_score", "Current data drift score (PSI)")

# baseline pixel mean from training distribution
BASELINE_MEAN = 0.45

# ── prometheus auto-instrumentation (adds /metrics) ─────────
Instrumentator().instrument(app).expose(app)

MLFLOW_URL = "http://host.docker.internal:5001/invocations"
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("app/static/index.html")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32)

def compute_psi(sample_mean: float, baseline_mean: float) -> float:
    """Population Stability Index (simplified scalar version)."""
    eps = 1e-6
    ratio = (sample_mean + eps) / (baseline_mean + eps)
    return abs((sample_mean - baseline_mean) * np.log(ratio))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction_requests.inc()
    start = time.time()
    try:
        image_bytes = await file.read()
        data = preprocess_image(image_bytes)

        # drift detection
        sample_mean = float(data.mean())
        psi = compute_psi(sample_mean, BASELINE_MEAN)
        drift_score.set(psi)

        payload  = {"inputs": data.tolist()}
        response = requests.post(MLFLOW_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        prediction_errors.inc()
        raise
    finally:
        prediction_latency.observe(time.time() - start)