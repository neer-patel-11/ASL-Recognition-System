Here's your complete `todo.md`:

```markdown
# ✅ MASTER TODO: Real-Time ASL Recognition System (Full MLOps Project)

---

# 🧠 1. PROBLEM DEFINITION & PLANNING

## 🎯 Define Objectives
- [ ] Define **ML Objective** — Predict ASL alphabet (A–Z, optionally 0–9, space, delete)
- [ ] Define **Business Objective** — Real-time recognition from webcam (<200ms latency), accessible UI for non-technical users
- [ ] Define **Success Metrics**
  - ML: Accuracy, F1-score, confusion matrix
  - System: latency, throughput (FPS), uptime
  - UX: ease of use, error rate by users

## 📄 Documentation
- [ ] Write **Problem Statement**
- [ ] Identify **Stakeholders** (hearing-impaired users, developers)
- [ ] List **Constraints** — No cloud, limited GPU (Colab/local)
- [ ] Identify **Risks** — Lighting variation, background noise, hand occlusion

---

# 📊 2. DATA COLLECTION & UNDERSTANDING

## 📦 Dataset
- [ ] Collect ASL dataset (Kaggle or custom images/video)
- [ ] Validate class balance, image quality, label correctness

## 🔍 EDA (Exploratory Data Analysis)
- [ ] Plot class distribution
- [ ] Plot image size distribution
- [ ] Plot pixel intensity distribution
- [ ] Detect outliers / corrupt images

## 📏 Baseline Statistics (for Drift Detection Later)
- [ ] Compute per-class: mean pixel intensity, variance, histogram (32 bins)
- [ ] Save to `data/baseline_stats.json`
  ```python
  # File: src/data/baseline_stats.py
  # Functions: compute_baseline(dataset_path) -> dict
  # Save: json.dump(stats, open('data/baseline_stats.json', 'w'))
  ```

---

# ⚙️ 3. DATA ENGINEERING PIPELINE

## 🛠 Data Ingestion Pipeline (Airflow DAG)
- [ ] **Create Airflow DAG** for data ingestion
  ```
  File: dags/ingest_dag.py
  DAG tasks: download → validate → store
  Use PythonOperator or BashOperator per step
  Schedule: @once or triggered manually
  ```
- [ ] **Write raw dataset downloader script**
  ```
  File: src/data/download.py
  Functions: download_kaggle_dataset(), extract_zip(), move_to_raw/
  Log: number of images fetched, time taken
  ```
- [ ] **Implement data validation checks**
  ```
  File: src/data/validate.py
  Checks:
    - File exists and is a valid image (not corrupt)
    - Label present and valid (A-Z)
    - Minimum resolution (e.g., 50x50)
    - No duplicate files
  Raise: DataValidationError on failure
  Log: pass/fail counts per class
  ```
- [ ] **Log pipeline throughput**
  ```
  Use time.perf_counter() around batch processing
  Log to logs/pipeline.log
  Expose as Prometheus counter (images_processed_total)
  Target: > 100 images/sec on CPU
  ```

## 🔄 Transformation Pipeline
- [ ] **Build preprocessing pipeline with augmentation**
  ```
  File: src/data/preprocess.py
  Steps: resize(224,224) → normalize([0.485,0.456,0.406]) → augment
  Augmentations:
    - RandomRotation(±15°)
    - HorizontalFlip
    - ColorJitter(brightness=0.3)
  Use: torchvision.transforms.Compose
  ```
- [ ] **Save processed dataset splits**
  ```
  File: src/data/split.py
  Output dirs: data/processed/train/, val/, test/
  Split ratio: 70 / 15 / 15
  Save split manifest: data/splits.json
  ```
- [ ] **Version data with DVC**
  ```bash
  dvc init
  dvc add data/raw data/processed
  dvc remote add -d localstore /path/to/dvcstore
  git add data/raw.dvc data/processed.dvc .dvc/config
  git commit -m "track data with DVC"
  ```

---

# 🧪 4. FEATURE ENGINEERING

## ✋ Feature Extraction Options
- [ ] **Implement MediaPipe hand landmark extractor**
  ```
  File: src/features/landmarks.py
  Class: LandmarkExtractor
  Extract 21 (x,y,z) keypoints → normalize relative to wrist
  Output: np.ndarray shape (63,)
  Handle no-hand-detected → return None
  ```
- [ ] **Build raw pixel feature pipeline (for CNN)**
  ```
  File: src/features/pixels.py
  Returns: torch.Tensor shape (3, 224, 224)
  Wrap torchvision preprocessing into a reusable transform
  Supports: file path, np.ndarray, PIL.Image inputs
  ```
- [ ] **Optionally implement background removal**
  ```
  File: src/features/bg_remove.py
  Use: rembg or cv2 GrabCut
  Log: accuracy with/without bg removal in MLflow for comparison
  ```

## 🧾 Feature Versioning
- [ ] **Version feature pipeline separately from model code**
  ```
  Module: src/features/ (no model imports allowed here)
  DVC stage: dvc run -n features -d src/features/ ...
  Git tag: git tag v1.0-features
  Document feature schema: docs/feature_spec.md
  ```

---

# 🤖 5. MODEL DEVELOPMENT

## 🧠 Model Architectures

- [ ] **Implement MobileNetV2 transfer learning model** *(primary model)*
  ```
  File: src/models/mobilenet.py
  Class: ASLMobileNet(nn.Module)
  __init__:
    - Load pretrained MobileNetV2
    - Freeze base layers
    - Replace classifier head → nn.Linear(..., 29)
  forward(x): returns logits shape (batch, 29)
  Save: models/mobilenet_v1.pth
  ```
- [ ] **Implement lightweight custom CNN baseline**
  ```
  File: src/models/tiny_cnn.py
  Class: TinyCNN(nn.Module)
  Architecture: Conv2d(3→32) → BN → ReLU → MaxPool
              → Conv2d(32→64) → BN → ReLU → MaxPool
              → Conv2d(64→128) → BN → ReLU → MaxPool
              → Flatten → FC(128*28*28 → 512) → FC(512 → 29)
  Target: < 500K parameters
  Purpose: baseline to compare vs transfer learning
  ```
- [ ] **Implement landmark-based MLP model** *(fastest for real-time)*
  ```
  File: src/models/landmark_mlp.py
  Input: 63-dim landmark vector
  Architecture: FC(63→256) → ReLU → Dropout(0.4)
              → FC(256→128) → ReLU → FC(128→29)
  Fastest option — no image preprocessing needed at inference
  ```

## ⚙️ Training Script
- [ ] **Build main training script with MLflow logging**
  ```
  File: src/train.py
  CLI args: --model [mobilenet|tinycnn|mlp] --epochs --lr --batch-size --run-name
  Flow:
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params({model, lr, batch_size, epochs, optimizer})
    for epoch in range(epochs):
      train_one_epoch()
      val_metrics = evaluate()
      mlflow.log_metrics({val_acc, val_f1, train_loss}, step=epoch)
    mlflow.pytorch.log_model(best_model, 'model')
    mlflow.log_artifact('confusion_matrix.png')
  ```
  ```bash
  python src/train.py --model mobilenet --epochs 20 --lr 0.001 --batch-size 32
  ```
- [ ] **Implement early stopping and LR scheduler**
  ```
  File: src/utils/training_utils.py
  Class: EarlyStopping(patience=5, min_delta=0.001)
    - Track best val_loss
    - Set self.early_stop = True when patience exceeded
  Scheduler: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  Log LR per epoch: mlflow.log_metric('lr', scheduler.get_last_lr()[0], step=epoch)
  ```
- [ ] **Log per-class accuracy and confusion matrix**
  ```
  File: src/evaluate.py
  Function: evaluate_model(model, dataloader) -> dict
  Steps:
    - Run inference on full val set
    - sklearn.metrics.classification_report() → per-class precision/recall/F1
    - Plot confusion matrix with seaborn.heatmap()
    - Save as confusion_matrix.png
    - mlflow.log_artifact('confusion_matrix.png')
  Return: {accuracy, macro_f1, per_class_f1: dict}
  ```

## ⚡ Model Optimization
- [ ] **Apply dynamic quantization to best model**
  ```
  File: src/optimize/quantize.py
  Steps:
    model = torch.load('models/best_model.pth')
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    torch.save(quantized, 'models/best_model_quantized.pth')
  Log: file size before/after, accuracy drop
  Target: < 10MB model file for serving
  ```
- [ ] **Benchmark inference latency**
  ```
  File: src/optimize/benchmark.py
  Test: 100 forward passes on a random (3,224,224) tensor
  Report: mean ± std latency in ms
  Test on: CPU (required), CUDA if available
  Log to MLflow: mlflow.log_metric('inference_latency_ms', mean_ms)
  Target: < 50ms per frame (leaves headroom for 200ms end-to-end budget)
  ```

---

# 📈 6. EXPERIMENT TRACKING (MLflow)

- [ ] **Configure MLflow tracking server**
  ```
  File: src/config.py
  MLFLOW_TRACKING_URI = 'http://localhost:5000'
  MLFLOW_EXPERIMENT_NAME = 'ASL-Recognition'

  Start server:
  mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
  ```
- [ ] **Track params, metrics, and artifacts beyond autolog**
  ```python
  # In src/train.py — manual logging additions:
  mlflow.log_params({'model': args.model, 'lr': args.lr, 'batch_size': args.batch_size})
  mlflow.log_metrics({'val_f1': f1, 'inference_ms': lat, 'model_size_mb': size})
  mlflow.log_artifact('data/baseline_stats.json')
  mlflow.log_artifact('confusion_matrix.png')
  mlflow.set_tag('git_commit', subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip())
  mlflow.set_tag('dvc_stage', 'train')
  ```
- [ ] **Register best model in MLflow Model Registry**
  ```python
  # After training:
  mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="ASL-Classifier")
  # Transition via UI: Staging → Production
  # Load in FastAPI:
  model = mlflow.pytorch.load_model("models:/ASL-Classifier/Production")
  ```
- [ ] **Compare experiments in MLflow UI**
  - Open `http://localhost:5000`
  - Select multiple runs → Compare → review metrics table
  - Identify best run by `val_f1` + `inference_latency_ms`

---

# 🔁 7. CI/CD + VERSION CONTROL

## DVC Pipeline
- [ ] **Define DVC pipeline DAG in `dvc.yaml`**
  ```yaml
  # File: dvc.yaml
  stages:
    ingest:
      cmd: python src/data/download.py
      deps: [src/data/download.py]
      outs: [data/raw/]
    validate:
      cmd: python src/data/validate.py
      deps: [src/data/validate.py, data/raw/]
      outs: [data/validated/]
    preprocess:
      cmd: python src/data/preprocess.py
      deps: [src/data/preprocess.py, data/validated/]
      outs: [data/processed/]
    train:
      cmd: python src/train.py --model mobilenet
      deps: [src/train.py, data/processed/]
      outs: [models/best_model.pth]
      metrics: [metrics/eval.json]
  ```
  ```bash
  dvc repro         # run full pipeline
  dvc dag           # visualize DAG
  dvc params diff   # compare params across commits
  ```
- [ ] **Set up GitHub Actions CI workflow**
  ```yaml
  # File: .github/workflows/ci.yml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with: { python-version: '3.10' }
        - run: pip install -r requirements.txt
        - run: flake8 src/ && black --check src/
        - run: pytest tests/ -v
        - run: dvc repro
  ```
- [ ] **Ensure reproducibility: link Git commit to MLflow run**
  ```python
  # In src/train.py:
  import subprocess
  git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
  mlflow.set_tag('git_commit', git_hash)
  # To reproduce any run: git checkout <hash> && dvc repro
  ```

---

# 🚀 8. MODEL SERVING (FastAPI)

## API Endpoints
- [ ] **Build `/predict` endpoint**
  ```
  File: backend/app/routes/predict.py
  Method: POST /predict
  Input: multipart/form-data (image file) OR JSON { "image_b64": "..." }
  Output:
    {
      "label": "A",
      "confidence": 0.97,
      "top3": [{"label":"A","conf":0.97}, {"label":"B","conf":0.02}, ...],
      "latency_ms": 42
    }
  Validation: check image size, format (JPEG/PNG only)
  Logging: append to logs/predictions.jsonl per request
  ```
- [ ] **Build `/health` and `/ready` endpoints**
  ```
  File: backend/app/routes/health.py

  GET /health
    → Always returns 200: { "status": "ok" }
    → Used for liveness probe

  GET /ready
    → Returns 200 if model is loaded: { "status": "ready", "model_version": "1.2" }
    → Returns 503 if model not yet loaded: { "status": "not_ready" }
    → Used for readiness probe (Docker healthcheck, orchestrator)
  ```
- [ ] **Implement Pydantic schemas**
  ```python
  # File: backend/app/schemas.py
  class PredictResponse(BaseModel):
      label: str
      confidence: float
      top3: list[dict]
      latency_ms: float

  class HealthResponse(BaseModel):
      status: str

  class ReadyResponse(BaseModel):
      status: str
      model_version: str | None
  # These ARE your LLD API spec — copy into docs/lld.md
  ```
- [ ] **Load model from MLflow registry on startup (not per request)**
  ```python
  # File: backend/app/main.py
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Startup: load once, cache globally
      app.state.model = mlflow.pytorch.load_model("models:/ASL-Classifier/Production")
      app.state.model.eval()
      logger.info(f"Model loaded successfully")
      yield
      # Shutdown: cleanup if needed

  app = FastAPI(lifespan=lifespan)
  ```

## Logging & Error Handling
- [ ] **Add centralized structured logging**
  ```
  File: backend/app/utils/logger.py
  Use: Python logging + loguru (pip install loguru)
  Log per request: timestamp, endpoint, input_shape, predicted_label, confidence, latency_ms
  Rotate: logs/app.log (10MB max, keep 5 backups)
  Separate: logs/errors.log for exceptions only
  Format: JSON lines for easy parsing
  ```
- [ ] **Implement comprehensive exception handling**
  ```
  File: backend/app/middleware/error_handler.py
  Custom exceptions:
    - InvalidImageError → HTTP 422 { "error": "invalid_image", "message": "..." }
    - ModelNotLoadedError → HTTP 503 { "error": "model_unavailable", "message": "..." }
    - NullPredictionError → HTTP 200 { "label": null, "message": "No hand detected" }
  All exceptions: logged with full stack trace to logs/errors.log
  Add FastAPI exception_handler for each custom exception
  ```

---

# 🖥 9. FRONTEND DEVELOPMENT

## Core Components
- [ ] **Build real-time webcam prediction component**
  ```
  File: frontend/src/components/WebcamFeed.jsx
  Library: react-webcam
  Logic:
    - Capture frame every ~150ms using setInterval
    - POST frame to /api/predict (as Blob or base64)
    - Display: large predicted letter, confidence % bar, top-3 alternatives
  Handle:
    - Camera permission denied → show instructions
    - No hand detected (null label) → show "Show your hand to the camera"
    - Slow network → debounce requests, show loading spinner
  ```
- [ ] **Build image upload prediction component**
  ```
  File: frontend/src/components/ImageUpload.jsx
  Features:
    - Drag-and-drop zone + file picker button
    - Preview uploaded image before sending
    - POST to /api/predict on submit
    - Show result as overlay on the image (label + confidence badge)
    - Clear button to reset
  ```
- [ ] **Build ML Pipeline visualization screen**
  ```
  File: frontend/src/pages/PipelineDashboard.jsx
  Content:
    - Embed MLflow UI via iframe (http://localhost:5000) OR fetch /api/experiments
    - Show last N experiment runs: run name, val_accuracy, F1, latency
    - Link to Grafana dashboard (http://localhost:3000)
    - Display DVC DAG as a static diagram (screenshot or Mermaid)
  ```
- [ ] **Write user manual**
  ```
  File: docs/user_manual.md
  Sections:
    1. What is this app (1 paragraph, plain English)
    2. How to open the app (URL, browser requirements)
    3. How to use the webcam feature (step-by-step with screenshots)
    4. How to upload an image (step-by-step)
    5. What the output means (label, confidence, top-3)
    6. Troubleshooting:
       - "No hand detected" → move hand closer, improve lighting
       - Slow response → check backend is running
       - Camera not working → allow browser permissions
  ```

---

# 🔗 10. SYSTEM ARCHITECTURE

## Architecture Diagram
- [ ] **Create architecture diagram**
  ```
  File: docs/architecture.png + docs/architecture.md
  Tool: draw.io, Excalidraw, or Mermaid

  Blocks to include:
    [User Browser]
       ↓ HTTP (port 80)
    [React Frontend — Docker container]
       ↓ REST API calls (port 8000)
    [FastAPI Backend — Docker container]
       ↓ loads model
    [MLflow Model Registry (port 5000)]
       ↓ metrics scrape (port 9090)
    [Prometheus — Docker container]
       ↓ visualize
    [Grafana — Docker container (port 3000)]

  Label: ports, data flow direction, docker network boundary
  ```

## HLD & LLD
- [ ] **Write HLD document**
  ```
  File: docs/hld.md
  Sections:
    1. System overview
    2. Design choices and rationale (why MobileNet? why FastAPI? why MLflow?)
    3. Component responsibilities
    4. Data flow (raw image → prediction)
    5. Tech stack table
  Reference architecture diagram
  ```
- [ ] **Write LLD document with full API specs**
  ```
  File: docs/lld.md
  For EACH endpoint document:
    - Method + path
    - Request: content-type, fields, types, constraints
    - Response: fields, types, example JSON
    - HTTP status codes and error responses
  Include: sequence diagram for /predict call flow
  Base on: backend/app/schemas.py
  ```

---

# 📦 11. DOCKERIZATION

- [ ] **Write backend Dockerfile**
  ```dockerfile
  # File: backend/Dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] **Write frontend Dockerfile**
  ```dockerfile
  # File: frontend/Dockerfile
  FROM node:18-alpine AS builder
  WORKDIR /app
  COPY package*.json .
  RUN npm ci
  COPY . .
  RUN npm run build

  FROM nginx:alpine
  COPY --from=builder /app/dist /usr/share/nginx/html
  COPY nginx.conf /etc/nginx/conf.d/default.conf
  EXPOSE 80
  # nginx.conf: proxy /api → backend:8000
  ```
- [ ] **Write `docker-compose.yml` with all services**
  ```yaml
  # File: docker-compose.yml
  version: '3.8'
  services:
    frontend:
      build: ./frontend
      ports: ["80:80"]
      depends_on: [backend]

    backend:
      build: ./backend
      ports: ["8000:8000"]
      depends_on: [mlflow]
      environment:
        MLFLOW_TRACKING_URI: http://mlflow:5000

    mlflow:
      image: ghcr.io/mlflow/mlflow:latest
      ports: ["5000:5000"]
      volumes: [mlflow_data:/mlflow]
      command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow/mlflow.db

    prometheus:
      image: prom/prometheus:latest
      ports: ["9090:9090"]
      volumes: [./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml]

    grafana:
      image: grafana/grafana:latest
      ports: ["3000:3000"]
      volumes: [grafana_data:/var/lib/grafana]

  volumes:
    mlflow_data:
    grafana_data:
  ```
  ```bash
  docker compose up -d        # start all services
  docker compose logs backend # check backend logs
  docker compose down         # stop all
  ```

---

# 📡 12. MONITORING (Prometheus + Grafana)

## Instrumentation
- [ ] **Add Prometheus metrics to FastAPI**
  ```
  File: backend/app/metrics.py
  Library: prometheus-client (pip install prometheus-client)

  Metrics to define:
    predict_requests_total     → Counter (label: status=success|error)
    predict_latency_seconds    → Histogram (buckets: .05, .1, .2, .5, 1.0)
    model_confidence_gauge     → Gauge (last prediction confidence)
    predict_errors_total       → Counter (label: error_type)

  Expose: GET /metrics endpoint
  Add to main.py: app.add_route('/metrics', make_asgi_app())
  ```
- [ ] **Implement drift detection metric**
  ```
  File: monitoring/drift_detector.py
  Class: DriftDetector(baseline_path='data/baseline_stats.json')
  Method: compute_drift(image_batch: np.ndarray) -> dict
    - Compute histogram of incoming batch pixel values
    - Compare vs saved baseline using KL divergence or JS distance
    - Return: {'drift_score': float, 'alert': bool}
  Threshold: alert if drift_score > 0.15
  Expose as Prometheus gauge: data_drift_score
  Call in /predict route: update gauge after each prediction
  ```
- [ ] **Configure Prometheus scrape targets**
  ```yaml
  # File: monitoring/prometheus.yml
  global:
    scrape_interval: 15s
  scrape_configs:
    - job_name: 'fastapi'
      static_configs:
        - targets: ['backend:8000']
      metrics_path: /metrics
    - job_name: 'node-exporter'
      static_configs:
        - targets: ['node-exporter:9100']
  ```
- [ ] **Build Grafana dashboard**
  ```
  File: monitoring/grafana/dashboards/asl_dashboard.json
  Panels to create:
    1. Request rate (req/s) — PromQL: rate(predict_requests_total[1m])
    2. P95 latency (ms)    — histogram_quantile(0.95, predict_latency_seconds_bucket)
    3. Error rate (%)      — rate(predict_errors_total[5m]) / rate(predict_requests_total[5m])
    4. Avg confidence      — model_confidence_gauge
    5. Drift score         — data_drift_score

  Alerts:
    - error_rate > 0.05 → send alert
    - drift_score > 0.15 → send alert

  Export JSON → save to monitoring/grafana/dashboards/
  Enable Grafana provisioning to auto-load on container start
  ```

---

# 🔁 13. RETRAINING PIPELINE

- [ ] **Implement retraining trigger logic**
  ```
  File: monitoring/retrain_trigger.py
  Triggers:
    - drift_score > threshold (from DriftDetector)
    - val_accuracy drops > 5% from baseline
  Action: POST /api/retrain OR run: dvc repro
  Log trigger event to MLflow: mlflow.set_tag('trigger', 'drift_detected')
  Version new model: bump MLflow model version
  ```

---

# 🧪 14. TESTING

- [ ] **Write API unit tests**
  ```
  File: tests/test_api.py
  Library: pytest + fastapi.testclient

  Test cases:
    test_predict_valid_image()     → 200 + valid label in response
    test_predict_no_image()        → 422 unprocessable
    test_predict_corrupt_file()    → 422 with error message
    test_predict_wrong_format()    → 422 (e.g., send a .txt file)
    test_health_endpoint()         → 200 { "status": "ok" }
    test_ready_endpoint()          → 200 or 503
  ```
  ```bash
  pytest tests/test_api.py -v
  ```
- [ ] **Write model unit tests**
  ```
  File: tests/test_model.py

  Test cases:
    test_output_shape()            → output shape == (batch_size, 29)
    test_confidence_sums_to_one()  → softmax output sums to ~1.0
    test_quantized_accuracy_drop() → quantized model within 1% of original
    test_inference_latency()       → mean latency < 200ms on CPU
    test_no_hand_returns_none()    → LandmarkExtractor returns None for blank image
  ```
- [ ] **Write data pipeline unit tests**
  ```
  File: tests/test_data.py

  Test cases:
    test_validate_rejects_corrupt_image()
    test_validate_rejects_missing_label()
    test_preprocess_output_shape()         → tensor shape (3, 224, 224)
    test_split_ratios_sum_to_one()
    test_baseline_stats_has_all_classes()  → 29 keys in baseline_stats.json
    test_augmentation_does_not_change_label()
  ```
- [ ] **Write test plan and test report**
  ```
  File: docs/test_plan.md
  Sections:
    1. Scope (what is being tested)
    2. Strategy (unit / integration / acceptance)
    3. Test cases table: ID | Input | Expected Output | Actual | Pass/Fail
    4. Acceptance criteria:
       - val_accuracy > 90%
       - inference_latency < 200ms
       - all API tests passing
       - error rate < 5% under load

  File: docs/test_report.md
  Fill after running: total cases, passed, failed, notes
  ```

---

# 🧾 15. LOGGING & ERROR HANDLING

- [ ] **Set up centralized structured logging**
  ```
  File: backend/app/utils/logger.py
  Use: loguru
  Log per request: {timestamp, endpoint, input_shape, label, confidence, latency_ms}
  Rotate: logs/app.log (10MB, 5 backups)
  Separate: logs/errors.log (exceptions + stack traces)
  Format: JSON lines
  ```
- [ ] **Implement global exception handler middleware**
  ```
  File: backend/app/middleware/error_handler.py
  Handle:
    InvalidImageError   → 422 + log warning
    ModelNotLoadedError → 503 + log critical
    NullPredictionError → 200 + null label (not an error, just no hand)
    Exception (catch-all) → 500 + log error with traceback
  ```

---

# 📚 16. DOCUMENTATION CHECKLIST

| Document | File | Status |
|---|---|---|
| Architecture Diagram | `docs/architecture.png` | - |
| HLD | `docs/hld.md` | - |
| LLD (API specs) | `docs/lld.md` | - |
| Test Plan | `docs/test_plan.md` | - |
| Test Report | `docs/test_report.md` | - |
| User Manual | `docs/user_manual.md` | - |
| Feature Spec | `docs/feature_spec.md` | - |

---

# 📁 RECOMMENDED PROJECT STRUCTURE

```
asl-recognition/
├── dags/                        # Airflow DAGs
│   └── ingest_dag.py
├── data/
│   ├── raw/                     # original dataset (DVC tracked)
│   ├── processed/               # train/val/test splits (DVC tracked)
│   ├── baseline_stats.json      # for drift detection
│   └── splits.json
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── baseline_stats.py
│   ├── features/
│   │   ├── landmarks.py
│   │   ├── pixels.py
│   │   └── bg_remove.py
│   ├── models/
│   │   ├── mobilenet.py
│   │   ├── tiny_cnn.py
│   │   └── landmark_mlp.py
│   ├── optimize/
│   │   ├── quantize.py
│   │   └── benchmark.py
│   ├── utils/
│   │   └── training_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       ├── routes/
│       │   ├── predict.py
│       │   └── health.py
│       ├── middleware/
│       │   └── error_handler.py
│       └── utils/
│           └── logger.py
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
│       ├── components/
│       │   ├── WebcamFeed.jsx
│       │   └── ImageUpload.jsx
│       └── pages/
│           └── PipelineDashboard.jsx
├── monitoring/
│   ├── prometheus.yml
│   ├── drift_detector.py
│   ├── retrain_trigger.py
│   └── grafana/
│       └── dashboards/
│           └── asl_dashboard.json
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_data.py
├── docs/
│   ├── architecture.md
│   ├── hld.md
│   ├── lld.md
│   ├── test_plan.md
│   ├── test_report.md
│   ├── user_manual.md
│   └── feature_spec.md
├── logs/
├── models/
├── .github/
│   └── workflows/
│       └── ci.yml
├── dvc.yaml
├── params.yaml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🎤 17. VIVA PREPARATION

- [ ] Trace a full prediction request end-to-end: webcam frame → FastAPI → MLflow model → Prometheus metric → Grafana panel
- [ ] Justify model choice (MobileNetV2 vs TinyCNN vs MLP)
- [ ] Explain how Git commit hash links to an MLflow run for reproducibility
- [ ] Explain DVC DAG and how `dvc repro` works
- [ ] Discuss challenges: lighting variation, real-time latency budget (50ms inference + 30ms preprocess + 20ms network + 100ms frontend = 200ms)
- [ ] Be ready to show MLflow UI and walk through an experiment comparison
- [ ] Be ready to show Grafana dashboard with live metrics

---

# ⭐ BONUS

- [ ] Sentence/word builder (accumulate letters over time)
- [ ] Text-to-speech output (Web Speech API)
- [ ] Gesture smoothing (majority vote over last 5 frames)
- [ ] Grad-CAM explainability visualization
- [ ] Multilingual support

---

# 🚀 FINAL CHECKLIST

- [ ] `docker compose up -d` starts all 5 services with no errors
- [ ] `/predict` responds in < 200ms
- [ ] MLflow UI shows at least 3 experiment runs compared
- [ ] Grafana dashboard shows live request metrics
- [ ] All pytest tests passing
- [ ] All 7 documentation files complete
- [ ] DVC DAG visualized (`dvc dag`)
- [ ] Confusion matrix artifact saved in MLflow
```

Everything is there — file paths, class names, code snippets, CLI commands, and the full folder structure at the bottom. Good luck with the project!