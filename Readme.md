
# ASL Recognition System with End-to-End MLOps Pipeline

```
Name :- Neer Patel
Roll no :- DA25M021
```

## Overview

This project implements an end-to-end **American Sign Language (ASL) recognition system** that classifies hand gesture images into **27 classes (A–Z + NIL)**.

It is designed as a complete **MLOps pipeline**, covering:
- Data ingestion (Airflow)
- Data & model versioning (DVC)
- Training & experiment tracking (MLflow)
- Model serving (FastAPI + MLflow)
- Monitoring & alerting (Prometheus, Grafana, Alertmanager)

---

## Key Features

- End-to-end reproducible ML pipeline
- Automated data ingestion with Airflow
- Experiment tracking with MLflow
- Model versioning using DVC
- Real-time inference API (FastAPI)
- Monitoring with Prometheus & Grafana
- Alerting via Alertmanager (email)
- Docker-based deployment

---

## Project Structure

```

asl-recognition/
├── dags/                     # Airflow DAGs
│   └── ingest_dag.py
├── src/
│   ├── data/                # Data pipeline scripts
│   ├── features/            # Feature engineering
│   ├── models/              # Model definitions
│   ├── utils/               # Training utilities
│   └── train.py             # Training entry point
├── app/                     # FastAPI backend
├── tests/                   # Unit & integration tests
├── data/
│   ├── raw/
│   └── processed/
├── models/                  # Trained models
├── monitoring/              # Prometheus + Alertmanager configs
├── docker-compose.api.yml
├── docker-compose.airflow.yml
├── dvc.yaml
├── params.yaml
└── README.md

````

---

## Dataset

- Source: Kaggle
- Dataset: `mayank0bhardwaj/alphabets`
- Classes: 27 (A–Z + NIL)

---

## Model Performance

### Best Model: Tiny-CNN

| Metric        | Value   |
|--------------|--------|
| Accuracy      | 35.46% |
| Macro F1      | 0.3157 |

### Per-Class F1 (Selected)

| Class | F1 Score |
|------|--------|
| Q    | 0.6667 |
| C    | 0.6415 |
| NIL  | 0.6000 |
| P    | 0.5806 |
| Z    | 0.5172 |
| U    | 0.0000 |

---

## Tech Stack

| Layer            | Tool |
|------------------|-----|
| API              | FastAPI |
| Training         | PyTorch |
| Tracking         | MLflow |
| Orchestration    | Airflow |
| Versioning       | DVC |
| Monitoring       | Prometheus |
| Visualization    | Grafana |
| Alerting         | Alertmanager |
| Containerization | Docker |

---

## Pipeline Overview

### Data Pipeline (Airflow)
1. Download dataset
2. Validate images
3. Split dataset (70/15/15)
4. Compute baseline stats
5. Track with DVC
6. Send email report

### Training Pipeline (DVC)
- Feature extraction
- Train models:
  - CNN (best)
  - MLP
  - KNN / Random Forest
- Store metrics and artifacts

### Serving Pipeline
- FastAPI receives image
- Preprocessing (resize, normalize)
- Calls MLflow model server
- Returns prediction

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/neer-patel-11/ASL-Recognition-System
cd ASL-Recognition-System
````



### File: .env

```
KAGGLE_USERNAME=kaggle_usernmae
KAGGLE_KEY=kaggle_key

AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__FERNET_KEY=''
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__SECRET_KEY=supersecretkey

# SMTP — using Gmail as example (or any SMTP)
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_STARTTLS=True
AIRFLOW__SMTP__SMTP_SSL=False
AIRFLOW__SMTP__SMTP_USER=abc@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=password
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_MAIL_FROM=abc@gmail.com

ALERT_EMAIL=abc@gmail.com


ALERT_SENDER_EMAIL=abc@gmail.com
ALERT_EMAIL_PASSWORD=password
ALERT_RECIPIENT_EMAIL=abc@gmail.com

```
---

### 2. Initialize DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "dvc init"

dvc remote add -d localstore /path/to/dvc-store
git add .dvc/config
git commit -m "add dvc remote"
```

---

### 3. Start Airflow

```bash
sudo docker compose -f docker-compose.airflow.yml up airflow-init
sudo docker compose -f docker-compose.airflow.yml up -d
```

Access UI:

```
http://localhost:8080
username: admin
password: admin
```

Trigger DAG:

```
asl_data_ingestion
```

---

### 4. Run DVC Pipeline

```bash
dvc repro
```

---

### 5. Start MLflow Model Server

```bash
PYTHONPATH=src mlflow models serve \
  -m "mlartifacts/1/models/<model-id>/artifacts" \
  --host 0.0.0.0 \
  --port 5001 \
  --env-manager local
```

---

### 6. Start API + Monitoring

```bash
sudo docker compose -f docker-compose.api.yml up --build
```

---

## Services

| Service    | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| API        | [http://localhost:8000](http://localhost:8000)             |
| Admin      | [http://localhost:8000/admin](http://localhost:8000/admin) |
| Grafana    | [http://localhost:3000](http://localhost:3000)             |
| Prometheus | [http://localhost:9090](http://localhost:9090)             |
| Airflow    | [http://localhost:8080](http://localhost:8080)             |
| MLflow     | [http://localhost:5000](http://localhost:5000)             |

---

## API Usage

### Predict Endpoint

**POST /predict**

Input:

* Multipart form-data
* Field: `file` (image)

Output:

```json
{
  "label": "C",
  "confidence": 0.82,
  "latency_ms": 148.3
}
```

---

## Monitoring

### Metrics

* prediction_requests_total
* prediction_errors_total
* prediction_latency_seconds
* model_drift_score

### Alerts

* High error rate (>5%)
* High latency (p99 > 2s)
* Data drift (PSI > 0.3)
* Service down
* High CPU / memory usage

---

## Testing

Run tests:

```bash
pytest tests/
```

Or via API:

```
POST /admin/tests/run
```

---

## Known Limitations

* Landmark-based models perform poorly (~5% accuracy)
* CNN accuracy limited due to dataset quality
* Class imbalance affects performance
* No real-time video support
* No cloud deployment

---

### THANK YOU


