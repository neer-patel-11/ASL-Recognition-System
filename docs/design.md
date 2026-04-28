# Design Document — ASL Recognition System

## 1. Design Overview

This project follows a **modular, service-oriented architecture** for building an end-to-end ASL recognition system with MLOps integration.

The system is divided into the following independent components:

- Data Pipeline (Airflow)
- Training Pipeline (DVC + MLflow)
- Model Serving (MLflow Model Server)
- Backend API (FastAPI)
- Frontend UI (HTML/JS)
- Monitoring Stack (Prometheus, Grafana, Alertmanager)

---

## 2. Design Paradigm

The system follows a **hybrid design approach**:

### Functional Paradigm
- Data pipelines (Airflow DAGs, DVC stages)
- Training scripts (`train.py`, feature pipelines)

### Object-Oriented Paradigm
- Model definitions (CNN, MLP classes)
- API service structure (FastAPI routers, dependency injection)

---

## 3. High-Level Design (HLD)

The system is structured into five logical layers:

### 1. Data Layer
- Dataset ingestion via Airflow
- Storage using DVC

### 2. Training Layer
- Feature engineering
- Model training (CNN, MLP, traditional ML)
- Experiment tracking with MLflow

### 3. Serving Layer
- MLflow Model Server (port 5001)
- FastAPI Backend (port 8000)

### 4. Monitoring Layer
- Prometheus (metrics collection)
- Grafana (visualization)
- Alertmanager (alerts)

### 5. Orchestration Layer
- Docker Compose (multi-service deployment)

---

## 4. Architecture Diagram

Refer to the report:

- Data + Training Pipeline
- Model Serving Architecture
- Monitoring Architecture

---

## 5. Low-Level Design (LLD)

Detailed API specifications are provided in:

👉 `API_SPEC.md`

---

## 6. Loose Coupling Design

The system strictly enforces **loose coupling** between components:

### UI ↔ Backend
- Communication via REST API (`/predict`)
- No shared state or direct dependency

### Backend ↔ Model Server
- Backend calls MLflow server via HTTP
- Model is not embedded in FastAPI

### Benefits
- Independent deployment
- Easy model swapping
- Scalability (can move model server to separate machine)

---

## 7. Design Decisions

### Why MLflow Model Server?
- Standardized inference interface
- Model versioning + deployment

### Why DVC?
- Data + model reproducibility
- Pipeline DAG tracking

### Why FastAPI?
- High performance (async)
- Built-in OpenAPI support

---

## 8. Scalability Considerations

- Stateless API → horizontally scalable
- Model server can be replicated
- Monitoring supports multi-instance scraping

---

## 9. Limitations

- No message queue (synchronous inference)
- Single-node deployment (Docker Compose)
- No autoscaling

---

## 10. Future Improvements

- Kubernetes deployment
- Async inference pipeline (Kafka / Celery)
- Model A/B testing
- CI/CD integration