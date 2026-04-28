# API Specification (Low-Level Design)

## Base URL

```

[http://localhost:8000](http://localhost:8000)

````

---

## 1. Public Endpoints

### 1.1 Health Check

**GET /health**

Response:
```json
{
  "status": "ok"
}
````

---

### 1.2 Readiness Check

**GET /ready**

Response:

```json
{
  "ready": true
}
```

---

### 1.3 Predict Endpoint

**POST /predict**

Description:

* Accepts an image file
* Returns predicted ASL class

#### Request

* Content-Type: `multipart/form-data`
* Field: `file`

#### Response

```json
{
  "label": "C",
  "confidence": 0.82,
  "latency_ms": 148.3
}
```

#### Errors

```json
{
  "error": "Invalid image"
}
```

---

### 1.4 Metrics Endpoint

**GET /metrics**

* Returns Prometheus metrics
* Content-Type: text/plain

---

## 2. Admin Endpoints (Auth Required)

### 2.1 Admin UI

**GET /admin**

* Returns admin dashboard (HTML)

---

### 2.2 MLflow Experiments

**GET /admin/mlflow/experiments**

Response:

```json
[
  {
    "experiment_id": "1",
    "runs": [...]
  }
]
```

---

### 2.3 MLflow Models

**GET /admin/mlflow/models**

Response:

```json
[
  {
    "name": "cnn",
    "versions": [1, 2]
  }
]
```

---

### 2.4 Metrics Summary

**GET /admin/mlflow/metrics**

Response:

```json
{
  "accuracy": 0.3546,
  "macro_f1": 0.3157
}
```

---

## 3. Model Serving (MLflow)

### Endpoint

```
http://localhost:5001/invocations
```

### Request

```json
{
  "inputs": [...]
}
```

### Response

```json
{
  "predictions": [...]
}
```

---

## 4. Data Flow

1. Client uploads image → `/predict`
2. FastAPI preprocesses image
3. Sends request → MLflow model server
4. Receives prediction
5. Returns response to client

---

## 5. Error Handling

| Case              | Response |
| ----------------- | -------- |
| Invalid file      | 400      |
| Model unavailable | 500      |
| Timeout           | 504      |

---

## 6. Security

* Admin endpoints protected via HTTP Basic Auth
* No authentication for public inference (can be extended)

---

## 7. Performance

* Avg latency: ~150 ms
* Supports concurrent requests via FastAPI async

---

## 8. Extensibility

* Add new endpoints without breaking existing API
* Model endpoint configurable via environment variables
