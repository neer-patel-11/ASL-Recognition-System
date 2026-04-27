# app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import numpy as np
from PIL import Image
import io

app = FastAPI()

MLFLOW_URL = "http://host.docker.internal:5001/invocations"

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("app/static/index.html")


# ✅ CNN preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)   # batch

    return image.astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    data = preprocess_image(image_bytes)

    payload = {"inputs": data.tolist()}  # MLflow expects JSON

    response = requests.post(MLFLOW_URL, json=payload)

    return response.json()