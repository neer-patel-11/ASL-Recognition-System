"""
Shared pytest fixtures for the ASL Recognition System test suite.
"""

import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Set env vars BEFORE importing app so they are picked up ──
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "admin123")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient shared across the session."""
    from app.main import app  # noqa: PLC0415
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def admin_auth():
    """HTTP Basic credentials for the admin user."""
    return ("admin", "asl_admin_2024")


@pytest.fixture()
def bad_auth():
    """HTTP Basic credentials that should be rejected."""
    return ("hacker", "wrongpassword")


@pytest.fixture()
def sample_image_bytes():
    """Return raw bytes for a synthetic 224×224 RGB image."""
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def tiny_image_bytes():
    """Return raw bytes for a very small (8×8) image — edge-case test."""
    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def grayscale_image_bytes():
    """Return raw bytes for a grayscale image — should be converted to RGB."""
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    img = Image.fromarray(arr, "L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()