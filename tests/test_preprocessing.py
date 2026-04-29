"""
Unit Tests — ASL Recognition System
Tests individual functions in isolation, no network calls.

Test Plan Reference: TC-U-001 through TC-U-015
"""

import io

import numpy as np
import pytest
from PIL import Image


# ── Helpers to import isolated functions ──────────────────────────────────────
def get_preprocess():
    from app.main import preprocess_image
    return preprocess_image


def get_psi():
    from app.main import compute_psi
    return compute_psi


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-001  preprocess_image — output shape
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_output_shape(sample_image_bytes):
    """Preprocessed image must have shape (1, 3, 224, 224)."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.shape == (1, 3, 224, 224), f"Unexpected shape: {result.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-002  preprocess_image — output dtype
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_output_dtype(sample_image_bytes):
    """Preprocessed output must be float32."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-003  preprocess_image — grayscale input auto-converted
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_grayscale_converted(grayscale_image_bytes):
    """Grayscale images must be converted to RGB — shape (1, 3, 224, 224)."""
    preprocess = get_preprocess()
    result = preprocess(grayscale_image_bytes)
    assert result.shape == (1, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-004  preprocess_image — tiny image resized correctly
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_tiny_image(tiny_image_bytes):
    """Even a tiny (8×8) image must be resized to (1, 3, 224, 224)."""
    preprocess = get_preprocess()
    result = preprocess(tiny_image_bytes)
    assert result.shape == (1, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-005  preprocess_image — pixel values normalised
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_pixel_range(sample_image_bytes):
    """After mean/std normalisation, pixel values should be in [-5, 5]."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.min() >= -5.0 and result.max() <= 5.0, (
        f"Pixel range out of bounds: [{result.min():.2f}, {result.max():.2f}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-006  preprocess_image — invalid bytes raise exception
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_invalid_bytes():
    """Garbage bytes must raise an exception (not silently produce output)."""
    preprocess = get_preprocess()
    with pytest.raises(Exception):
        preprocess(b"this is not an image")


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-007  compute_psi — identical distributions → PSI ≈ 0
# ─────────────────────────────────────────────────────────────────────────────
def test_psi_identical():
    """PSI should be near 0 when sample and baseline are identical."""
    psi = get_psi()
    result = psi(0.45, 0.45)
    assert result < 0.01, f"Expected near-zero PSI, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-008  compute_psi — large drift → PSI > 0.2
# ─────────────────────────────────────────────────────────────────────────────
def test_psi_large_drift():
    """PSI should be large when distributions differ significantly."""
    psi = get_psi()
    result = psi(0.9, 0.45)
    assert result > 0.2, f"Expected large PSI, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-009  compute_psi — always non-negative
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("s,b", [(0.1, 0.5), (0.5, 0.1), (0.45, 0.45), (0.0, 0.45)])
def test_psi_non_negative(s, b):
    """PSI must always be ≥ 0."""
    psi = get_psi()
    result = psi(s, b)
    assert result >= 0, f"PSI negative for s={s}, b={b}: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-010  compute_psi — epsilon prevents division by zero
# ─────────────────────────────────────────────────────────────────────────────
def test_psi_zero_inputs():
    """compute_psi must not raise ZeroDivisionError with zero inputs."""
    psi = get_psi()
    result = psi(0.0, 0.0)
    assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-011  app imports cleanly
# ─────────────────────────────────────────────────────────────────────────────
def test_app_imports():
    """The FastAPI app object must be importable without errors."""
    from app.main import app  # noqa: PLC0415
    assert app is not None


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-012  preprocess_image batch dimension
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_batch_dim(sample_image_bytes):
    """First dimension of output must be exactly 1 (batch size)."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-013  preprocess_image channels first
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_channels_first(sample_image_bytes):
    """Output must be channels-first (C=3 at dimension 1)."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.shape[1] == 3


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-014  preprocess_image spatial dimensions
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocess_spatial_dims(sample_image_bytes):
    """Spatial dimensions must be 224×224 regardless of input size."""
    preprocess = get_preprocess()
    result = preprocess(sample_image_bytes)
    assert result.shape[2] == 224
    assert result.shape[3] == 224


# ─────────────────────────────────────────────────────────────────────────────
# TC-U-015  admin credentials env var override
# ─────────────────────────────────────────────────────────────────────────────
def test_admin_env_vars():
    """ADMIN_USERNAME and ADMIN_PASSWORD env vars must be respected."""
    import os  # noqa: PLC0415
    assert os.environ.get("ADMIN_USERNAME") == "admin"
    assert os.environ.get("ADMIN_PASSWORD") == "admin"