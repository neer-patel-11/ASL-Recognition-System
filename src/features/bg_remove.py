
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def remove_bg_rembg(
    source: Union[str, Path, np.ndarray, Image.Image],
    replace_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Remove background using rembg (U2-Net deep learning model).
    Best quality but slower (~0.5s per image on CPU).

    Args:
        source: image path, ndarray, or PIL.Image
        replace_color: RGB color to fill background (default white)
    Returns:
        np.ndarray shape (H, W, 3) with background replaced
    """
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        raise ImportError("rembg not installed. Run: pip install rembg")

    # Load to PIL
    if isinstance(source, (str, Path)):
        img = Image.open(str(source)).convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source).convert("RGB")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        raise TypeError(f"Unsupported type: {type(source)}")

    # Remove background → RGBA output
    result_rgba = rembg_remove(img)                # PIL RGBA image
    result_arr  = np.array(result_rgba)            # shape (H, W, 4)

    # Composite over replace_color using alpha channel
    alpha   = result_arr[:, :, 3:4] / 255.0        # normalized alpha (H,W,1)
    fg      = result_arr[:, :, :3].astype(float)   # foreground RGB
    bg      = np.full_like(fg, replace_color, dtype=float)
    blended = (alpha * fg + (1 - alpha) * bg).clip(0, 255).astype(np.uint8)

    return blended


def remove_bg_grabcut(
    source: Union[str, Path, np.ndarray, Image.Image],
    iterations: int = 5,
    replace_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Remove background using OpenCV GrabCut.
    No extra dependencies — works on CPU only.
    Less accurate than rembg for complex backgrounds.

    Args:
        source: image path, ndarray (BGR), or PIL.Image
        iterations: GrabCut iterations (more = better quality, slower)
        replace_color: BGR color to fill background (default white)
    Returns:
        np.ndarray shape (H, W, 3) BGR with background replaced
    """
    # Load as BGR ndarray
    if isinstance(source, (str, Path)):
        img_bgr = cv2.imread(str(source))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {source}")
    elif isinstance(source, np.ndarray):
        img_bgr = source
    elif isinstance(source, Image.Image):
        img_bgr = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported type: {type(source)}")

    h, w = img_bgr.shape[:2]

    # Define rect: slight margin from edges (assumes hand is centered)
    margin = max(10, min(h, w) // 10)
    rect   = (margin, margin, w - 2 * margin, h - 2 * margin)

    mask   = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterations,
                cv2.GC_INIT_WITH_RECT)

    # Pixels labeled GC_FGD or GC_PR_FGD → foreground
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    # Apply mask
    result = img_bgr.copy()
    bg_color_bgr = (replace_color[2], replace_color[1], replace_color[0])
    bg_fill = np.full_like(img_bgr, bg_color_bgr)
    result  = np.where(fg_mask[:, :, np.newaxis] == 1, img_bgr, bg_fill)

    return result.astype(np.uint8)


class BackgroundRemover:
    """
    Unified interface for background removal.
    Automatically falls back to GrabCut if rembg is unavailable.

    Usage:
        remover = BackgroundRemover(method="rembg")
        clean_img = remover.remove("path/to/image.jpg")
    """

    METHODS = ("rembg", "grabcut")

    def __init__(self, method: str = "rembg"):
        assert method in self.METHODS, f"method must be one of {self.METHODS}"
        self.method = method
        logger.info(f"BackgroundRemover initialized: method={method}")

    def remove(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        replace_color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """
        Remove background from image.
        Returns np.ndarray (H, W, 3).
        Falls back to GrabCut if rembg fails.
        """
        if self.method == "rembg":
            try:
                return remove_bg_rembg(source, replace_color)
            except ImportError:
                logger.warning("rembg not available, falling back to GrabCut")
                return remove_bg_grabcut(source, replace_color=replace_color)
            except Exception as e:
                logger.warning(f"rembg failed ({e}), falling back to GrabCut")
                return remove_bg_grabcut(source, replace_color=replace_color)

        return remove_bg_grabcut(source, replace_color=replace_color)

    def remove_batch(self, sources: list) -> list:
        """Process a list of images. Returns list of ndarray results."""
        results = []
        for i, src in enumerate(sources):
            try:
                results.append(self.remove(src))
            except Exception as e:
                logger.warning(f"bg_remove failed for item {i}: {e}")
                results.append(None)
        success = sum(1 for r in results if r is not None)
        logger.info(f"bg_remove batch: {success}/{len(sources)} succeeded")
        return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python bg_remove.py <image_path> [rembg|grabcut]")
        sys.exit(1)

    path   = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "rembg"

    remover = BackgroundRemover(method=method)
    output  = remover.remove(path)

    out_path = Path(path).stem + f"_nobg_{method}.jpg"
    cv2.imwrite(out_path, output if method == "grabcut"
                else cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}  shape={output.shape}")