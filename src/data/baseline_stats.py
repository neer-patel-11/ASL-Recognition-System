# File: src/data/baseline_stats.py
import os
import json
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_baseline(processed_dir: str = "data/processed/train") -> dict:
    """
    Compute per-class pixel mean, variance, and histogram.
    Saves to data/baseline_stats.json for drift detection later.
    """
    stats = {}
    all_means, all_vars = [], []

    classes = sorted(os.listdir(processed_dir))

    for label in classes:
        label_dir = os.path.join(processed_dir, label)
        if not os.path.isdir(label_dir):
            continue

        pixel_values = []
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = Image.open(os.path.join(label_dir, fname)).convert("RGB")
                img = img.resize((224, 224))
                arr = np.array(img).flatten().astype(np.float32) / 255.0
                pixel_values.append(arr)
            except Exception:
                continue

        if not pixel_values:
            continue

        pixels = np.stack(pixel_values)
        mean = float(np.mean(pixels))
        var = float(np.var(pixels))
        hist, bin_edges = np.histogram(pixels.flatten(), bins=32, range=(0, 1))

        stats[label] = {
            "mean": round(mean, 6),
            "variance": round(var, 6),
            "histogram": hist.tolist(),
            "bin_edges": [round(e, 4) for e in bin_edges.tolist()],
            "sample_count": len(pixel_values),
        }
        all_means.append(mean)
        all_vars.append(var)
        logger.info(f"  {label}: mean={mean:.4f}, var={var:.4f}, n={len(pixel_values)}")

    stats["_global"] = {
        "mean": round(float(np.mean(all_means)), 6),
        "variance": round(float(np.mean(all_vars)), 6),
    }

    out_path = "data/baseline_stats.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Baseline stats saved to {out_path}")
    return stats


if __name__ == "__main__":
    compute_baseline()