# File: src/features/run_feature_check.py
"""
DVC stage entry point for feature pipeline validation.
Tests landmark + pixel extraction on a sample image per class.
Writes: data/feature_check_report.json
NOT meant to process all images — Airflow does that during training.
"""
import os
import sys
import json
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.landmarks import LandmarkExtractor
from features.pixels    import PixelExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_TRAIN_DIR = "data/processed/train"
OUTPUT_REPORT       = "data/feature_check_report.json"


def get_one_image_per_class(train_dir: str) -> dict:
    """Return {label: first_image_path} for every class folder found."""
    samples = {}
    if not os.path.exists(train_dir):
        logger.warning(f"Train dir not found: {train_dir}")
        return samples

    for label in sorted(os.listdir(train_dir)):
        label_dir = os.path.join(train_dir, label)
        if not os.path.isdir(label_dir):
            continue
        images = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            samples[label] = os.path.join(label_dir, images[0])

    return samples


def main():
    samples = get_one_image_per_class(PROCESSED_TRAIN_DIR)

    if not samples:
        raise RuntimeError(
            f"No images found in {PROCESSED_TRAIN_DIR}. "
            "Run the Airflow ingestion DAG first."
        )

    logger.info(f"Found {len(samples)} classes to test")

    landmark_extractor = LandmarkExtractor()
    pixel_extractor    = PixelExtractor(mode="eval")

    report = {
        "total_classes":      len(samples),
        "landmark_detected":  0,
        "landmark_failed":    0,
        "pixel_success":      0,
        "pixel_failed":       0,
        "classes":            {},
        "landmark_output_shape": "(63,)",
        "pixel_output_shape":    "(3, 224, 224)",
    }

    start = time.perf_counter()

    for label, img_path in samples.items():
        result = {"image": img_path}

        # Test landmark extraction
        try:
            features = landmark_extractor.extract(img_path)
            if features is not None:
                result["landmark"] = "detected"
                result["landmark_shape"] = str(features.shape)
                report["landmark_detected"] += 1
            else:
                result["landmark"] = "no_hand_detected"
                report["landmark_failed"] += 1
        except Exception as e:
            result["landmark"] = f"error: {e}"
            report["landmark_failed"] += 1

        # Test pixel extraction
        try:
            tensor = pixel_extractor.extract(img_path)
            result["pixel"] = "ok"
            result["pixel_shape"] = str(tuple(tensor.shape))
            report["pixel_success"] += 1
        except Exception as e:
            result["pixel"] = f"error: {e}"
            report["pixel_failed"] += 1

        report["classes"][label] = result

    elapsed = time.perf_counter() - start
    report["elapsed_sec"]        = round(elapsed, 2)
    report["landmark_detect_rate"] = round(
        report["landmark_detected"] / len(samples) * 100, 1
    )

    # Save report
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info(f"Classes tested       : {report['total_classes']}")
    logger.info(f"Landmark detected    : {report['landmark_detected']}")
    logger.info(f"Landmark failed      : {report['landmark_failed']}")
    logger.info(f"Landmark detect rate : {report['landmark_detect_rate']}%")
    logger.info(f"Pixel success        : {report['pixel_success']}")
    logger.info(f"Elapsed              : {elapsed:.2f}s")
    logger.info(f"Report saved to      : {OUTPUT_REPORT}")
    logger.info("=" * 50)

    # Fail if pixel extraction broken (landmarks can fail on static images)
    if report["pixel_failed"] > 0:
        raise RuntimeError(
            f"Pixel extraction failed for {report['pixel_failed']} classes. "
            "Check logs above."
        )


if __name__ == "__main__":
    main()