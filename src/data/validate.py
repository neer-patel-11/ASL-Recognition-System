# File: src/data/validate.py
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)

VALID_LABELS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["NIL"])
MIN_RESOLUTION = (50, 50)


class DataValidationError(Exception):
    pass


def _find_image_root(base_path: str) -> str:
    """
    Walk base_path to find the folder that contains A-Z + NIL subfolders.
    Handles cases where kagglehub downloads into a nested version folder.
    """
    for root, dirs, _ in os.walk(base_path):
        letter_count = sum(1 for d in dirs if len(d.upper()) == 1 and d.upper().isalpha())
        if letter_count >= 20:
            return root
    return base_path  # fallback


def validate_dataset(dataset_path: str, train_csv_path: str = None) -> dict:
    """
    Validate all images in folder-based dataset.
    dataset_path: path returned by kagglehub (may be a version folder).
    train_csv_path: ignored — kept for interface compatibility.
    """
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": [],
        "class_counts": {},
        "invalid_labels": [],
        "corrupt_images": [],
        "missing_files": [],
    }

    if not os.path.isdir(dataset_path):
        raise DataValidationError(f"Dataset path not found: {dataset_path}")

    image_root = _find_image_root(dataset_path)
    logger.info(f"Image root resolved to: {image_root}")

    for label_folder in sorted(os.listdir(image_root)):
        label_dir = os.path.join(image_root, label_folder)
        if not os.path.isdir(label_dir):
            continue

        label = label_folder.strip().upper()

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(label_dir, fname)
            results["total"] += 1
            error = None

            # Check 1: valid label
            if label not in VALID_LABELS:
                error = f"Invalid label '{label}' for {fname}"
                results["invalid_labels"].append(fname)

            # Check 2: image not corrupt + min resolution + grayscale mode
            else:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
                            error = f"Resolution too small {w}x{h}: {img_path}"
                        elif img.mode not in ("L", "LA"):
                            logger.warning(f"Unexpected mode {img.mode}: {img_path}")
                except Exception as e:
                    error = f"Corrupt image {img_path}: {e}"
                    results["corrupt_images"].append(fname)

            if error:
                results["failed"] += 1
                results["errors"].append(error)
                logger.warning(error)
            else:
                results["passed"] += 1
                results["class_counts"][label] = results["class_counts"].get(label, 0) + 1

    logger.info(
        f"Validation complete: {results['passed']} passed, {results['failed']} failed"
    )
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/generated images"
    r = validate_dataset(path)
    print(r)