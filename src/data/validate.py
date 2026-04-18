# File: src/data/validate.py
import os
import logging
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

VALID_LABELS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "NOTHING", "SPACE"])
MIN_RESOLUTION = (50, 50)


class DataValidationError(Exception):
    pass


def validate_dataset(dataset_path: str, train_csv_path: str) -> dict:
    """
    Validate all images listed in train.csv.
    Returns validation stats dict.
    """
    df = pd.read_csv(train_csv_path)

    # Normalize column names (handle different capitalizations)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected: 'filename' and 'label' columns
    if "filename" not in df.columns or "label" not in df.columns:
        raise DataValidationError(
            f"train.csv must have 'filename' and 'label' columns. Got: {list(df.columns)}"
        )

    results = {
        "total": len(df),
        "passed": 0,
        "failed": 0,
        "errors": [],
        "class_counts": {},
        "invalid_labels": [],
        "corrupt_images": [],
        "missing_files": [],
    }

    # Find image root directory
    image_root = None
    for root, dirs, _ in os.walk(dataset_path):
        if "train" in dirs:
            image_root = os.path.join(root, "train")
            break
    if image_root is None:
        image_root = dataset_path  # fallback

    seen_files = set()

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        # label = str(row["label"]).strip().upper()
        label = str(row["label"]).strip().upper().replace(" ", "")
        img_path = os.path.join(image_root, filename)

        error = None

        # Check 1: duplicate
        if filename in seen_files:
            error = f"Duplicate file: {filename}"
        else:
            seen_files.add(filename)

        # Check 2: file exists
        if not os.path.exists(img_path):
            error = f"Missing file: {img_path}"
            results["missing_files"].append(filename)

        # Check 3: valid label
        elif label not in VALID_LABELS:
            error = f"Invalid label '{label}' for {filename}"
            results["invalid_labels"].append(filename)

        # Check 4: image not corrupt + min resolution
        else:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
                        error = f"Resolution too small {w}x{h}: {filename}"
            except Exception as e:
                error = f"Corrupt image {filename}: {e}"
                results["corrupt_images"].append(filename)

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
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/train.csv"
    r = validate_dataset(path, csv)
    print(r)