# File: src/data/download.py
import os
import time
import logging
import kagglehub
from kagglehub import KaggleDatasetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dest_dir: str = "data/raw") -> dict:
    """
    Download asl-alphabets dataset using kagglehub.
    Returns stats dict for email reporting.
    """
    os.makedirs(dest_dir, exist_ok=True)
    start = time.perf_counter()

    logger.info("Starting dataset download: aishikai/asl-alphabets")

    # Download full dataset (returns local cache path)
    dataset_path = kagglehub.dataset_download("aishikai/asl-alphabets")

    elapsed = time.perf_counter() - start
    logger.info(f"Downloaded to: {dataset_path} in {elapsed:.2f}s")

    # Count files
    image_count = 0
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_count += 1

    stats = {
        "dataset_path": dataset_path,
        "image_count": image_count,
        "download_time_sec": round(elapsed, 2),
        "dest_dir": dest_dir,
    }

    logger.info(f"Download stats: {stats}")
    return stats


def load_train_csv(dataset_path: str):
    """Load training CSV using ENV variable."""
    import pandas as pd

    # Read from ENV
    csv_name = os.environ.get("TRAIN_CSV_NAME", "train.csv")

    logger.info(f"Looking for CSV: {csv_name}")

    csv_path = os.path.join(dataset_path, csv_name)

    # If not found, fallback to recursive search
    if not os.path.exists(csv_path):
        logger.warning(f"{csv_name} not found at root. Searching recursively...")
        found = False
        for root, _, files in os.walk(dataset_path):
            if csv_name in files:
                csv_path = os.path.join(root, csv_name)
                found = True
                break

        if not found:
            raise FileNotFoundError(
                f"{csv_name} not found anywhere in dataset_path={dataset_path}"
            )

    df = pd.read_csv(csv_path)

    logger.info(
        f"{csv_name} loaded: {len(df)} rows, columns: {list(df.columns)}"
    )

    return df, csv_path
if __name__ == "__main__":
    stats = download_kaggle_dataset()
    print(stats)