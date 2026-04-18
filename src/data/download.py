# File: src/data/download.py
import os
import time
import logging
import kagglehub
from data.config_loader import load_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dest_dir: str = None) -> dict:
    """
    Download dataset using config-driven handle.
    """
    params = load_params()

    data_cfg = params["data"]
    dest_dir = dest_dir or data_cfg["raw_dir"]
    dataset_handle = data_cfg["dataset_handle"]

    os.makedirs(dest_dir, exist_ok=True)
    start = time.perf_counter()

    logger.info(f"Downloading dataset: {dataset_handle}")

    dataset_path = kagglehub.dataset_download(dataset_handle)

    elapsed = time.perf_counter() - start
    logger.info(f"Downloaded to: {dataset_path} in {elapsed:.2f}s")

    # Count images
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
        "dataset_handle": dataset_handle,
    }

    logger.info(f"Download stats: {stats}")
    return stats


def load_train_csv(dataset_path: str):
    """
    Load training CSV using params.yaml config.
    """
    import pandas as pd

    params = load_params()
    data_cfg = params["data"]

    csv_name = data_cfg.get("train_csv", "train.csv")

    logger.info(f"Looking for CSV: {csv_name}")

    csv_path = os.path.join(dataset_path, csv_name)

    # fallback recursive search
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