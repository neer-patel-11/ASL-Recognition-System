# File: src/data/download.py
import os, time, logging
import kagglehub
from config_loader import load_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dest_dir: str = None) -> dict:
    params = load_params()
    data_cfg = params["data"]
    dest_dir = dest_dir or data_cfg["raw_dir"]
    dataset_handle = data_cfg["dataset_handle"]

    os.makedirs(dest_dir, exist_ok=True)
    start = time.perf_counter()

    logger.info(f"Downloading dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    elapsed = time.perf_counter() - start

    # Count images and classes
    image_count = 0
    class_counts = {}
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                label = os.path.basename(root).upper()
                class_counts[label] = class_counts.get(label, 0) + 1
                image_count += 1

    stats = {
        "dataset_path":      dataset_path,
        "image_count":       image_count,
        "class_count":       len(class_counts),
        "class_counts":      class_counts,
        "download_time_sec": round(elapsed, 2),
        "dest_dir":          dest_dir,
        "dataset_handle":    dataset_handle,
    }

    logger.info(f"Download stats: {stats}")
    return stats


if __name__ == "__main__":
    print(download_kaggle_dataset())