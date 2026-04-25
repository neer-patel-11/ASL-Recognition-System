# File: src/data/split.py
import os, json, shutil, logging
from sklearn.model_selection import train_test_split
from data.config_loader import load_params

logger = logging.getLogger(__name__)


def split_dataset(
    dataset_path: str,
    train_csv_path: str = None,   # ignored — kept for interface compatibility
    output_dir: str = None,
    train_ratio: float = None,
    val_ratio: float = None,
    random_seed: int = None,
) -> dict:
    params = load_params()

    output_dir  = output_dir  or params["data"]["processed_dir"]
    train_ratio = train_ratio or params["split"]["train_ratio"]
    val_ratio   = val_ratio   or params["split"]["val_ratio"]
    random_seed = random_seed or params["split"]["random_seed"]

    # Build flat list of (abs_path, label) from folder structure
    samples = []   # [(abs_image_path, label), ...]
    # for label_folder in sorted(os.listdir(dataset_path)):
    #     label_dir = os.path.join(dataset_path, label_folder)
    #     if not os.path.isdir(label_dir):
    #         continue
    #     label = label_folder.strip().upper()
    #     for fname in os.listdir(label_dir):
    #         if fname.lower().endswith((".jpg", ".jpeg", ".png")):
    #             samples.append((os.path.join(label_dir, fname), label))

    # NEW — finds the 'generated images' subfolder first, then reads labels
    def _find_image_root(base_path: str) -> str:
        """
        Walk base_path to find the folder that contains A-Z + NIL subfolders.
        Handles cases where kagglehub downloads to a version folder.
        """
        for root, dirs, _ in os.walk(base_path):
            dirs_upper = [d.upper() for d in dirs]
            # If this folder contains at least 20 single-letter subfolders, it's our root
            letter_count = sum(1 for d in dirs_upper if len(d) == 1 and d.isalpha())
            if letter_count >= 20:
                return root
        return base_path  # fallback

    image_root = _find_image_root(dataset_path)
    logger.info(f"Image root resolved to: {image_root}")

    for label_folder in sorted(os.listdir(image_root)):
        label_dir = os.path.join(image_root, label_folder)
        if not os.path.isdir(label_dir):
            continue
        label = label_folder.strip().upper()
        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(label_dir, fname), label))

    if not samples:
        raise ValueError(f"No images found in {dataset_path}")

    paths, labels = zip(*samples)

    # Stratified split: train / (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=random_seed,
    )
    val_ratio_of_temp = val_ratio / (1 - train_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_of_temp),
        stratify=temp_labels,
        random_state=random_seed,
    )

    splits = {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }
    stats = {"splits": {}}

    for split_name, (split_paths, split_labels) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        copied = 0

        for src, label in zip(split_paths, split_labels):
            label_dir = os.path.join(split_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            dst = os.path.join(label_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

        # Count class distribution
        from collections import Counter
        dist = dict(Counter(split_labels))

        stats["splits"][split_name] = {
            "count":              len(split_paths),
            "copied":             copied,
            "class_distribution": dist,
        }
        logger.info(f"{split_name}: {len(split_paths)} images, {copied} copied")

    manifest_path = os.path.join(output_dir, "splits.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "train_ratio": train_ratio,
                "val_ratio":   val_ratio,
                "test_ratio":  round(1 - train_ratio - val_ratio, 2),
                "random_seed": random_seed,
                "splits": {k: v["count"] for k, v in stats["splits"].items()},
            },
            f, indent=2,
        )

    stats["manifest_path"] = manifest_path
    return stats