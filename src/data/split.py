import os, json, shutil, logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data.config_loader import load_params          

logger = logging.getLogger(__name__)


def split_dataset(
    dataset_path: str,
    train_csv_path: str,
    output_dir: str = None,          
    train_ratio: float = None,
    val_ratio: float = None,
    random_seed: int = None,
) -> dict:
    params = load_params()

    # Fill defaults from config
    output_dir   = output_dir   or params["data"]["processed_dir"]
    train_ratio  = train_ratio  or params["split"]["train_ratio"]
    val_ratio    = val_ratio    or params["split"]["val_ratio"]
    random_seed  = random_seed  or params["split"]["random_seed"]

    df = pd.read_csv(train_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    image_root = None
    for root, dirs, _ in os.walk(dataset_path):
        if "train" in dirs:
            image_root = os.path.join(root, "train")
            break
    if image_root is None:
        image_root = dataset_path

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df["label"], random_state=random_seed,
    )
    val_ratio_of_temp = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp),
        stratify=temp_df["label"], random_state=random_seed,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    stats = {"splits": {}}

    for split_name, split_df in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        copied = 0
        for _, row in split_df.iterrows():
            filename = str(row["filename"]).strip()
            label    = str(row["label"]).strip().upper()
            label_dir = os.path.join(split_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join(image_root, filename)
            dst = os.path.join(label_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
        split_df.to_csv(os.path.join(output_dir, f"{split_name}.csv"), index=False)
        stats["splits"][split_name] = {
            "count": len(split_df),
            "copied": copied,
            "class_distribution": split_df["label"].value_counts().to_dict(),
        }
        logger.info(f"{split_name}: {len(split_df)} images")

    manifest_path = os.path.join(output_dir, "splits.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "train_ratio": train_ratio, "val_ratio": val_ratio,
            "test_ratio": round(1 - train_ratio - val_ratio, 2),
            "random_seed": random_seed,
            "splits": {k: v["count"] for k, v in stats["splits"].items()},
        }, f, indent=2)

    stats["manifest_path"] = manifest_path
    return stats