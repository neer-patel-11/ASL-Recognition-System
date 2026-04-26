# File: src/models/dataset.py
"""
Dataset classes for training.
- LandmarkDataset  → for TraditionalML and LandmarkMLP (returns 63-dim vectors)
- PixelDataset     → for TinyCNN (returns image tensors)

Both read from data/processed/{train|val|test}/ folder structure.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Shared utility
# ─────────────────────────────────────────────────────────────

def _collect_samples(root_dir: str) -> Tuple[list, list, list]:
    """
    Walk root_dir expecting structure:
        root_dir/
            A/  img1.jpg  img2.jpg ...
            B/  ...
    Returns:
        paths  — list of absolute image paths
        labels — list of string labels
        classes — sorted list of unique classes
    """
    paths, labels = [], []
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset dir not found: {root_dir}")

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.upper()
        for img_file in label_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(img_file))
                labels.append(label)

    classes = sorted(set(labels))
    logger.info(f"Collected {len(paths)} images, {len(classes)} classes from {root_dir}")
    return paths, labels, classes


# ─────────────────────────────────────────────────────────────
#  Landmark Dataset  (Traditional ML + MLP)
# ─────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    """
    Extracts MediaPipe hand landmarks on-the-fly (or loads from cache).
    Returns (features: torch.Tensor shape (63,), label_idx: int).
    Skips images where no hand is detected.
    """

    def __init__(
        self,
        root_dir: str,
        classes: Optional[list] = None,
        cache_path: Optional[str] = None,
    ):
        self.paths, self.labels, discovered_classes = _collect_samples(root_dir)
        self.classes = classes or discovered_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.cache_path = cache_path

        # Lazy-load extractor to avoid import overhead at top level
        self._extractor = None

        # Load or build cache
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True).item()
            self._features = data["features"]   # (N, 63)
            self._label_idx = data["label_idx"] # (N,)
            self._valid_paths = data["paths"]
            logger.info(f"Loaded landmark cache: {len(self._features)} samples from {cache_path}")
        else:
            self._features = None  # built on first access or via build_cache()

    def _get_extractor(self):
        if self._extractor is None:
            from features.landmarks import LandmarkExtractor
            self._extractor = LandmarkExtractor()
        return self._extractor

    def build_cache(self):
        """
        Pre-extract all landmarks and cache to disk.
        NEVER drops samples — uses zero-vector fallback if detection fails.
        """
        extractor = self._get_extractor()
        features, label_idx, valid_paths = [], [], []

        for path, label in zip(self.paths, self.labels):
            feat = extractor.extract(path)

            # ✅ FIX: fallback instead of skipping
            if feat is None:
                feat = np.zeros(63, dtype=np.float32)

            features.append(feat)
            label_idx.append(self.class_to_idx[label])
            valid_paths.append(path)

        self._features = np.array(features, dtype=np.float32)
        self._label_idx = np.array(label_idx, dtype=np.int64)
        self._valid_paths = valid_paths

        logger.info(
            f"Extracted {len(self._features)} samples (no samples dropped)"
        )

        # 🔍 debug (optional but useful)
        from collections import Counter
        dist = Counter(self._label_idx)
        logger.info(f"Class distribution AFTER extraction: {dist}")

        if self.cache_path:
            np.save(
                self.cache_path,
                {
                    "features": self._features,
                    "label_idx": self._label_idx,
                    "paths": valid_paths,
                },
            )
            logger.info(f"Saved landmark cache to {self.cache_path}")

        return self
    def get_numpy(self):
        """Return (X, y) as numpy arrays — used by sklearn models."""
        if self._features is None:
            self.build_cache()
        return self._features, self._label_idx

    def __len__(self):
        if self._features is None:
            self.build_cache()
        return len(self._features)

    def __getitem__(self, idx):
        if self._features is None:
            self.build_cache()
        x = torch.tensor(self._features[idx], dtype=torch.float32)
        y = int(self._label_idx[idx])
        return x, y


# ─────────────────────────────────────────────────────────────
#  Pixel Dataset  (TinyCNN)
# ─────────────────────────────────────────────────────────────

class PixelDataset(Dataset):
    """
    Loads images and applies torchvision transforms.
    Returns (tensor: shape (C, H, W), label_idx: int).
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        classes: Optional[list] = None,
    ):
        self.paths, self.labels, discovered_classes = _collect_samples(root_dir)
        self.classes = classes or discovered_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = self.class_to_idx[self.labels[idx]]
        return img, y
