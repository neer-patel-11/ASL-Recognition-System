import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet defaults — same as used by MobileNetV2, ResNet, EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Augmented transform for training data.
    Reads values from params.yaml if available, else uses defaults.
    """
    # Try to load from config
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.config_loader import load_params
        p = load_params()["preprocessing"]
        image_size = p.get("image_size", image_size)
        aug = p.get("augmentations", {})
        rotation  = aug.get("random_rotation_degrees", 15)
        h_flip    = aug.get("horizontal_flip", True)
        brightness = aug.get("color_jitter_brightness", 0.3)
        mean = p.get("normalize_mean", IMAGENET_MEAN)
        std  = p.get("normalize_std",  IMAGENET_STD)
    except Exception:
        rotation, h_flip, brightness = 15, True, 0.3
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    t = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(brightness=brightness),
    ]
    if h_flip:
        t.append(transforms.RandomHorizontalFlip())
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(t)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic transform for validation / inference.
    No augmentation — just resize + normalize.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.config_loader import load_params
        p = load_params()["preprocessing"]
        image_size = p.get("image_size", image_size)
        mean = p.get("normalize_mean", IMAGENET_MEAN)
        std  = p.get("normalize_std",  IMAGENET_STD)
    except Exception:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class PixelExtractor:
    """
    Converts any image source into a normalized torch.Tensor (3, H, W).

    Usage:
        extractor = PixelExtractor(mode="train")
        tensor = extractor.extract("path/to/image.jpg")
        # → torch.Tensor shape (3, 224, 224)
    """

    def __init__(self, mode: str = "eval", image_size: int = 224):
        """
        Args:
            mode: "train" (with augmentation) or "eval" (no augmentation)
            image_size: target size (default 224 for MobileNet/ResNet)
        """
        assert mode in ("train", "eval"), "mode must be 'train' or 'eval'"
        self.mode = mode
        self.image_size = image_size

        if mode == "train":
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_eval_transform(image_size)

        logger.info(f"PixelExtractor initialized: mode={mode}, size={image_size}")

    def _to_pil(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> Image.Image:
        """Convert any input type to PIL.Image RGB."""
        if isinstance(source, (str, Path)):
            return Image.open(str(source)).convert("RGB")

        if isinstance(source, np.ndarray):
            if source.dtype != np.uint8:
                source = (source * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(source).convert("RGB")

        if isinstance(source, Image.Image):
            return source.convert("RGB")

        raise TypeError(f"Unsupported input type: {type(source)}")

    def extract(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Extract pixel features from one image.

        Returns:
            torch.Tensor shape (3, image_size, image_size)
        Raises:
            FileNotFoundError: if path doesn't exist
            TypeError: if source type is unsupported
        """
        if isinstance(source, (str, Path)) and not Path(source).exists():
            raise FileNotFoundError(f"Image not found: {source}")

        pil_img = self._to_pil(source)
        tensor  = self.transform(pil_img)

        assert tensor.shape == (3, self.image_size, self.image_size), \
            f"Expected (3,{self.image_size},{self.image_size}), got {tensor.shape}"

        return tensor

    def extract_batch(self, sources: list) -> torch.Tensor:
        """
        Extract from a list → returns stacked tensor (N, 3, H, W).
        Skips failed images and logs warnings.
        """
        tensors = []
        for src in sources:
            try:
                tensors.append(self.extract(src))
            except Exception as e:
                logger.warning(f"Skipping {src}: {e}")

        if not tensors:
            raise ValueError("No images could be processed in batch")

        return torch.stack(tensors)  # shape: (N, 3, H, W)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pixels.py <image_path> [train|eval]")
        sys.exit(1)

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "eval"

    extractor = PixelExtractor(mode=mode)
    tensor = extractor.extract(path)

    print(f"Tensor shape : {tensor.shape}")
    print(f"Tensor dtype : {tensor.dtype}")
    print(f"Value range  : [{tensor.min():.4f}, {tensor.max():.4f}]")