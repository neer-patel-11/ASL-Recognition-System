import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional

import cv2
import mediapipe as mp
from PIL import Image

logger = logging.getLogger(__name__)

# Landmark index constants (from MediaPipe)
WRIST_IDX = 0
NUM_LANDMARKS = 21
LANDMARK_DIM = 3   # x, y, z per landmark
OUTPUT_DIM = NUM_LANDMARKS * LANDMARK_DIM  # 63


class LandmarkExtractor:
    """
    Extracts and normalizes 21 hand landmarks from an image.

    Usage:
        extractor = LandmarkExtractor()
        features = extractor.extract("path/to/image.jpg")
        # → np.ndarray shape (63,) or None if no hand detected
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        self._static_image_mode       = static_image_mode
        self._max_num_hands           = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._hands = None   # lazy init — mediapipe loads on first use

    def _init_hands(self):
        """Lazy initialize MediaPipe Hands (avoids loading at import time)."""
        if self._hands is None:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self._static_image_mode,
                max_num_hands=self._max_num_hands,
                min_detection_confidence=self._min_detection_confidence,
            )

    def _load_image_as_rgb(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[np.ndarray]:
        """Accept file path, np.ndarray (BGR or RGB), or PIL.Image → RGB ndarray."""
        if isinstance(source, (str, Path)):
            bgr = cv2.imread(str(source))
            if bgr is None:
                logger.warning(f"Could not read image: {source}")
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(source, np.ndarray):
            # If 3-channel assume BGR (OpenCV default), convert to RGB
            if source.ndim == 3 and source.shape[2] == 3:
                return cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            return source  # already grayscale or RGB — pass through

        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))

        logger.warning(f"Unsupported input type: {type(source)}")
        return None

    def _normalize_landmarks(self, landmarks) -> np.ndarray:
        """
        Normalize all landmarks relative to the wrist (landmark 0).
        Wrist becomes origin (0, 0, 0).
        Result: shape (63,) — flattened [x0,y0,z0, x1,y1,z1, ...]
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32,
        )  # shape: (21, 3)

        # Subtract wrist position so wrist = origin
        wrist = coords[WRIST_IDX]
        coords -= wrist

        # Scale by max absolute value so range is [-1, 1]
        scale = np.max(np.abs(coords))
        if scale > 1e-6:
            coords /= scale

        return coords.flatten()  # shape: (63,)

    def extract(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[np.ndarray]:
        """
        Extract landmarks from one image.

        Returns:
            np.ndarray shape (63,) if hand detected
            None if no hand found
        """
        self._init_hands()

        rgb = self._load_image_as_rgb(source)
        if rgb is None:
            return None

        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            logger.debug(f"No hand detected in image")
            return None

        # Take first detected hand only
        hand_landmarks = results.multi_hand_landmarks[0]
        features = self._normalize_landmarks(hand_landmarks)

        assert features.shape == (OUTPUT_DIM,), \
            f"Expected ({OUTPUT_DIM},), got {features.shape}"

        return features

    def extract_batch(
        self, sources: list, show_progress: bool = False
    ) -> list:
        """
        Extract landmarks from a list of images.
        Returns list of (features or None) per image.
        """
        results = []
        iterator = sources
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sources, desc="Extracting landmarks")
            except ImportError:
                pass

        for src in iterator:
            results.append(self.extract(src))

        detected = sum(1 for r in results if r is not None)
        logger.info(
            f"Batch complete: {detected}/{len(sources)} hands detected "
            f"({detected/len(sources)*100:.1f}%)"
        )
        return results

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python landmarks.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    with LandmarkExtractor() as extractor:
        features = extractor.extract(img_path)

    if features is None:
        print("No hand detected.")
    else:
        print(f"Features shape : {features.shape}")
        print(f"Features range : [{features.min():.4f}, {features.max():.4f}]")
        print(f"First 9 values : {features[:9]}")