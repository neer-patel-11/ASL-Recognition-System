# File: src/models/traditional_ml.py
"""
Traditional ML model for ASL classification using landmark features (63-dim).
Supports: random_forest, svm, knn — switchable from params.yaml.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("random_forest", "svm", "knn")


def build_traditional_model(cfg: dict):
    """
    Build a sklearn estimator from params.yaml config.

    Expected cfg shape (params.yaml → models → traditional):
        algorithm: random_forest   # or svm / knn
        random_forest:
            n_estimators: 200
            max_depth: null
            min_samples_split: 2
            n_jobs: -1
            random_state: 42
        svm:
            C: 10.0
            kernel: rbf
            gamma: scale
            probability: true
        knn:
            n_neighbors: 7
            weights: distance
            metric: euclidean
    """
    algo = cfg.get("algorithm", "random_forest")
    assert algo in SUPPORTED_MODELS, (
        f"Unknown algorithm '{algo}'. Choose from {SUPPORTED_MODELS}"
    )

    if algo == "random_forest":
        p = cfg.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth") or None,
            min_samples_split=p.get("min_samples_split", 2),
            n_jobs=p.get("n_jobs", -1),
            random_state=p.get("random_state", 42),
            verbose=0,
        )

    elif algo == "svm":
        p = cfg.get("svm", {})
        model = SVC(
            C=p.get("C", 10.0),
            kernel=p.get("kernel", "rbf"),
            gamma=p.get("gamma", "scale"),
            probability=p.get("probability", True),
            random_state=42,
        )

    elif algo == "knn":
        p = cfg.get("knn", {})
        model = KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 7),
            weights=p.get("weights", "distance"),
            metric=p.get("metric", "euclidean"),
            n_jobs=-1,
        )

    logger.info(f"Built traditional model: {algo} → {model}")
    return model, algo


class TraditionalMLModel:
    """
    Wrapper around a sklearn estimator.
    Handles fit / predict / save / load.
    """

    def __init__(self, cfg: dict):
        self.model, self.algo = build_traditional_model(cfg)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: shape (N, 63) — normalized landmark features
            y: shape (N,)   — string class labels (e.g. 'A', 'B', ...)
        """
        y_enc = self.label_encoder.fit_transform(y)
        logger.info(
            f"Fitting {self.algo} on {X.shape[0]} samples, "
            f"{len(self.label_encoder.classes_)} classes"
        )
        self.model.fit(X, y_enc)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted string labels."""
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability matrix (N, num_classes)."""
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{self.algo} does not support predict_proba. "
                "Set probability=true for SVM."
            )
        return self.model.predict_proba(X)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)
        logger.info(f"Saved traditional model to {path}")

    @classmethod
    def load(cls, path: str) -> "TraditionalMLModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.label_encoder = data["label_encoder"]
        instance.algo = type(instance.model).__name__
        instance.is_fitted = True
        return instance
