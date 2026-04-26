# File: src/evaluate.py
"""
Evaluation utilities shared across all three model types.
Outputs: accuracy, macro-F1, per-class report, confusion matrix PNG.
"""
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred, classes, labels=None, output_dir=None, model_name="model"):
    """
    Compute accuracy, macro-F1, per-class metrics.
    Save confusion matrix PNG and metrics JSON.

    Args:
        y_true:     integer label indices
        y_pred:     integer label indices
        classes:    ordered list of class name strings
        labels:     optional list of label indices to include (forces all classes)
        output_dir: where to save confusion_matrix.png and metrics.json
        model_name: prefix for saved files

    Returns:
        dict with accuracy, macro_f1, per_class_f1
    """
    os.makedirs(output_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {
        cls: round(report[cls]["f1-score"], 4)
        for cls in classes
        if cls in report
    }

    metrics = {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro_f1, 4),
        "per_class_f1": per_class_f1,
    }

    # ── Save metrics JSON ──
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred)
    _save_confusion_matrix(cm, classes, output_dir, model_name)

    # ── Console summary ──
    logger.info(f"{'='*50}")
    logger.info(f"Model : {model_name}")
    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"Macro-F1  : {macro_f1:.4f}")
    logger.info(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    return metrics


def _save_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    output_dir: str,
    model_name: str,
):
    fig_size = max(12, len(classes) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.4,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")