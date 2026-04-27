# File: src/train.py
"""
Main training script for the ASL recognition pipeline — with MLflow tracking.

Usage:
    python src/train.py                          # uses params.yaml models.active
    python src/train.py --model traditional_ml
    python src/train.py --model mlp
    python src/train.py --model cnn

Everything (hyperparams, paths, model choice) is read from params.yaml.
MLflow tracking URI is read from MLFLOW_TRACKING_URI env var, defaulting to
http://localhost:5000 (the Docker service).
"""
import argparse
import logging
import os
import sys
import time

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.config_loader import load_params
from evaluate import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_MODELS = ("traditional_ml", "mlp", "cnn")

# ── MLflow setup ──────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT_NAME", "ASL-Recognition")

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment   : {MLFLOW_EXPERIMENT}")


# ──────────────────────────────────────────────────────────────
#  Traditional ML trainer
# ──────────────────────────────────────────────────────────────

def train_traditional(params: dict):
    from models.traditional_ml import TraditionalMLModel
    from models.dataset import LandmarkDataset

    cfg        = params["models"]["traditional"]
    data_cfg   = params["data"]
    output_dir = params["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    processed  = data_cfg["processed_dir"]
    algo       = cfg.get("algorithm", "random_forest")

    with mlflow.start_run(run_name=f"traditional_{algo}") as run:
        # ── Log params ──
        mlflow.set_tag("model_type", "traditional_ml")
        mlflow.set_tag("algorithm",  algo)
        mlflow.log_param("algorithm", algo)
        # Log algorithm-specific params
        algo_cfg = cfg.get(algo, {})
        for k, v in algo_cfg.items():
            mlflow.log_param(f"{algo}.{k}", v)

        logger.info(f"MLflow run id: {run.info.run_id}")

        # ── Datasets ──
        logger.info("Building landmark datasets …")
        train_ds = LandmarkDataset(
            root_dir=os.path.join(processed, "train"),
            cache_path=os.path.join(processed, "landmark_cache_train.npy"),
        )
        val_ds = LandmarkDataset(
            root_dir=os.path.join(processed, "val"),
            classes=train_ds.classes,
            cache_path=os.path.join(processed, "landmark_cache_val.npy"),
        )

        X_train, y_train = train_ds.get_numpy()
        X_val,   y_val   = val_ds.get_numpy()
        classes          = train_ds.classes
        y_train_str      = [classes[i] for i in y_train]

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples",   len(X_val))
        mlflow.log_param("num_classes",   len(classes))
        mlflow.log_param("feature_dim",   X_train.shape[1])

        # ── Train ──
        model = TraditionalMLModel(cfg)
        t0 = time.perf_counter()
        model.fit(X_train, y_train_str)
        elapsed = time.perf_counter() - t0
        mlflow.log_metric("training_time_sec", round(elapsed, 2))
        logger.info(f"Training done in {elapsed:.1f}s")

        # ── Evaluate ──
        y_pred_str = model.predict(X_val)
        y_pred_idx = [classes.index(p) for p in y_pred_str]
        y_val_idx  = y_val.tolist()

        metrics = compute_metrics(
            y_val_idx,
            y_pred_idx,
            classes,
            labels=list(range(len(classes))),  # ✅ force all 27 classes
            output_dir=os.path.join(output_dir, "traditional"),
            model_name=f"traditional_{algo}",
        )

        # ── Log metrics ──
        mlflow.log_metric("val_accuracy", metrics["accuracy"])
        mlflow.log_metric("val_macro_f1", metrics["macro_f1"])
        for cls, f1 in metrics["per_class_f1"].items():
            mlflow.log_metric(f"f1_{cls}", f1)

        # ── Log artifacts ──
        cm_path = os.path.join(output_dir, "traditional", f"traditional_{algo}_confusion_matrix.png")
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # ── Save & register model ──
        model_path = os.path.join(output_dir, "traditional", "traditional_model.pkl")
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Register in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            reg = mlflow.register_model(model_uri, f"ASL-Traditional-{algo.upper()}")
            logger.info(f"Registered model version: {reg.version}")
        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")

        _save_dvc_metrics(metrics, output_dir, "traditional")
        logger.info(f"traditional_ml done | acc={metrics['accuracy']} | f1={metrics['macro_f1']}")

    return metrics


# ──────────────────────────────────────────────────────────────
#  MLP trainer
# ──────────────────────────────────────────────────────────────

def train_mlp(params: dict):
    from models.landmark_mlp import build_mlp
    from models.dataset import LandmarkDataset
    from utils.training_utils import (
        EarlyStopping, get_optimizer, get_scheduler,
        train_one_epoch, evaluate_epoch,
    )

    cfg        = params["models"]["mlp"]
    train_cfg  = params["training"]
    data_cfg   = params["data"]
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    processed  = data_cfg["processed_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="landmark_mlp") as run:
        # ── Log params ──
        mlflow.set_tag("model_type", "mlp")
        mlflow.set_tag("device",     str(device))
        mlflow.log_params({
            "hidden_dims":      str(cfg.get("hidden_dims", [256, 128])),
            "dropout":          cfg.get("dropout", 0.4),
            "use_batch_norm":   cfg.get("use_batch_norm", True),
            "epochs":           train_cfg.get("epochs", 50),
            "batch_size":       train_cfg.get("batch_size", 128),
            "optimizer":        train_cfg.get("optimizer", "adam"),
            "lr":               train_cfg.get("lr", 0.001),
            "weight_decay":     train_cfg.get("weight_decay", 1e-4),
            "scheduler":        train_cfg.get("scheduler", "cosine"),
            "label_smoothing":  train_cfg.get("label_smoothing", 0.1),
            "patience":         train_cfg.get("patience", 10),
        })
        logger.info(f"MLflow run id: {run.info.run_id}")
        logger.info(f"Device: {device}")

        # ── Datasets ──
        logger.info("Building landmark datasets …")
        train_ds = LandmarkDataset(
            root_dir=os.path.join(processed, "train"),
            cache_path=os.path.join(processed, "landmark_cache_train.npy"),
        )
        val_ds = LandmarkDataset(
            root_dir=os.path.join(processed, "val"),
            classes=train_ds.classes,
            cache_path=os.path.join(processed, "landmark_cache_val.npy"),
        )

        classes     = train_ds.classes
        num_classes = len(classes)
        batch_size  = train_cfg.get("batch_size", 128)

        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("val_samples",   len(val_ds))
        mlflow.log_param("num_classes",   num_classes)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # ── Model ──
        model      = build_mlp(cfg, num_classes).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("num_parameters", num_params)

        # ── Optimizer / scheduler / loss ──
        optimizer  = get_optimizer(model, train_cfg)
        num_epochs = train_cfg.get("epochs", 50)
        scheduler  = get_scheduler(optimizer, train_cfg, num_epochs)
        criterion  = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.1))
        stopper    = EarlyStopping(
            patience=train_cfg.get("patience", 10),
            min_delta=train_cfg.get("min_delta", 1e-4),
        )

        best_val_acc = 0.0

        # ── Training loop ──
        for epoch in range(1, num_epochs + 1):
            train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_m   = evaluate_epoch(model, val_loader, criterion, device)

            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Log every epoch to MLflow
            mlflow.log_metrics({
                "train_loss": round(train_m["loss"], 4),
                "train_acc":  round(train_m["accuracy"], 4),
                "val_loss":   round(val_m["loss"], 4),
                "val_acc":    round(val_m["accuracy"], 4),
                "lr":         current_lr,
            }, step=epoch)

            logger.info(
                f"[MLP] epoch {epoch:03d}/{num_epochs} | "
                f"train_loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            stopper.step(val_m["loss"], model)
            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                mlflow.set_tag("stopped_early", True)
                mlflow.log_param("stopped_at_epoch", epoch)
                break

        stopper.restore_best(model)

        # ── Final evaluation ──
        val_m   = evaluate_epoch(model, val_loader, criterion, device)
        metrics = compute_metrics(
            val_m["targets"], val_m["preds"], classes,
            output_dir=os.path.join(output_dir, "mlp"),
            model_name="mlp",
        )

        # ── Log final metrics ──
        mlflow.log_metrics({
            "final_val_accuracy": metrics["accuracy"],
            "final_val_macro_f1": metrics["macro_f1"],
            "best_val_accuracy":  best_val_acc,
        })
        for cls, f1 in metrics["per_class_f1"].items():
            mlflow.log_metric(f"f1_{cls}", f1)

        # ── Log artifacts ──
        cm_path = os.path.join(output_dir, "mlp", "mlp_confusion_matrix.png")
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # ── Save & log model ──
        model_save_dir = os.path.join(output_dir, "mlp")
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, "mlp_best.pth")


        from mlflow_wrappers.mlp_wrapper import MLPWrapper

        model_path = os.path.join(model_save_dir, "mlp_best.pth")

        torch.save({
            "model_state": model.state_dict(),
            "classes": classes,
            "cfg": cfg
        }, model_path)

        mlflow.pyfunc.log_model(
            artifact_path="pytorch_model",
            python_model=MLPWrapper(),
            artifacts={"model_path": model_path}
        )

        # torch.save({"model_state": model.state_dict(), "classes": classes, "cfg": cfg}, model_path)

        # Log PyTorch model to MLflow
        # mlflow.pytorch.log_model(model, artifact_path="pytorch_model")

        # Register in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/pytorch_model"
        try:
            reg = mlflow.register_model(model_uri, "ASL-LandmarkMLP")
            logger.info(f"Registered model version: {reg.version}")
        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")

        _save_dvc_metrics(metrics, output_dir, "mlp")
        logger.info(f"mlp done | acc={metrics['accuracy']} | f1={metrics['macro_f1']}")

    return metrics


# ──────────────────────────────────────────────────────────────
#  CNN trainer
# ──────────────────────────────────────────────────────────────

def train_cnn(params: dict):
    from models.tiny_cnn import build_cnn
    from models.dataset import PixelDataset
    from features.pixels import build_train_transform, build_eval_transform
    from utils.training_utils import (
        EarlyStopping, get_optimizer, get_scheduler,
        train_one_epoch, evaluate_epoch,
    )

    cfg        = params["models"]["cnn"]
    train_cfg  = params["training"]
    data_cfg   = params["data"]
    prep_cfg   = params["preprocessing"]
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    processed  = data_cfg["processed_dir"]
    image_size = prep_cfg.get("image_size", 224)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="tiny_cnn") as run:
        # ── Log params ──
        mlflow.set_tag("model_type", "cnn")
        mlflow.set_tag("device",     str(device))
        mlflow.log_params({
            "channels":         str(cfg.get("channels", [32, 64, 128, 256])),
            "fc_dim":           cfg.get("fc_dim", 512),
            "dropout":          cfg.get("dropout", 0.5),
            "input_channels":   cfg.get("input_channels", 3),
            "image_size":       image_size,
            "epochs":           train_cfg.get("epochs", 50),
            "batch_size":       train_cfg.get("batch_size", 64),
            "optimizer":        train_cfg.get("optimizer", "adam"),
            "lr":               train_cfg.get("lr", 0.001),
            "weight_decay":     train_cfg.get("weight_decay", 1e-4),
            "scheduler":        train_cfg.get("scheduler", "cosine"),
            "label_smoothing":  train_cfg.get("label_smoothing", 0.1),
            "patience":         train_cfg.get("patience", 10),
            # Augmentation params
            "aug_rotation_deg": prep_cfg.get("augmentations", {}).get("random_rotation_degrees", 15),
            "aug_hflip":        prep_cfg.get("augmentations", {}).get("horizontal_flip", True),
            "aug_brightness":   prep_cfg.get("augmentations", {}).get("color_jitter_brightness", 0.3),
            "normalize_mean":   str(prep_cfg.get("normalize_mean", [0.5, 0.5, 0.5])),
            "normalize_std":    str(prep_cfg.get("normalize_std",  [0.5, 0.5, 0.5])),
        })
        logger.info(f"MLflow run id: {run.info.run_id}")
        logger.info(f"Device: {device}")

        # ── Datasets ──
        train_transform = build_train_transform(image_size)
        eval_transform  = build_eval_transform(image_size)
        train_ds = PixelDataset(os.path.join(processed, "train"), transform=train_transform)
        val_ds   = PixelDataset(os.path.join(processed, "val"),   transform=eval_transform, classes=train_ds.classes)

        classes     = train_ds.classes
        num_classes = len(classes)
        batch_size  = train_cfg.get("batch_size", 64)

        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("val_samples",   len(val_ds))
        mlflow.log_param("num_classes",   num_classes)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # ── Model ──
        model      = build_cnn(cfg, num_classes, image_size).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("num_parameters", num_params)

        # ── Optimizer / scheduler / loss ──
        optimizer  = get_optimizer(model, train_cfg)
        num_epochs = train_cfg.get("epochs", 50)
        scheduler  = get_scheduler(optimizer, train_cfg, num_epochs)
        criterion  = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.1))
        stopper    = EarlyStopping(
            patience=train_cfg.get("patience", 10),
            min_delta=train_cfg.get("min_delta", 1e-4),
        )

        best_val_acc = 0.0

        # ── Training loop ──
        for epoch in range(1, num_epochs + 1):
            train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_m   = evaluate_epoch(model, val_loader, criterion, device)

            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Log every epoch to MLflow
            mlflow.log_metrics({
                "train_loss": round(train_m["loss"], 4),
                "train_acc":  round(train_m["accuracy"], 4),
                "val_loss":   round(val_m["loss"], 4),
                "val_acc":    round(val_m["accuracy"], 4),
                "lr":         current_lr,
            }, step=epoch)

            logger.info(
                f"[CNN] epoch {epoch:03d}/{num_epochs} | "
                f"train_loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            stopper.step(val_m["loss"], model)
            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                mlflow.set_tag("stopped_early", True)
                mlflow.log_param("stopped_at_epoch", epoch)
                break

        stopper.restore_best(model)

        # ── Final evaluation ──
        val_m   = evaluate_epoch(model, val_loader, criterion, device)
        metrics = compute_metrics(
            val_m["targets"], val_m["preds"], classes,
            output_dir=os.path.join(output_dir, "cnn"),
            model_name="tiny_cnn",
        )

        # ── Log final metrics ──
        mlflow.log_metrics({
            "final_val_accuracy": metrics["accuracy"],
            "final_val_macro_f1": metrics["macro_f1"],
            "best_val_accuracy":  best_val_acc,
        })
        for cls, f1 in metrics["per_class_f1"].items():
            mlflow.log_metric(f"f1_{cls}", f1)

        # ── Log artifacts ──
        cm_path = os.path.join(output_dir, "cnn", "tiny_cnn_confusion_matrix.png")
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # ── Save & log model ──
        model_save_dir = os.path.join(output_dir, "cnn")
        os.makedirs(model_save_dir, exist_ok=True)
        # model_path = os.path.join(model_save_dir, "cnn_best.pth")
        # torch.save({"model_state": model.state_dict(), "classes": classes, "cfg": cfg}, model_path)

        # # Log PyTorch model to MLflow
        # mlflow.pytorch.log_model(model, artifact_path="pytorch_model")

        from mlflow_wrappers.cnn_wrapper import CNNWrapper

        model_path = os.path.join(model_save_dir, "cnn_best.pth")

        torch.save({
            "model_state": model.state_dict(),
            "classes": classes,
            "cfg": cfg
        }, model_path)

        mlflow.pyfunc.log_model(
            artifact_path="pytorch_model",
            python_model=CNNWrapper(),
            artifacts={"model_path": model_path}
        )

        # Register in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/pytorch_model"
        try:
            reg = mlflow.register_model(model_uri, "ASL-TinyCNN")
            logger.info(f"Registered model version: {reg.version}")
        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")

        _save_dvc_metrics(metrics, output_dir, "cnn")
        logger.info(f"cnn done | acc={metrics['accuracy']} | f1={metrics['macro_f1']}")

    return metrics


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _save_dvc_metrics(metrics: dict, output_dir: str, model_name: str):
    import json
    metrics_dir = os.path.join(output_dir, model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "eval.json")
    with open(path, "w") as f:
        json.dump({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, f, indent=2)
    logger.info(f"DVC metrics written to {path}")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        default=None,
        help="Model to train. Overrides params.yaml models.active",
    )
    args = parser.parse_args()

    params     = load_params()
    model_name = args.model or params["models"].get("active", "cnn")

    setup_mlflow()

    logger.info(f"{'='*60}")
    logger.info(f"  Training model : {model_name}")
    logger.info(f"  MLflow URI     : {MLFLOW_TRACKING_URI}")
    logger.info(f"{'='*60}")

    if model_name == "traditional_ml":
        train_traditional(params)
    elif model_name == "mlp":
        train_mlp(params)
    elif model_name == "cnn":
        train_cnn(params)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {VALID_MODELS}")


if __name__ == "__main__":
    main()