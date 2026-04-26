# File: src/utils/training_utils.py
"""
Reusable training utilities:
- EarlyStopping
- train_one_epoch / evaluate_epoch  (for PyTorch models)
- get_optimizer / get_scheduler
"""
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val_loss doesn't improve for `patience` epochs.
    Also saves the best model weights in memory.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_weights = None
        self.early_stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


# ─────────────────────────────────────────────────────────────
#  Optimizer factory
# ─────────────────────────────────────────────────────────────

def get_optimizer(model: nn.Module, cfg: dict):
    """
    Build optimizer from params.yaml training block.

    Expected cfg (params.yaml → training):
        optimizer: adam        # adam | sgd | adamw
        lr: 0.001
        weight_decay: 1e-4
        momentum: 0.9          # only used by sgd
    """
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, cfg: dict, num_epochs: int):
    """
    Build LR scheduler from params.yaml.

    Expected cfg (params.yaml → training):
        scheduler: cosine      # cosine | step | none
        step_size: 10          # for StepLR
        gamma: 0.1             # for StepLR
    """
    sched_name = cfg.get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.1),
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ─────────────────────────────────────────────────────────────
#  Epoch-level train / eval loops
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": all_preds,
        "targets": all_targets,
    }
