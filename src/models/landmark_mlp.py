# File: src/models/landmark_mlp.py
"""
Landmark-based MLP for ASL classification.
Input: 63-dim normalized hand landmark vector.
All hyperparams driven by params.yaml.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LandmarkMLP(nn.Module):
    """
    Fully-connected MLP operating on 63-dim landmark vectors.

    Architecture (fully configurable from params.yaml):
        Input(63) → FC(hidden[0]) → BN → ReLU → Dropout
                  → FC(hidden[1]) → BN → ReLU → Dropout
                  → ...
                  → FC(num_classes)

    All layer sizes, dropout rate, and batch-norm usage come from cfg.
    """

    def __init__(
        self,
        num_classes: int = 27,
        hidden_dims: list = None,
        dropout: float = 0.4,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = 63  # 21 landmarks × 3 (x, y, z)
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

        # Log param count
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LandmarkMLP | hidden={hidden_dims} | dropout={dropout} | "
            f"bn={use_batch_norm} | params={total:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, 63)
        Returns:
            logits: shape (batch, num_classes)
        """
        return self.net(x)


def build_mlp(cfg: dict, num_classes: int) -> LandmarkMLP:
    """
    Build LandmarkMLP from params.yaml config block.

    Expected cfg (params.yaml → models → mlp):
        hidden_dims: [256, 128]
        dropout: 0.4
        use_batch_norm: true
    """
    return LandmarkMLP(
        num_classes=num_classes,
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        dropout=cfg.get("dropout", 0.4),
        use_batch_norm=cfg.get("use_batch_norm", True),
    )
