# File: src/models/tiny_cnn.py
"""
Ultra-tiny CNN for ASL classification on raw pixel input.
No pretrained weights — trained from scratch.
All hyperparams driven by params.yaml.

Architecture strategy: Depthwise Separable Convolutions (DSConv)
  Standard Conv(in, out, k):  params = in * out * k * k
  DSConv(in, out, k):         params = in * k * k  +  in * out   (depthwise + pointwise)
  Savings ratio ≈ 1 / out  (e.g. 8× fewer params for 8 output channels)

Target: ~100K parameters
Default config achieves ~97K params.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────

class DSConvBlock(nn.Module):
    """
    Depthwise Separable Conv block:
        Depthwise  Conv(in_ch, in_ch, k, groups=in_ch)  — spatial mixing
        Pointwise  Conv(in_ch, out_ch, 1)               — channel mixing
        BatchNorm → ReLU6 → optional MaxPool
    ReLU6 clips activations at 6, which helps quantisation later.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel,
                      padding=kernel // 2, groups=in_ch, bias=False),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """
    Ultra-lightweight CNN (~100K params) trained from scratch.

    Default architecture with channels=[16, 32, 64, 96]:
        Input  (3,  224, 224)
        DSConv( 3 →  16) + Pool  →  (16, 112, 112)   ~  0.2K params
        DSConv(16 →  32) + Pool  →  (32,  56,  56)   ~  0.7K params
        DSConv(32 →  64) + Pool  →  (64,  28,  28)   ~  2.1K params
        DSConv(64 →  96) + Pool  →  (96,  14,  14)   ~  6.7K params
        AdaptiveAvgPool(2×2)     →  (96,   2,   2)
        Flatten                  →  384
        FC(384 → fc_dim=128)     →  ~49K params
        ReLU6 → Dropout
        FC(128 → 27)             →  ~3.5K params
        ─────────────────────────────────────────────
        Total                    ≈  97K parameters

    Why AdaptiveAvgPool(2×2) instead of (4×4)?
        (4×4) → flatten_dim = 96*16 = 1536 → FC alone = 1536*128 = 197K  ❌
        (2×2) → flatten_dim = 96*4  =  384 → FC alone =  384*128 =  49K  ✅
    """

    def __init__(
        self,
        num_classes: int = 27,
        channels: list = None,
        fc_dim: int = 128,
        dropout: float = 0.4,
        image_size: int = 224,       # kept for interface compatibility
        input_channels: int = 3,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 96]

        # ── Backbone ──────────────────────────────────────────
        conv_layers = []
        in_ch = input_channels
        for out_ch in channels:
            conv_layers.append(DSConvBlock(in_ch, out_ch, pool=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*conv_layers)

        # ── Global spatial compression ─────────────────────────
        # (2×2) keeps param count of the FC layer small
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        flatten_dim = channels[-1] * 2 * 2   # 96*4 = 384 by default

        # ── Classifier ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, fc_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"TinyCNN (DSConv) | channels={channels} | fc_dim={fc_dim} | "
            f"pool=2×2 | dropout={dropout} | params={total:,}"
        )
        if total > 200_000:
            logger.warning(
                f"Parameter count {total:,} exceeds 200K — "
                "consider reducing channels or fc_dim."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.backbone(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────

def build_cnn(cfg: dict, num_classes: int, image_size: int = 224) -> TinyCNN:
    """
    Build TinyCNN from params.yaml config block.

    Expected cfg (params.yaml → models → cnn):
        channels: [16, 32, 64, 96]   # reduced from [32,64,128,256]
        fc_dim: 128                  # reduced from 512
        dropout: 0.4
        input_channels: 3
    """
    return TinyCNN(
        num_classes=num_classes,
        channels=cfg.get("channels", [16, 32, 64, 96]),
        fc_dim=cfg.get("fc_dim", 128),
        dropout=cfg.get("dropout", 0.4),
        image_size=image_size,
        input_channels=cfg.get("input_channels", 3),
    )