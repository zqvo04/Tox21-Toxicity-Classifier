"""Loss functions for masked multi-label toxicity classification.

All losses operate on raw logits of shape (B, n_tasks) and accept a binary
`mask` of the same shape where 1 = label present, 0 = label missing (NaN).
Missing entries never contribute to the loss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# pos_weight estimation (for class imbalance)
# -----------------------------------------------------------------------------
def compute_pos_weight(
    y: np.ndarray, mask: Optional[np.ndarray] = None, clip: float = 50.0
) -> torch.Tensor:
    """Per-task pos_weight = (#neg / #pos), ignoring missing labels.

    Parameters
    ----------
    y    : (N, n_tasks) labels (may contain NaN for missing).
    mask : (N, n_tasks) optional validity mask (1 valid, 0 missing).
    clip : upper bound to avoid exploding weights on very rare tasks.
    """
    y = np.asarray(y, dtype="float32")
    if mask is None:
        mask = (~np.isnan(y)).astype("float32")
    y_filled = np.nan_to_num(y, nan=0.0) * mask

    pos = (y_filled * mask).sum(axis=0)
    valid = mask.sum(axis=0)
    neg = valid - pos
    # Avoid division by zero.
    pos = np.clip(pos, 1.0, None)
    weight = np.clip(neg / pos, 1e-3, clip)
    return torch.tensor(weight, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Masked BCE
# -----------------------------------------------------------------------------
class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits that ignores masked (missing) labels.

    Parameters
    ----------
    pos_weight : optional (n_tasks,) tensor for positive-class up-weighting.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=None, pos_weight=pw, reduction="none"
        )
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom


# -----------------------------------------------------------------------------
# Focal loss
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Masked multi-label focal loss (Lin et al., 2017).

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : focusing parameter (default 2.0).
    alpha : weight for the positive class (default 0.25).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # BCE per element (no reduction).
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * bce

        focal = focal * mask
        denom = mask.sum().clamp_min(1.0)
        return focal.sum() / denom


def build_loss(name: str = "bce", pos_weight: Optional[torch.Tensor] = None,
               gamma: float = 2.0, alpha: float = 0.25) -> nn.Module:
    """Factory: 'bce' -> MaskedBCEWithLogitsLoss, 'focal' -> FocalLoss."""
    name = name.lower()
    if name in ("bce", "masked_bce"):
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"Unknown loss: {name}")
