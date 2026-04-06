"""Loss functions for binary road segmentation.

Wraps battle-tested SMP loss implementations into configurable
compound losses driven by the experiment config.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import (
    DiceLoss,
    FocalLoss,
    JaccardLoss,
    SoftBCEWithLogitsLoss,
    TverskyLoss,
)

_MODE = "binary"


class CompoundLoss(nn.Module):
    """Weighted sum of two loss functions."""

    def __init__(
        self,
        loss_a: nn.Module,
        loss_b: nn.Module,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ) -> None:
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.weight_a * self.loss_a(y_pred, y_true) + self.weight_b * self.loss_b(y_pred, y_true)


def create_loss(loss_type: str, params: Dict[str, Any]) -> nn.Module:
    """Create a loss function from config.

    Supported types:
        - ``"bce_dice"``: SoftBCEWithLogitsLoss + DiceLoss
        - ``"focal_dice"``: FocalLoss + DiceLoss
        - ``"bce_jaccard"``: SoftBCEWithLogitsLoss + JaccardLoss
        - ``"dice"``: DiceLoss only
        - ``"tversky"``: TverskyLoss only
        - ``"focal_tversky"``: FocalLoss + TverskyLoss
    """
    from_logits = params.get("from_logits", True)
    smooth = params.get("smooth", 1.0)

    if loss_type == "bce_dice":
        return CompoundLoss(
            SoftBCEWithLogitsLoss(),
            DiceLoss(mode=_MODE, from_logits=from_logits, smooth=smooth),
            weight_a=params.get("bce_weight", 0.5),
            weight_b=params.get("dice_weight", 0.5),
        )

    if loss_type == "focal_dice":
        return CompoundLoss(
            FocalLoss(mode=_MODE, gamma=params.get("focal_gamma", 2.0)),
            DiceLoss(mode=_MODE, from_logits=from_logits, smooth=smooth),
            weight_a=params.get("focal_weight", 0.5),
            weight_b=params.get("dice_weight", 0.5),
        )

    if loss_type == "bce_jaccard":
        return CompoundLoss(
            SoftBCEWithLogitsLoss(),
            JaccardLoss(mode=_MODE, from_logits=from_logits, smooth=smooth),
            weight_a=params.get("bce_weight", 0.5),
            weight_b=params.get("jaccard_weight", 0.5),
        )

    if loss_type == "dice":
        return DiceLoss(mode=_MODE, from_logits=from_logits, smooth=smooth)

    if loss_type == "tversky":
        return TverskyLoss(
            mode=_MODE,
            from_logits=from_logits,
            smooth=smooth,
            alpha=params.get("alpha", 0.3),
            beta=params.get("beta", 0.7),
        )

    if loss_type == "focal_tversky":
        return CompoundLoss(
            FocalLoss(mode=_MODE, gamma=params.get("focal_gamma", 2.0)),
            TverskyLoss(
                mode=_MODE,
                from_logits=from_logits,
                smooth=smooth,
                alpha=params.get("alpha", 0.3),
                beta=params.get("beta", 0.7),
            ),
            weight_a=params.get("focal_weight", 0.5),
            weight_b=params.get("tversky_weight", 0.5),
        )

    raise ValueError(f"Unknown loss type: '{loss_type}'")
