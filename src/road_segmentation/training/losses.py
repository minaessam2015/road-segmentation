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


class BoundaryWeightedBCEDiceLoss(nn.Module):
    """BCE + Dice loss with higher weight on boundary pixels.

    Computes a per-pixel weight map from the ground truth mask:
    - Pixels near road boundaries get ``edge_weight``
    - Other pixels get weight 1.0

    The boundary is detected via morphological gradient (dilation - erosion)
    of the ground truth mask, controlled by ``boundary_width``.

    This targets the hardest pixels for road segmentation — thin edges
    where FP/FN errors concentrate. Literature reports +2-5 IoU points
    from boundary weighting on road extraction tasks.
    """

    def __init__(
        self,
        edge_weight: float = 5.0,
        boundary_width: int = 3,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.boundary_width = boundary_width
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.dice_loss = DiceLoss(mode=_MODE, from_logits=True, smooth=smooth)

    def _compute_boundary_weights(self, y_true: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel boundary weights from ground truth.

        Uses max-pool based morphological gradient (works on GPU, no OpenCV needed).

        Args:
            y_true: (N, 1, H, W) binary mask.

        Returns:
            (N, 1, H, W) weight map with edge_weight on boundaries, 1.0 elsewhere.
        """
        k = self.boundary_width
        pad = k // 2

        # Dilation: max-pool the mask
        dilated = torch.nn.functional.max_pool2d(y_true, kernel_size=k, stride=1, padding=pad)
        # Erosion: min-pool = -max_pool(-x)
        eroded = -torch.nn.functional.max_pool2d(-y_true, kernel_size=k, stride=1, padding=pad)
        # Boundary = dilation - erosion
        boundary = (dilated - eroded).clamp(0, 1)

        # Weight map: 1.0 everywhere + (edge_weight - 1) on boundaries
        weights = 1.0 + boundary * (self.edge_weight - 1.0)
        return weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (N, 1, H, W) raw logits.
            y_true: (N, 1, H, W) binary targets {0, 1}.
        """
        weights = self._compute_boundary_weights(y_true)

        # Weighted BCE (from logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, weight=weights, reduction="mean",
        )

        # Dice loss (unweighted — it's already imbalance-aware)
        dice = self.dice_loss(y_pred, y_true)

        return self.bce_weight * bce + self.dice_weight * dice


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

    if loss_type == "boundary_bce_dice":
        return BoundaryWeightedBCEDiceLoss(
            edge_weight=params.get("edge_weight", 5.0),
            boundary_width=params.get("boundary_width", 3),
            bce_weight=params.get("bce_weight", 0.5),
            dice_weight=params.get("dice_weight", 0.5),
            smooth=params.get("smooth", 1.0),
        )

    raise ValueError(f"Unknown loss type: '{loss_type}'")
