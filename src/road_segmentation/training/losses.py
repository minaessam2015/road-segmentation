"""Loss functions for binary road segmentation.

Wraps battle-tested SMP loss implementations into configurable
compound losses driven by the experiment config.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion via min-pool (separable 3x1 then 1x3).

    min_pool(x) == -max_pool(-x). Separable kernel is equivalent to a 3x3
    cross-shaped structuring element — enough for centerline extraction.
    """
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation via 3x3 max-pool."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable skeletonization (Shit et al., CVPR 2021).

    Iteratively peels off one pixel layer per step. For a road of
    width W pixels, the centerline emerges after ~W/2 iterations.

    Our width analysis shows thin roads are ≤ 10 px wide, so iters=5
    centerlines every thin road we care about. Larger iters handles
    wider roads at higher compute cost.

    Args:
        x:     (N, 1, H, W) sigmoid probabilities or {0,1} targets.
        iters: number of erosion iterations (radius of the skeleton).
    """
    img1 = _soft_open(x)
    skel = F.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x)
        img1 = _soft_open(x)
        delta = F.relu(x - img1)
        # Accumulate new skeleton pixels that weren't already in `skel`.
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CLDiceLoss(nn.Module):
    """Centerline Dice loss (clDice, Shit et al., CVPR 2021).

    Penalizes topology breaks on thin tubular structures. Pixel-Dice
    is dominated by wide roads; clDice weights every centerline pixel
    equally regardless of road width, so a 3-pixel gap in a residential
    street costs the same as a 3-pixel gap in a highway.

    Two terms, both computed on the soft skeleton:
        Tprec = |skel(pred) ∩ gt| / |skel(pred)|    (skeleton precision)
        Tsens = |skel(gt) ∩ pred| / |skel(gt)|      (skeleton recall — the
                                                     one that punishes breaks)
        clDice = 1 - 2·Tprec·Tsens / (Tprec + Tsens)

    Args:
        iters: skeletonization radius. Must cover the widest relevant
               road (iters ≈ width/2). Default 5 = up to 10 px wide,
               matching the narrow-bucket cutoff from our width analysis.
        smooth: numerical stability.
        from_logits: apply sigmoid to predictions first.
    """

    def __init__(
        self,
        iters: int = 5,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.iters = iters
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        # Ensure (N, 1, H, W). SMP/trainer sometimes passes (N, H, W).
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.float()

        skel_pred = soft_skeletonize(y_pred, self.iters)
        skel_true = soft_skeletonize(y_true, self.iters)

        # Topology precision: predicted skeleton pixels that hit GT mass.
        tprec = (skel_pred * y_true).sum() + self.smooth
        tprec = tprec / (skel_pred.sum() + self.smooth)

        # Topology sensitivity: GT skeleton pixels covered by prediction.
        # This is the term that punishes breaks in thin roads.
        tsens = (skel_true * y_pred).sum() + self.smooth
        tsens = tsens / (skel_true.sum() + self.smooth)

        cldice = 2.0 * tprec * tsens / (tprec + tsens)
        return 1.0 - cldice


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

    if loss_type == "cldice":
        return CLDiceLoss(
            iters=params.get("iters", 5),
            smooth=params.get("smooth", 1.0),
            from_logits=from_logits,
        )

    if loss_type == "cldice_dice":
        return CompoundLoss(
            CLDiceLoss(
                iters=params.get("iters", 5),
                smooth=smooth,
                from_logits=from_logits,
            ),
            DiceLoss(mode=_MODE, from_logits=from_logits, smooth=smooth),
            weight_a=params.get("cldice_weight", 0.5),
            weight_b=params.get("dice_weight", 0.5),
        )

    if loss_type == "cldice_bce_dice":
        # clDice for topology + boundary-weighted BCE+Dice for pixel accuracy.
        return CompoundLoss(
            CLDiceLoss(
                iters=params.get("iters", 5),
                smooth=smooth,
                from_logits=from_logits,
            ),
            CompoundLoss(
                SoftBCEWithLogitsLoss(),
                DiceLoss(mode=_MODE, from_logits=from_logits, smooth=smooth),
                weight_a=params.get("bce_weight", 0.5),
                weight_b=params.get("dice_weight", 0.5),
            ),
            weight_a=params.get("cldice_weight", 0.4),
            weight_b=params.get("pixel_weight", 0.6),
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
