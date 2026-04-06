"""Metric tracking for binary segmentation using SMP metrics.

Uses micro-averaging (pool all pixels, then compute) which is more
stable than per-image averaging when road coverage is ~4%.
"""

from __future__ import annotations

from typing import Dict

import torch
from segmentation_models_pytorch.metrics import (
    f1_score,
    get_stats,
    iou_score,
    precision,
    recall,
)


class MetricTracker:
    """Accumulates TP/FP/FN/TN across batches and computes epoch-level metrics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._tp: int = 0
        self._fp: int = 0
        self._fn: int = 0
        self._tn: int = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update stats from a batch.

        Args:
            y_pred: Raw logits ``(N, 1, H, W)``.
            y_true: Binary targets ``(N, 1, H, W)``.
        """
        tp, fp, fn, tn = get_stats(
            y_pred, y_true.long(), mode="binary", threshold=self.threshold,
        )
        self._tp += int(tp.sum().item())
        self._fp += int(fp.sum().item())
        self._fn += int(fn.sum().item())
        self._tn += int(tn.sum().item())

    def compute(self) -> Dict[str, float]:
        """Compute IoU, Dice, Precision, Recall from accumulated stats."""
        # Reshape to (1, 1) for SMP metric functions
        tp = torch.tensor([[self._tp]], dtype=torch.long)
        fp = torch.tensor([[self._fp]], dtype=torch.long)
        fn = torch.tensor([[self._fn]], dtype=torch.long)
        tn = torch.tensor([[self._tn]], dtype=torch.long)

        return {
            "iou": iou_score(tp, fp, fn, tn, reduction="micro").item(),
            "dice": f1_score(tp, fp, fn, tn, reduction="micro").item(),
            "precision": precision(tp, fp, fn, tn, reduction="micro").item(),
            "recall": recall(tp, fp, fn, tn, reduction="micro").item(),
        }
