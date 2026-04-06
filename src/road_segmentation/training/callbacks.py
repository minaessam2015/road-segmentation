"""Training callbacks: early stopping and model EMA."""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs with no improvement before stopping.
        mode: ``"max"`` (higher is better, e.g. IoU) or ``"min"`` (e.g. loss).
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self._best: Optional[float] = None

        if mode == "max":
            self._is_better = lambda new, best: new > best + min_delta
        else:
            self._is_better = lambda new, best: new < best - min_delta

    def __call__(self, metric_value: float) -> bool:
        """Returns True if training should stop."""
        if self._best is None or self._is_better(metric_value, self._best):
            self._best = metric_value
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered: no improvement for {self.patience} epochs. "
                f"Best {self.mode}: {self._best:.4f}"
            )
            return True
        return False

    @property
    def best_value(self) -> Optional[float]:
        return self._best


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy updated as:
        ``shadow = decay * shadow + (1 - decay) * current``

    Use ``self.module`` for evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    @property
    def module(self) -> nn.Module:
        return self.shadow

