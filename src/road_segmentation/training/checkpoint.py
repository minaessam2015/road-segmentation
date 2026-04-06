"""Checkpoint save/load with full training state for resume support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Metadata restored from a checkpoint."""
    epoch: int
    best_metric: float
    best_metric_name: str
    config_dict: Dict[str, Any]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_metric: float,
    best_metric_name: str,
    config_dict: Dict[str, Any],
    ema_model: Optional[nn.Module] = None,
) -> None:
    """Save full training state to a ``.pth`` file.

    Contents: model weights, optimizer state, scheduler state, AMP scaler
    state, epoch counter, best metric, and the full config dict for
    reproducibility and resume compatibility checking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "config": config_dict,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    if ema_model is not None:
        state["ema_state_dict"] = ema_model.state_dict()

    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema_model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> CheckpointState:
    """Load a checkpoint and restore training state.

    Model weights are always loaded. Optimizer, scheduler, and scaler
    states are restored only if the corresponding objects are provided
    (pass ``None`` for inference-only loading).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Model weights loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    if ema_model is not None and "ema_state_dict" in state:
        ema_model.load_state_dict(state["ema_state_dict"])

    saved_config = state.get("config", {})
    return CheckpointState(
        epoch=state["epoch"],
        best_metric=state["best_metric"],
        best_metric_name=state.get("best_metric_name", "val_iou"),
        config_dict=saved_config,
    )


def verify_config_compatibility(
    saved_config: Dict[str, Any],
    current_config: Dict[str, Any],
) -> None:
    """Warn if critical config fields differ between saved and current.

    Critical fields: model architecture, encoder, classes, image size.
    """
    critical_paths = [
        ("model", "arch"),
        ("model", "encoder_name"),
        ("model", "classes"),
        ("data", "image_size"),
    ]

    for section, key in critical_paths:
        saved_val = saved_config.get(section, {}).get(key)
        current_val = current_config.get(section, {}).get(key)
        if saved_val is not None and current_val is not None and saved_val != current_val:
            logger.warning(
                f"Config mismatch on {section}.{key}: "
                f"checkpoint={saved_val}, current={current_val}"
            )
