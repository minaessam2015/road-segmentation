"""Model creation and encoder management using segmentation_models_pytorch."""

from __future__ import annotations

from typing import Any, List, Optional

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **decoder_kwargs: Any,
) -> nn.Module:
    """Create a segmentation model via SMP.

    Args:
        arch: SMP architecture name (e.g. "Unet", "DeepLabV3Plus", "Linknet").
        encoder_name: Backbone encoder name (e.g. "resnet34", "efficientnet-b0").
        encoder_weights: Pretrained weights source (e.g. "imagenet") or None.
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output classes (1 for binary segmentation).
        **decoder_kwargs: Additional arguments forwarded to SMP.

    Returns:
        An ``nn.Module`` segmentation model.
    """
    model = smp.create_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **decoder_kwargs,
    )
    return model


def get_encoder_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return encoder parameters for differential learning rates."""
    return list(model.encoder.parameters())


def get_decoder_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return decoder + segmentation head parameters."""
    params = list(model.decoder.parameters())
    if hasattr(model, "segmentation_head"):
        params.extend(model.segmentation_head.parameters())
    return params


def freeze_encoder(model: nn.Module) -> None:
    """Freeze all encoder parameters (set requires_grad=False)."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
