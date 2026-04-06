"""Albumentations augmentation pipelines built from config."""

from __future__ import annotations

from typing import Any, Dict, List

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _resolve_augmentation(name: str, params: Dict[str, Any]) -> A.BasicTransform:
    """Resolve an augmentation by name from the albumentations module."""
    if not hasattr(A, name):
        raise ValueError(
            f"Unknown augmentation '{name}'. "
            f"Must be a valid albumentations class (e.g. HorizontalFlip, RandomRotate90)."
        )
    cls = getattr(A, name)
    return cls(**params)


def build_transforms(
    aug_steps: List[Dict[str, Any]],
    image_size: int,
    mean: List[float],
    std: List[float],
) -> A.Compose:
    """Build an Albumentations pipeline from a list of augmentation steps.

    The pipeline always starts with Resize and ends with Normalize + ToTensorV2.
    The aug_steps in between are resolved dynamically from config.
    """
    transforms = [A.Resize(image_size, image_size)]

    for step in aug_steps:
        name = step.get("name", "") if isinstance(step, dict) else step.name
        params = step.get("params", {}) if isinstance(step, dict) else step.params
        transforms.append(_resolve_augmentation(name, params))

    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_train_transform(
    image_size: int,
    aug_steps: List[Any],
    mean: List[float],
    std: List[float],
) -> A.Compose:
    """Build the training transform pipeline."""
    return build_transforms(aug_steps, image_size, mean, std)


def get_val_transform(
    image_size: int,
    mean: List[float],
    std: List[float],
) -> A.Compose:
    """Build the validation transform: Resize + Normalize + ToTensorV2 only."""
    return build_transforms([], image_size, mean, std)
