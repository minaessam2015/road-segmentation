"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Random 256x256 RGB image as uint8 numpy array."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask_binary() -> np.ndarray:
    """Random 256x256 binary mask with ~5% road pixels."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Draw some "road" lines
    mask[100:105, 50:200] = 255  # horizontal road
    mask[50:200, 128:133] = 255  # vertical road
    return mask


@pytest.fixture
def sample_prob_map() -> np.ndarray:
    """Synthetic probability map with known road regions."""
    prob = np.random.uniform(0.0, 0.2, (256, 256)).astype(np.float32)
    prob[100:105, 50:200] = np.random.uniform(0.7, 0.95, (5, 150)).astype(np.float32)
    prob[50:200, 128:133] = np.random.uniform(0.7, 0.95, (150, 5)).astype(np.float32)
    return prob


@pytest.fixture
def sample_image_bytes(sample_image_rgb) -> bytes:
    """Encode sample image as PNG bytes."""
    img = Image.fromarray(sample_image_rgb)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_model():
    """Small U-Net model for testing (not pretrained, random weights)."""
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.eval()
    return model


@pytest.fixture
def tmp_checkpoint(tmp_path, sample_model) -> Path:
    """Save a dummy checkpoint for testing."""
    ckpt_path = tmp_path / "test_checkpoint.pth"
    torch.save({
        "model_state_dict": sample_model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 5,
        "best_metric": 0.55,
        "best_metric_name": "val_iou",
        "config": {
            "model": {
                "arch": "Unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            },
            "data": {"image_size": 256},
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }, ckpt_path)
    return ckpt_path
