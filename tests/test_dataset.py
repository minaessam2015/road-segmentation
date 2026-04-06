"""Tests for data loading and preprocessing."""

from __future__ import annotations

import numpy as np
import torch

from road_segmentation.data.transforms import build_transforms, get_val_transform


class TestTransforms:
    def test_val_transform_output_shape(self, sample_image_rgb, sample_mask_binary):
        transform = get_val_transform(
            image_size=256,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        result = transform(image=sample_image_rgb, mask=sample_mask_binary)
        assert result["image"].shape == (3, 256, 256)
        assert result["mask"].shape == (256, 256)

    def test_val_transform_normalizes(self, sample_image_rgb, sample_mask_binary):
        transform = get_val_transform(256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        result = transform(image=sample_image_rgb, mask=sample_mask_binary)
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert result["image"].min() >= -4.0
        assert result["image"].max() <= 4.0

    def test_val_transform_preserves_mask_values(self, sample_image_rgb, sample_mask_binary):
        transform = get_val_transform(256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        result = transform(image=sample_image_rgb, mask=sample_mask_binary)
        unique = torch.unique(result["mask"]).tolist()
        # Mask should still be binary (0 and 255 mapped to 0 and 255 as uint8, or 0.0/1.0)
        assert all(v in [0, 1, 255, 0.0, 1.0, 255.0] for v in unique)

    def test_build_transforms_with_augmentations(self, sample_image_rgb, sample_mask_binary):
        aug_steps = [
            {"name": "HorizontalFlip", "params": {"p": 1.0}},
        ]
        transform = build_transforms(aug_steps, 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        result = transform(image=sample_image_rgb, mask=sample_mask_binary)
        assert result["image"].shape == (3, 256, 256)

    def test_build_transforms_resize(self, sample_image_rgb, sample_mask_binary):
        """Ensure resize works when input is not the target size."""
        big_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        big_mask = np.zeros((1024, 1024), dtype=np.uint8)
        transform = get_val_transform(512, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        result = transform(image=big_image, mask=big_mask)
        assert result["image"].shape == (3, 512, 512)
        assert result["mask"].shape == (512, 512)
