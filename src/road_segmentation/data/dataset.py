"""PyTorch Dataset and DataLoader factory for road segmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from road_segmentation.data.eda import ImageMaskPair


class RoadSegmentationDataset(Dataset):
    """PyTorch Dataset that loads satellite images and binary road masks.

    - Images are loaded as RGB ``(H, W, 3)`` uint8 arrays.
    - Masks are loaded as single-channel ``(H, W)`` uint8 arrays with
      values ``{0, 1}`` (converted from the original RGB masks where
      road = white and background = black).
    - An Albumentations pipeline is applied jointly to image and mask
      (spatial transforms affect both; photometric transforms only
      affect the image).
    - Returns ``{"image": (C, H, W) float32, "mask": (1, H, W) float32}``.
    """

    def __init__(
        self,
        pairs: List[ImageMaskPair],
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Load image as RGB numpy array
        image = np.array(Image.open(pair.image_path).convert("RGB"))

        # Load mask: RGB -> grayscale -> binary {0, 1}
        mask = np.array(Image.open(pair.mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]   # (C, H, W) float32 after ToTensorV2
            mask = transformed["mask"]     # (H, W) tensor

        # Ensure mask shape is (1, H, W) float32 for BCEWithLogitsLoss
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": image, "mask": mask}


def create_dataloaders(
    train_pairs: List[ImageMaskPair],
    val_pairs: List[ImageMaskPair],
    train_transform: A.Compose,
    val_transform: A.Compose,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_dataset = RoadSegmentationDataset(train_pairs, transform=train_transform)
    val_dataset = RoadSegmentationDataset(val_pairs, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
