"""Training entrypoint for road segmentation.

Usage:
    python scripts/train.py --config configs/unet_resnet34.yaml
    python scripts/train.py --config configs/unet_resnet34.yaml \\
        --override training.epochs=10 data.batch_size=4 data.subset_size=100
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_segmentation.config import apply_overrides, load_config
from road_segmentation.data.dataset import create_dataloaders
from road_segmentation.data.eda import discover_image_mask_pairs
from road_segmentation.data.split import split_pairs
from road_segmentation.data.transforms import get_train_transform, get_val_transform
from road_segmentation.models.factory import count_parameters, create_model, freeze_encoder
from road_segmentation.paths import DEEPGLOBE_DATASET_DIR
from road_segmentation.training.losses import create_loss
from road_segmentation.training.trainer import Trainer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train road segmentation model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key=value")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load and override config
    config = load_config(args.config)
    if args.override:
        apply_overrides(config, args.override)
        logger.info(f"Applied {len(args.override)} config overrides")

    seed_everything(config.seed)
    device = get_device()

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    dataset_dir = Path(config.data.dataset_dir) if config.data.dataset_dir else DEEPGLOBE_DATASET_DIR
    logger.info(f"Loading dataset from: {dataset_dir}")

    pairs = discover_image_mask_pairs(dataset_dir)
    logger.info(f"Found {len(pairs)} image-mask pairs")

    train_pairs, val_pairs = split_pairs(
        pairs,
        val_ratio=config.data.val_split_ratio,
        seed=config.data.val_split_seed,
        subset_size=config.data.subset_size,
    )

    train_transform = get_train_transform(
        image_size=config.data.image_size,
        aug_steps=config.augmentations.train,
        mean=config.normalization.mean,
        std=config.normalization.std,
    )
    val_transform = get_val_transform(
        image_size=config.data.image_size,
        mean=config.normalization.mean,
        std=config.normalization.std,
    )

    train_loader, val_loader = create_dataloaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    logger.info(f"DataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = create_model(
        arch=config.model.arch,
        encoder_name=config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        in_channels=config.model.in_channels,
        classes=config.model.classes,
        **config.model.decoder_kwargs,
    )

    if config.training.freeze_encoder_epochs > 0:
        freeze_encoder(model)

    param_counts = count_parameters(model)
    logger.info(
        f"Model: {config.model.arch} + {config.model.encoder_name} | "
        f"Total params: {param_counts['total'] / 1e6:.1f}M | "
        f"Trainable: {param_counts['trainable'] / 1e6:.1f}M"
    )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    loss_fn = create_loss(config.loss.type, config.loss.params)
    logger.info(f"Loss: {config.loss.type}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    best_metrics = trainer.train()
    logger.info(f"Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
