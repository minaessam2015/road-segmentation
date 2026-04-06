"""Training visualization: curves and prediction overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_image(
    image: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """Convert a normalized ``(C, H, W)`` tensor back to ``(H, W, C)`` uint8."""
    img = image.cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def plot_training_curves(
    history: List[Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training curves in a 2x2 grid.

    Top-left: train/val loss.  Top-right: IoU and Dice.
    Bottom-left: Precision and Recall.  Bottom-right: Learning rate.
    """
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="Train loss")
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="Val loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # IoU & Dice
    axes[0, 1].plot(epochs, [h["val_iou"] for h in history], label="Val IoU")
    axes[0, 1].plot(epochs, [h["val_dice"] for h in history], label="Val Dice")
    axes[0, 1].set_title("Segmentation Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision & Recall
    axes[1, 0].plot(epochs, [h["val_precision"] for h in history], label="Precision")
    axes[1, 0].plot(epochs, [h["val_recall"] for h in history], label="Recall")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    axes[1, 1].plot(epochs, [h["lr"] for h in history], color="green")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_samples(
    images: torch.Tensor,
    masks_gt: torch.Tensor,
    masks_pred: torch.Tensor,
    mean: List[float],
    std: List[float],
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize predictions vs ground truth.

    For each sample shows 4 columns:
      1. Input image (denormalized)
      2. Ground truth mask
      3. Predicted mask (thresholded)
      4. Error overlay: TP=green, FP=red, FN=blue on the input image

    Args:
        images: ``(N, C, H, W)`` normalized tensors.
        masks_gt: ``(N, 1, H, W)`` binary ground truth.
        masks_pred: ``(N, 1, H, W)`` sigmoid probabilities.
        mean, std: Normalization stats for denormalization.
        threshold: Binarization threshold for predictions.
        save_path: Optional path to save the figure.
    """
    n_samples = min(len(images), 8)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction", "Error Map (TP/FP/FN)"]

    for i in range(n_samples):
        img = denormalize_image(images[i], mean, std)
        gt = masks_gt[i, 0].cpu().numpy().astype(bool)
        pred = (masks_pred[i, 0].cpu().numpy() >= threshold)

        # Error overlay
        overlay = img.copy().astype(np.float32)
        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt

        # Blend: TP=green, FP=red, FN=blue (alpha=0.5)
        alpha = 0.5
        overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
        overlay[fp] = overlay[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 255]) * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 3].imshow(overlay)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig
