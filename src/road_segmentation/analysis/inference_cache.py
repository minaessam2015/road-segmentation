"""Resumable inference cache for metric analysis.

Runs the model on the full validation set and caches each prediction
as a separate .npz file. Re-running skips already-cached samples,
so Colab disconnects or crashes don't lose progress.

Cache layout:
    <cache_dir>/
        predictions/
            {sample_id}.npz   # prob_map (float16), gt_mask (uint8)
        metadata.json          # model version, checkpoint path, image_size

Each .npz contains:
    prob_map: (H, W) float16 — sigmoid probability, resized to original size
    gt_mask:  (H, W) uint8   — binary ground truth {0, 1}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from road_segmentation.data.eda import ImageMaskPair


@dataclass
class CacheStats:
    total_samples: int
    already_cached: int
    newly_cached: int
    failed: int
    elapsed_s: float

    @property
    def processed(self) -> int:
        return self.newly_cached + self.failed


def _sanitize_id(sample_id: str) -> str:
    """Make sample ID safe for filenames (strip path separators)."""
    return sample_id.replace("/", "_").replace("\\", "_")


def get_cache_dir(base_dir: Path) -> Path:
    """Return the predictions subdirectory, creating it if needed."""
    preds = Path(base_dir) / "predictions"
    preds.mkdir(parents=True, exist_ok=True)
    return preds


def cache_path_for(sample_id: str, base_dir: Path) -> Path:
    return get_cache_dir(base_dir) / f"{_sanitize_id(sample_id)}.npz"


def is_cached(sample_id: str, base_dir: Path) -> bool:
    return cache_path_for(sample_id, base_dir).exists()


def load_cached(sample_id: str, base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached (prob_map, gt_mask) for a sample.

    Returns:
        prob_map: (H, W) float32 (promoted from stored float16)
        gt_mask:  (H, W) uint8 {0, 1}
    """
    data = np.load(cache_path_for(sample_id, base_dir))
    return data["prob_map"].astype(np.float32), data["gt_mask"]


def _load_sample(pair: ImageMaskPair) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load image + GT mask as arrays.

    Returns:
        image_rgb: (H, W, 3) uint8
        gt_mask:   (H, W) uint8 {0, 1}
        size:      (W, H)
    """
    image = np.array(Image.open(pair.image_path).convert("RGB"))
    mask = np.array(Image.open(pair.mask_path).convert("L"))
    gt = (mask > 0).astype(np.uint8)
    h, w = image.shape[:2]
    return image, gt, (w, h)


def _preprocess(
    image: np.ndarray,
    image_size: int,
    mean: List[float],
    std: List[float],
    device: torch.device,
) -> torch.Tensor:
    """Resize, normalize (ImageNet stats), return (1, 3, image_size, image_size) tensor."""
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def build_predictions_cache(
    val_pairs: List[ImageMaskPair],
    model: torch.nn.Module,
    base_dir: Path,
    device: torch.device,
    image_size: int = 1024,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> CacheStats:
    """Run inference and cache predictions for each val sample.

    Skips samples that already have a cached .npz.
    Safe to interrupt — will resume from where it left off.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata once
    meta_path = base_dir / "metadata.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump({"image_size": image_size, "mean": mean, "std": std}, f, indent=2)

    model.eval()
    start = time.time()
    already = 0
    newly = 0
    failed = 0

    pbar = tqdm(val_pairs, desc="Caching predictions")
    for pair in pbar:
        if is_cached(pair.sample_id, base_dir):
            already += 1
            pbar.set_postfix({"cached": already, "new": newly, "failed": failed})
            continue

        try:
            image, gt, (orig_w, orig_h) = _load_sample(pair)
            tensor = _preprocess(image, image_size, mean, std, device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)

            prob_map = probs[0, 0].cpu().numpy()  # (image_size, image_size)

            # Resize back to original dimensions
            if (orig_w, orig_h) != (image_size, image_size):
                prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Save as float16 to halve the disk footprint
            np.savez_compressed(
                cache_path_for(pair.sample_id, base_dir),
                prob_map=prob_map.astype(np.float16),
                gt_mask=gt.astype(np.uint8),
            )
            newly += 1
        except Exception as e:
            failed += 1
            print(f"Failed {pair.sample_id}: {e}")

        pbar.set_postfix({"cached": already, "new": newly, "failed": failed})
        if on_progress:
            on_progress(already + newly + failed, len(val_pairs))

    elapsed = time.time() - start
    return CacheStats(
        total_samples=len(val_pairs),
        already_cached=already,
        newly_cached=newly,
        failed=failed,
        elapsed_s=elapsed,
    )


def iterate_cached(
    val_pairs: List[ImageMaskPair],
    base_dir: Path,
) -> Callable:
    """Generator yielding (sample_id, prob_map, gt_mask) for cached samples."""
    def _gen():
        for pair in val_pairs:
            if is_cached(pair.sample_id, base_dir):
                prob, gt = load_cached(pair.sample_id, base_dir)
                yield pair.sample_id, prob, gt
    return _gen
