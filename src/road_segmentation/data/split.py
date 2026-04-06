"""Train/validation splitting with stratification by road coverage."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from road_segmentation.data.eda import ImageMaskPair

logger = logging.getLogger(__name__)


def _compute_coverage(pair: ImageMaskPair) -> float:
    """Compute the fraction of road pixels in a mask."""
    with Image.open(pair.mask_path) as mask:
        arr = np.array(mask.convert("L"))
    return float((arr > 0).mean())


def _compute_coverage_bins(
    coverages: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Bin coverage values into quantile-based groups for stratification.

    Merges bins with fewer than 2 samples into adjacent bins to prevent
    sklearn's stratified split from failing.
    """
    # Use quantile-based edges, but clamp to actual data range
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(coverages, percentiles)
    edges = np.unique(edges)  # remove duplicates from tied quantiles

    bins = np.digitize(coverages, edges[1:-1])  # n_bins - 1 internal edges

    # Merge any bin with < 2 samples into the previous bin
    unique, counts = np.unique(bins, return_counts=True)
    for b, c in zip(unique, counts):
        if c < 2 and b > 0:
            bins[bins == b] = b - 1

    return bins


def split_pairs(
    pairs: List[ImageMaskPair],
    val_ratio: float = 0.15,
    seed: int = 42,
    subset_size: Optional[int] = None,
) -> Tuple[List[ImageMaskPair], List[ImageMaskPair]]:
    """Split image-mask pairs into train and validation sets.

    Uses stratification by road coverage to ensure both splits have
    similar coverage distributions (important given the severe class
    imbalance — mean coverage ~4%).

    Args:
        pairs: All discovered image-mask pairs.
        val_ratio: Fraction for validation (default 0.15).
        seed: Random seed for reproducible splits.
        subset_size: If set, subsample to this many pairs first
                     (for quick debug runs).

    Returns:
        (train_pairs, val_pairs) tuple.
    """
    if subset_size is not None and subset_size < len(pairs):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(pairs), size=subset_size, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {subset_size} pairs for quick experiment.")

    logger.info(f"Computing road coverage for {len(pairs)} masks...")
    coverages = np.array([_compute_coverage(p) for p in pairs])

    # Fewer bins for small datasets to ensure each bin has >= 2 samples
    n_bins = min(10, max(2, len(pairs) // 10))
    bins = _compute_coverage_bins(coverages, n_bins=n_bins)

    # Fall back to non-stratified split if bins still have too few samples
    val_count = max(1, int(len(pairs) * val_ratio))
    unique_bins = np.unique(bins)
    if len(unique_bins) > val_count:
        logger.warning("Too many stratification bins for dataset size; using random split.")
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=val_ratio, random_state=seed,
        )
    else:
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=val_ratio, random_state=seed, stratify=bins,
        )

    logger.info(
        f"Split: {len(train_pairs)} train, {len(val_pairs)} val "
        f"(ratio={val_ratio:.2f})"
    )
    return train_pairs, val_pairs
