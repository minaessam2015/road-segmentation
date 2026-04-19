"""Connectivity analysis for road segmentation.

Pixel IoU does not measure routability. A prediction with a 3-pixel
break in a road has nearly identical IoU to a continuous one but is
unusable for navigation. This module quantifies connectivity by
comparing connected components between GT and prediction.

Two signals per sample:
    - connectivity_fraction: 1 − (fragmented GT components / total GT components)
    - component_ratio:       n_pred_components / n_gt_components
                              (1.0 = ideal; >1 = over-segmentation, <1 = missed roads)

A GT component is "fragmented" when the prediction covers it using
≥ 2 predicted components that each contain ≥ ``min_fragment_pct`` of
the GT component's pixel area. Small spurious predicted pieces are
ignored — they are noise, not fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ConnectivityResult:
    sample_id: str
    n_gt_components: int
    n_pred_components: int
    n_fragmented: int
    connectivity_fraction: float  # 1 − fragmented / n_gt
    component_ratio: float         # n_pred / n_gt (or np.inf if n_gt == 0)
    # Detail per GT component — useful for failure gallery
    gt_component_sizes: List[int]
    fragments_per_gt: List[int]    # how many pred components split each GT


def _components(binary: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Wrapper around cv2.connectedComponentsWithStats with 8-connectivity.

    Returns:
        n_components: int (including background label 0)
        labels:       (H, W) int32 — label 0 is background
        stats:        (n_components, 5) — (x, y, w, h, area)
    """
    mask = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n, labels, stats


def analyze_sample(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_id: str = "",
    min_fragment_pct: float = 0.10,
    min_component_area: int = 50,
) -> ConnectivityResult:
    """Compute connectivity metrics for a single (gt, pred) pair.

    Args:
        gt_mask:         (H, W) uint8 — binary ground truth
        pred_mask:       (H, W) uint8 — binary prediction (same threshold as main IoU)
        min_fragment_pct: a predicted component must cover ≥ this fraction of a GT
                          component to count as a "non-trivial" fragment
        min_component_area: ignore GT/pred components smaller than this (spurious)

    Returns:
        ConnectivityResult with per-sample + per-GT-component details.
    """
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)

    n_gt, gt_labels, gt_stats = _components(gt_bin)
    n_pred, pred_labels, pred_stats = _components(pred_bin)

    # Filter tiny components (label 0 is background — skip it)
    valid_gt_ids = [
        i for i in range(1, n_gt)
        if gt_stats[i, cv2.CC_STAT_AREA] >= min_component_area
    ]
    valid_pred_ids = [
        i for i in range(1, n_pred)
        if pred_stats[i, cv2.CC_STAT_AREA] >= min_component_area
    ]

    # Count predicted components that land on each GT component
    fragments_per_gt: List[int] = []
    gt_sizes: List[int] = []
    n_fragmented = 0

    for gt_id in valid_gt_ids:
        gt_component = gt_labels == gt_id
        gt_size = int(gt_component.sum())
        gt_sizes.append(gt_size)

        # Count non-trivial predicted components overlapping this GT component
        overlaps = pred_labels[gt_component]
        # Unique pred IDs touching this GT, with their overlap count
        unique_ids, counts = np.unique(overlaps[overlaps > 0], return_counts=True)
        # Keep only valid + non-trivial pred pieces (≥ min_fragment_pct of gt_size)
        non_trivial = [
            pid for pid, cnt in zip(unique_ids, counts)
            if pid in valid_pred_ids and cnt >= min_fragment_pct * gt_size
        ]
        frag_count = len(non_trivial)
        fragments_per_gt.append(frag_count)
        if frag_count >= 2:
            n_fragmented += 1

    total_gt = len(valid_gt_ids)
    total_pred = len(valid_pred_ids)

    if total_gt == 0:
        connectivity_fraction = 1.0  # nothing to fragment
        component_ratio = float("inf") if total_pred > 0 else 1.0
    else:
        connectivity_fraction = 1.0 - n_fragmented / total_gt
        component_ratio = total_pred / total_gt

    return ConnectivityResult(
        sample_id=sample_id,
        n_gt_components=total_gt,
        n_pred_components=total_pred,
        n_fragmented=n_fragmented,
        connectivity_fraction=round(connectivity_fraction, 4),
        component_ratio=round(component_ratio, 4) if not np.isinf(component_ratio) else float("inf"),
        gt_component_sizes=gt_sizes,
        fragments_per_gt=fragments_per_gt,
    )


def aggregate(results: List[ConnectivityResult]) -> Dict[str, float]:
    """Aggregate per-sample connectivity metrics into dataset-level summary."""
    if not results:
        return {}

    # Exclude samples with no GT components from the connectivity fraction mean
    frac_values = [r.connectivity_fraction for r in results if r.n_gt_components > 0]
    # Component ratio: ignore inf (pred on no-GT samples)
    ratio_values = [r.component_ratio for r in results
                    if r.n_gt_components > 0 and not np.isinf(r.component_ratio)]

    total_gt = sum(r.n_gt_components for r in results)
    total_pred = sum(r.n_pred_components for r in results)
    total_fragmented = sum(r.n_fragmented for r in results)

    return {
        "mean_connectivity_fraction": float(np.mean(frac_values)) if frac_values else 0.0,
        "median_connectivity_fraction": float(np.median(frac_values)) if frac_values else 0.0,
        "pct_perfectly_connected": float((np.array(frac_values) == 1.0).mean()) if frac_values else 0.0,
        "mean_component_ratio": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "median_component_ratio": float(np.median(ratio_values)) if ratio_values else 0.0,
        "total_gt_components": total_gt,
        "total_pred_components": total_pred,
        "total_fragmented": total_fragmented,
        "dataset_fragmentation_rate": total_fragmented / total_gt if total_gt > 0 else 0.0,
        "n_samples": len(results),
    }
