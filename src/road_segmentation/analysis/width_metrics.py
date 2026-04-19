"""Width-stratified IoU.

Aggregate IoU is pixel-weighted: highways (wide roads) dominate residential
streets (thin roads) by orders of magnitude in pixel count. This module
breaks IoU out by road width to expose long-tail failure on thin roads.

Two bucket strategies:
    - FIXED thresholds:   narrow ≤10px, medium 10–30px, wide >30px
    - PERCENTILE buckets: p33 and p66 cuts on the actual pixel-width distribution

Two FP attribution strategies (FPs have no "true" width):
    - FP_IN:  attribute each FP to the bucket defined by the distance transform
              of the GT mask evaluated at the FP pixel's location
              (how close it is to real roads) — full IoU per bucket
    - FP_OUT: exclude FPs, report recall-only per bucket (TP / (TP + FN))
              — measures how well the model finds each width class

Both variants are computed — the notebook presents whichever is clearer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

BUCKET_NAMES_FIXED = ("narrow", "medium", "wide")
FIXED_EDGES = (10.0, 30.0)  # pixel width cut-offs


def fixed_bucket(width_px: np.ndarray) -> np.ndarray:
    """Map an array of widths to {0=narrow, 1=medium, 2=wide} using fixed edges."""
    out = np.zeros_like(width_px, dtype=np.int8)
    out[(width_px > FIXED_EDGES[0]) & (width_px <= FIXED_EDGES[1])] = 1
    out[width_px > FIXED_EDGES[1]] = 2
    return out


def percentile_edges(widths: np.ndarray, n_buckets: int = 3) -> np.ndarray:
    """Compute bucket edges from percentile cuts of a width distribution.

    For n_buckets=3, returns the (33.3, 66.7) percentiles.
    """
    pcts = np.linspace(0, 100, n_buckets + 1)[1:-1]
    return np.percentile(widths, pcts)


def percentile_bucket(width_px: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map widths to bucket indices using custom edges."""
    return np.digitize(width_px, edges).astype(np.int8)


def collect_pixel_widths(
    gt_masks: Iterable[np.ndarray],
    max_samples: int = 10_000_000,
) -> np.ndarray:
    """Concatenate the per-road-pixel width distribution across all GTs.

    Width at each road pixel = 2 × distance_transform_edt (diameter estimate).
    Caps at ``max_samples`` pixels (reservoir-sample) to keep memory bounded.
    """
    collected: List[np.ndarray] = []
    rng = np.random.default_rng(42)
    remaining = max_samples

    for gt in gt_masks:
        if gt.sum() == 0:
            continue
        dist = distance_transform_edt(gt.astype(np.uint8))
        widths = 2.0 * dist[gt > 0]  # only road pixels
        if widths.size > remaining:
            idx = rng.choice(widths.size, size=remaining, replace=False)
            widths = widths[idx]
        collected.append(widths)
        remaining -= widths.size
        if remaining <= 0:
            break

    if not collected:
        return np.array([], dtype=np.float32)
    return np.concatenate(collected).astype(np.float32)


@dataclass
class WidthBucketStats:
    """Accumulators for a single bucket across all samples."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0


def stratify_sample(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    bucket_fn,
    n_buckets: int = 3,
) -> Tuple[Dict[int, WidthBucketStats], Dict[int, WidthBucketStats]]:
    """Compute per-bucket TP/FP/FN for a single sample, two FP strategies.

    Args:
        prob_map:   (H, W) float32 sigmoid probability
        gt_mask:    (H, W) uint8 binary ground truth
        threshold:  binarization threshold
        bucket_fn:  callable width_px -> bucket index
        n_buckets:  number of buckets (for result dict initialization)

    Returns:
        fp_in_stats:  per-bucket stats where FPs are attributed via GT-distance
                      lookup at the FP pixel's location
        fp_out_stats: per-bucket stats where FPs are excluded (recall-only)
    """
    pred = (prob_map >= threshold).astype(np.uint8)
    gt = gt_mask.astype(np.uint8)

    # GT distance transform — distance from each pixel to the nearest non-road
    # pixel. At road pixels this is their local "radius", so width ≈ 2×distance.
    if gt.sum() > 0:
        dist_gt = distance_transform_edt(gt)
    else:
        dist_gt = np.zeros_like(gt, dtype=np.float32)

    # TP / FN widths come from the GT at each positive GT pixel
    # FP bucketing (FP_IN variant) uses the distance of FP pixels *within a
    # dilated GT neighborhood* — FPs far from any GT road get bucketed by
    # however close they are to one, so they still land in a bucket.
    width_at_px = 2.0 * dist_gt

    tp_mask = (pred == 1) & (gt == 1)
    fp_mask = (pred == 1) & (gt == 0)
    fn_mask = (pred == 0) & (gt == 1)

    # For FP pixels, use distance from background pixel to nearest road × 2.
    # This attributes FPs near wide roads to the wide bucket (likely boundary
    # over-prediction) and FPs far from any road to the narrow bucket
    # (likely hallucination). For background FPs, distance is to nearest road
    # (from dist_bg), which equates to how "close" the FP is to a real road.
    if fp_mask.any():
        # Distance from background pixels to nearest road pixel
        dist_bg = distance_transform_edt(1 - gt) if gt.sum() > 0 else np.full_like(gt, 1000, dtype=np.float32)
        # Look up local GT road width at the *nearest* GT pixel for each FP.
        # Approximation: use 2 × dist_bg as a proxy for FP bucket — smaller
        # means "FP is close to a road" (likely boundary error on a thin road).
        # For our use, just bucket FPs by `2 × dist_bg`: FPs *very* close to
        # a road are boundary errors; FPs far from any road are hallucinations.
        fp_widths = 2.0 * dist_bg[fp_mask]
    else:
        fp_widths = np.array([], dtype=np.float32)

    tp_widths = width_at_px[tp_mask]
    fn_widths = width_at_px[fn_mask]

    tp_buckets = bucket_fn(tp_widths) if tp_widths.size else np.array([], dtype=np.int8)
    fn_buckets = bucket_fn(fn_widths) if fn_widths.size else np.array([], dtype=np.int8)
    fp_buckets = bucket_fn(fp_widths) if fp_widths.size else np.array([], dtype=np.int8)

    fp_in_stats = {i: WidthBucketStats() for i in range(n_buckets)}
    fp_out_stats = {i: WidthBucketStats() for i in range(n_buckets)}

    for i in range(n_buckets):
        tp_count = int((tp_buckets == i).sum())
        fn_count = int((fn_buckets == i).sum())
        fp_count = int((fp_buckets == i).sum())

        fp_in_stats[i].tp = tp_count
        fp_in_stats[i].fn = fn_count
        fp_in_stats[i].fp = fp_count

        # FP_OUT: same TP/FN, zero FPs
        fp_out_stats[i].tp = tp_count
        fp_out_stats[i].fn = fn_count
        fp_out_stats[i].fp = 0

    return fp_in_stats, fp_out_stats


def aggregate_bucket_stats(
    sample_stats: List[Dict[int, WidthBucketStats]],
    n_buckets: int = 3,
) -> Dict[int, WidthBucketStats]:
    """Sum per-sample bucket stats into a single global bucket stat (micro-avg)."""
    out = {i: WidthBucketStats() for i in range(n_buckets)}
    for per_sample in sample_stats:
        for i, s in per_sample.items():
            out[i].tp += s.tp
            out[i].fp += s.fp
            out[i].fn += s.fn
    return out


def format_bucket_table(
    stats: Dict[int, WidthBucketStats],
    bucket_labels: List[str],
) -> List[Dict[str, float]]:
    """Return a list of dicts suitable for pandas.DataFrame."""
    rows = []
    for i, label in enumerate(bucket_labels):
        s = stats[i]
        rows.append({
            "bucket": label,
            "tp": s.tp,
            "fp": s.fp,
            "fn": s.fn,
            "iou": round(s.iou(), 4),
            "precision": round(s.precision(), 4),
            "recall": round(s.recall(), 4),
        })
    return rows
