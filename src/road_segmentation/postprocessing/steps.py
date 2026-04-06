"""Individual post-processing steps, each independent and testable.

Every function takes a mask or probability map and returns a processed version.
Each step is designed to be used standalone or composed in a pipeline.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# =========================================================================
# Step 1: Threshold Optimization
# =========================================================================


def find_optimal_threshold(
    prob_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    thresholds: np.ndarray | None = None,
) -> Tuple[float, dict]:
    """Sweep thresholds on validation set and return the one maximizing IoU.

    Args:
        prob_maps: List of probability maps (H, W), float32 [0, 1].
        gt_masks: List of ground truth binary masks (H, W), {0, 1}.
        thresholds: Array of thresholds to try. Defaults to 0.20-0.80 in 0.05 steps.

    Returns:
        (best_threshold, results_dict) where results_dict maps threshold -> IoU.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.05)

    results = {}
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for prob, gt in zip(prob_maps, gt_masks):
            pred = (prob >= t).astype(np.uint8)
            gt_bin = (gt > 0).astype(np.uint8)
            tp += int(np.sum((pred == 1) & (gt_bin == 1)))
            fp += int(np.sum((pred == 1) & (gt_bin == 0)))
            fn += int(np.sum((pred == 0) & (gt_bin == 1)))
        iou = tp / max(tp + fp + fn, 1)
        results[round(float(t), 2)] = round(iou, 6)

    best_t = max(results, key=results.get)
    return best_t, results


def apply_threshold(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize a probability map at the given threshold.

    Returns: uint8 mask with values {0, 255}.
    """
    return ((prob_map >= threshold) * 255).astype(np.uint8)


# =========================================================================
# Step 2: Connected Component Filtering
# =========================================================================


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100,
) -> np.ndarray:
    """Remove connected components smaller than min_area pixels.

    Args:
        mask: Binary mask (H, W), uint8 {0, 255}.
        min_area: Minimum component area in pixels.

    Returns: Cleaned mask with small blobs removed.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned.astype(np.uint8)


# =========================================================================
# Step 3: Morphological Closing
# =========================================================================


def morphological_close(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply morphological closing to fill small gaps in roads.

    Uses an elliptical kernel for more natural road shapes.

    Args:
        mask: Binary mask (H, W), uint8 {0, 255}.
        kernel_size: Size of the structuring element.

    Returns: Closed mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morphological_open(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological opening to remove small noise.

    Args:
        mask: Binary mask (H, W), uint8 {0, 255}.
        kernel_size: Size of the structuring element.

    Returns: Opened mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# =========================================================================
# Step 4: Test-Time Augmentation
# =========================================================================


def apply_tta(
    model,
    image_tensor,
    device,
) -> np.ndarray:
    """Run 4-fold TTA (original + 3 rotations) and average probabilities.

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Single image tensor (1, C, H, W).
        device: torch device.

    Returns: Averaged probability map (H, W), float32 [0, 1].
    """
    import torch

    model.eval()
    preds = []

    with torch.no_grad():
        for k in range(4):  # 0, 90, 180, 270 degrees
            # Rotate input
            rotated = torch.rot90(image_tensor, k=k, dims=[2, 3])
            logits = model(rotated.to(device))
            prob = torch.sigmoid(logits)
            # Rotate prediction back
            prob_back = torch.rot90(prob, k=-k, dims=[2, 3])
            preds.append(prob_back.cpu())

        # Also do horizontal flip
        flipped = torch.flip(image_tensor, dims=[3])
        logits = model(flipped.to(device))
        prob = torch.sigmoid(logits)
        prob_back = torch.flip(prob, dims=[3])
        preds.append(prob_back.cpu())

        # Vertical flip
        flipped = torch.flip(image_tensor, dims=[2])
        logits = model(flipped.to(device))
        prob = torch.sigmoid(logits)
        prob_back = torch.flip(prob, dims=[2])
        preds.append(prob_back.cpu())

    # Average all 6 predictions
    avg = torch.stack(preds).mean(dim=0)
    return avg[0, 0].numpy()  # (H, W)


# =========================================================================
# Step 5: Skeletonize
# =========================================================================


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel-wide centerlines from a binary road mask.

    Args:
        mask: Binary mask (H, W), uint8 {0, 255}.

    Returns: Skeleton mask (H, W), uint8 {0, 255}.
    """
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


# =========================================================================
# Step 6: Graph Extraction and Cleanup
# =========================================================================


def skeleton_to_graph(skeleton: np.ndarray) -> dict:
    """Convert a skeleton image to a graph with nodes and edges.

    Uses contour detection to extract polylines from the skeleton.

    Args:
        skeleton: Skeleton mask (H, W), uint8 {0, 255}.

    Returns:
        Dict with "nodes" (list of [x, y]) and "edges" (list of polylines).
    """
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    edges = []
    nodes = set()

    for contour in contours:
        if len(contour) < 2:
            continue
        points = contour.squeeze().tolist()
        if isinstance(points[0], (int, float)):
            continue  # single point
        edges.append(points)
        # First and last points are nodes (endpoints/junctions)
        nodes.add(tuple(points[0]))
        nodes.add(tuple(points[-1]))

    return {
        "nodes": [list(n) for n in nodes],
        "edges": edges,
    }


def prune_short_branches(
    graph: dict,
    min_length_px: int = 20,
) -> dict:
    """Remove edges shorter than min_length_px.

    Args:
        graph: Dict with "nodes" and "edges".
        min_length_px: Minimum edge length in pixels.

    Returns: Pruned graph.
    """
    pruned_edges = []
    for edge in graph["edges"]:
        length = sum(
            np.sqrt((edge[i][0] - edge[i + 1][0]) ** 2 + (edge[i][1] - edge[i + 1][1]) ** 2)
            for i in range(len(edge) - 1)
        )
        if length >= min_length_px:
            pruned_edges.append(edge)

    # Recompute nodes from remaining edges
    nodes = set()
    for edge in pruned_edges:
        nodes.add(tuple(edge[0]))
        nodes.add(tuple(edge[-1]))

    return {
        "nodes": [list(n) for n in nodes],
        "edges": pruned_edges,
    }


def simplify_edges(
    graph: dict,
    tolerance: float = 2.0,
) -> dict:
    """Simplify edge geometry using Douglas-Peucker algorithm.

    Args:
        graph: Dict with "nodes" and "edges".
        tolerance: Simplification tolerance in pixels.

    Returns: Graph with simplified edges.
    """
    simplified_edges = []
    for edge in graph["edges"]:
        contour = np.array(edge, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, tolerance, closed=False)
        simplified = approx.squeeze().tolist()
        if isinstance(simplified[0], (int, float)):
            simplified = [simplified]
        if len(simplified) >= 2:
            simplified_edges.append(simplified)

    nodes = set()
    for edge in simplified_edges:
        nodes.add(tuple(edge[0]))
        nodes.add(tuple(edge[-1]))

    return {
        "nodes": [list(n) for n in nodes],
        "edges": simplified_edges,
    }


# =========================================================================
# Step 7: Confidence-Weighted Gap Bridging
# =========================================================================


def bridge_gaps(
    mask: np.ndarray,
    prob_map: np.ndarray,
    max_gap_px: int = 30,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Connect nearby dead-end road segments using the probability map.

    For each dead-end pixel in the skeleton, search for nearby dead-ends
    within max_gap_px distance. If the average probability along the
    straight-line path exceeds min_confidence, draw a bridge.

    Args:
        mask: Binary mask (H, W), uint8 {0, 255}.
        prob_map: Probability map (H, W), float32 [0, 1].
        max_gap_px: Maximum gap distance to bridge.
        min_confidence: Minimum average probability along the bridge path.

    Returns: Mask with bridged gaps.
    """
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)

    # Find endpoints: skeleton pixels with exactly 1 neighbor
    endpoints = _find_endpoints(skeleton)

    if len(endpoints) < 2:
        return mask

    result = mask.copy()

    # For each endpoint, find nearby endpoints and check bridge viability
    connected = set()
    for i, (y1, x1) in enumerate(endpoints):
        for j, (y2, x2) in enumerate(endpoints):
            if i >= j:
                continue
            if (i, j) in connected:
                continue

            dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if dist > max_gap_px or dist < 3:
                continue

            # Sample probability along the line
            avg_prob = _line_probability(prob_map, y1, x1, y2, x2)
            if avg_prob >= min_confidence:
                # Draw bridge
                cv2.line(result, (x1, y1), (x2, y2), 255, thickness=1)
                connected.add((i, j))

    return result


def _find_endpoints(skeleton: np.ndarray) -> list:
    """Find pixels in skeleton with exactly 1 neighbor (dead ends)."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    endpoints = np.argwhere((skeleton > 0) & (neighbor_count == 1))
    return endpoints.tolist()  # list of [y, x]


def _line_probability(
    prob_map: np.ndarray, y1: int, x1: int, y2: int, x2: int
) -> float:
    """Average probability along a straight line between two points."""
    n_samples = max(abs(y2 - y1), abs(x2 - x1), 1)
    ys = np.linspace(y1, y2, n_samples + 1).astype(int)
    xs = np.linspace(x1, x2, n_samples + 1).astype(int)

    # Clamp to image bounds
    h, w = prob_map.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    return float(prob_map[ys, xs].mean())


# =========================================================================
# Step 8: GeoJSON Export
# =========================================================================


def graph_to_geojson(graph: dict) -> dict:
    """Convert a road graph to GeoJSON FeatureCollection.

    Args:
        graph: Dict with "nodes" and "edges" (in pixel coordinates).

    Returns: GeoJSON FeatureCollection with LineString features.
    """
    features = []
    for edge in graph["edges"]:
        if len(edge) < 2:
            continue
        length = sum(
            np.sqrt((edge[i][0] - edge[i + 1][0]) ** 2 + (edge[i][1] - edge[i + 1][1]) ** 2)
            for i in range(len(edge) - 1)
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": edge,
            },
            "properties": {
                "length_px": round(float(length), 1),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "num_segments": len(features),
            "coordinate_system": "pixel",
        },
    }
