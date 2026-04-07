"""Composable post-processing pipeline with toggleable steps.

Each step can be enabled/disabled independently for ablation studies.
The pipeline tracks which steps are active and returns intermediate
results for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from road_segmentation.postprocessing.steps import (
    apply_threshold,
    bridge_gaps,
    graph_to_geojson,
    morphological_close,
    morphological_open,
    prune_short_branches,
    remove_small_components,
    simplify_edges,
    skeleton_to_graph,
    skeletonize_mask,
)


@dataclass
class PipelineConfig:
    """Configuration for toggling and parameterizing each step."""

    # Step 1: Threshold
    threshold: float = 0.5

    # Step 2: Component filtering
    remove_small_components: bool = True
    min_component_area: int = 100

    # Step 3: Morphology
    morphological_closing: bool = True
    closing_kernel_size: int = 5
    morphological_opening: bool = False
    opening_kernel_size: int = 3

    # Step 4: Gap bridging
    gap_bridging: bool = False
    max_gap_px: int = 30
    min_bridge_confidence: float = 0.3

    # Step 5: Skeletonize + graph
    skeletonize: bool = False
    prune_branches: bool = True
    min_branch_length_px: int = 20
    simplify_tolerance: float = 2.0

    # Step 6: GeoJSON output
    geojson: bool = False


@dataclass
class PipelineResult:
    """Result from running the pipeline, with intermediates for analysis."""

    # Final outputs
    mask: np.ndarray  # binary mask after all mask-level steps
    skeleton: Optional[np.ndarray] = None
    graph: Optional[dict] = None
    geojson: Optional[dict] = None

    # Intermediate masks (for ablation — keyed by step name)
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metadata
    steps_applied: list = field(default_factory=list)


def run_pipeline(
    prob_map: np.ndarray,
    config: PipelineConfig,
    collect_intermediates: bool = False,
) -> PipelineResult:
    """Run the post-processing pipeline on a probability map.

    Args:
        prob_map: Model output probability map (H, W), float32 [0, 1].
        config: Pipeline configuration with step toggles.
        collect_intermediates: If True, store mask after each step for analysis.

    Returns:
        PipelineResult with final outputs and optional intermediates.
    """
    result = PipelineResult(mask=np.zeros_like(prob_map, dtype=np.uint8))
    intermediates = {}

    # Step 1: Threshold (always applied)
    mask = apply_threshold(prob_map, config.threshold)
    result.steps_applied.append(f"threshold={config.threshold}")
    if collect_intermediates:
        intermediates["01_threshold"] = mask.copy()

    # Step 2: Remove small components
    if config.remove_small_components:
        mask = remove_small_components(mask, min_area=config.min_component_area)
        result.steps_applied.append(f"remove_small_components(min_area={config.min_component_area})")
        if collect_intermediates:
            intermediates["02_component_filter"] = mask.copy()

    # Step 3a: Morphological opening (noise removal)
    if config.morphological_opening:
        mask = morphological_open(mask, kernel_size=config.opening_kernel_size)
        result.steps_applied.append(f"morph_open(k={config.opening_kernel_size})")
        if collect_intermediates:
            intermediates["03a_morph_open"] = mask.copy()

    # Step 3b: Morphological closing (gap filling)
    if config.morphological_closing:
        mask = morphological_close(mask, kernel_size=config.closing_kernel_size)
        result.steps_applied.append(f"morph_close(k={config.closing_kernel_size})")
        if collect_intermediates:
            intermediates["03b_morph_close"] = mask.copy()

    # Step 4: Confidence-weighted gap bridging
    if config.gap_bridging:
        mask = bridge_gaps(
            mask, prob_map,
            max_gap_px=config.max_gap_px,
            min_confidence=config.min_bridge_confidence,
        )
        result.steps_applied.append(
            f"gap_bridging(max_gap={config.max_gap_px}, min_conf={config.min_bridge_confidence})"
        )
        if collect_intermediates:
            intermediates["04_gap_bridging"] = mask.copy()

    result.mask = mask

    # Step 5: Skeletonize + graph extraction
    if config.skeletonize:
        skeleton = skeletonize_mask(mask)
        result.skeleton = skeleton
        result.steps_applied.append("skeletonize")
        if collect_intermediates:
            intermediates["05_skeleton"] = skeleton.copy()

        graph = skeleton_to_graph(skeleton)

        if config.prune_branches:
            graph = prune_short_branches(graph, min_length_px=config.min_branch_length_px)
            result.steps_applied.append(f"prune_branches(min_len={config.min_branch_length_px})")

        graph = simplify_edges(graph, tolerance=config.simplify_tolerance)
        result.steps_applied.append(f"simplify(tol={config.simplify_tolerance})")
        result.graph = graph

        # Step 6: GeoJSON
        if config.geojson:
            result.geojson = graph_to_geojson(graph)
            result.steps_applied.append("geojson_export")

    result.intermediates = intermediates
    return result


# =========================================================================
# Ablation helper: run pipeline with incremental steps
# =========================================================================


def ablation_configs() -> Dict[str, PipelineConfig]:
    """Return a dict of named configs, each adding one step incrementally.

    Used for ablation studies to measure the contribution of each step.
    """
    return {
        "00_raw_threshold_0.5": PipelineConfig(
            threshold=0.5,
            remove_small_components=False,
            morphological_closing=False,
            morphological_opening=False,
            gap_bridging=False,
            skeletonize=False,
        ),
        "01_optimal_threshold": PipelineConfig(
            threshold=0.0,  # placeholder — set after threshold sweep
            remove_small_components=False,
            morphological_closing=False,
            morphological_opening=False,
            gap_bridging=False,
            skeletonize=False,
        ),
        "02_+component_filter": PipelineConfig(
            threshold=0.0,
            remove_small_components=True,
            min_component_area=100,
            morphological_closing=False,
            morphological_opening=False,
            gap_bridging=False,
            skeletonize=False,
        ),
        "03_+morph_close": PipelineConfig(
            threshold=0.0,
            remove_small_components=True,
            min_component_area=100,
            morphological_closing=True,
            closing_kernel_size=5,
            morphological_opening=False,
            gap_bridging=False,
            skeletonize=False,
        ),
        "04_+morph_open": PipelineConfig(
            threshold=0.0,
            remove_small_components=True,
            min_component_area=100,
            morphological_closing=True,
            closing_kernel_size=5,
            morphological_opening=True,
            opening_kernel_size=3,
            gap_bridging=False,
            skeletonize=False,
        ),
        "05_+gap_bridging": PipelineConfig(
            threshold=0.0,
            remove_small_components=True,
            min_component_area=100,
            morphological_closing=True,
            closing_kernel_size=5,
            morphological_opening=True,
            opening_kernel_size=3,
            gap_bridging=True,
            max_gap_px=30,
            min_bridge_confidence=0.3,
            skeletonize=False,
        ),
    }
