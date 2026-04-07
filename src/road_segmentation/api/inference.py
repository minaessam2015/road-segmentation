"""Inference engine: model loading, prediction, and post-processing.

This module is designed to be loaded once at API startup and reused
across requests. The model is held in memory as a singleton.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from road_segmentation.models.factory import create_model

logger = logging.getLogger(__name__)

# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SegmentationResult:
    """Container for a single inference result."""

    mask: np.ndarray  # binary mask (H, W), uint8 {0, 255}
    probability_map: np.ndarray  # raw probabilities (H, W), float32 [0, 1]
    road_coverage_pct: float
    confidence_mean: float
    original_size: Tuple[int, int]  # (width, height)
    inference_time_ms: float


class InferenceEngine:
    """Loads a trained model and runs inference on satellite images.

    Usage:
        engine = InferenceEngine.from_checkpoint("checkpoints/best.pth")
        result = engine.predict(image_bytes)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        image_size: int = 512,
        threshold: float = 0.5,
        model_version: str = "v1.0",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = image_size
        self.threshold = threshold
        self.model_version = model_version

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> "InferenceEngine":
        """Load model from a training checkpoint.

        Reads the config stored inside the checkpoint to reconstruct
        the exact model architecture that was trained.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct model from saved config
        config = state.get("config", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        model = create_model(
            arch=model_cfg.get("arch", "Unet"),
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=None,  # weights come from checkpoint
            in_channels=model_cfg.get("in_channels", 3),
            classes=model_cfg.get("classes", 1),
        )
        model.load_state_dict(state["model_state_dict"])

        image_size = data_cfg.get("image_size", 512)
        model_version = (
            f"{model_cfg.get('arch', 'Unet')}_{model_cfg.get('encoder_name', 'resnet34')}"
        )

        logger.info(
            f"Model loaded: {model_version} | "
            f"image_size={image_size} | device={device}"
        )

        return cls(
            model=model,
            device=device,
            image_size=image_size,
            threshold=threshold,
            model_version=model_version,
        )

    def predict(self, image_bytes: bytes) -> SegmentationResult:
        """Run inference on raw image bytes.

        Args:
            image_bytes: Raw bytes of a satellite image (PNG, JPG, etc.).

        Returns:
            SegmentationResult with mask, probabilities, and metadata.
        """
        t0 = time.time()

        # Decode image
        image = self._decode_image(image_bytes)
        original_h, original_w = image.shape[:2]

        # Preprocess
        tensor = self._preprocess(image)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)

        # Post-process
        prob_map = probs[0, 0].cpu().numpy()  # (H_resized, W_resized)

        # Resize back to original dimensions
        prob_map_full = cv2.resize(
            prob_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )
        mask = (prob_map_full >= self.threshold).astype(np.uint8) * 255

        # Metrics
        road_pixels = (mask > 0).sum()
        total_pixels = mask.size
        road_coverage_pct = float(road_pixels / total_pixels * 100)
        confidence_mean = float(prob_map_full[prob_map_full >= self.threshold].mean()) if road_pixels > 0 else 0.0

        inference_time_ms = (time.time() - t0) * 1000

        return SegmentationResult(
            mask=mask,
            probability_map=prob_map_full,
            road_coverage_pct=round(road_coverage_pct, 2),
            confidence_mean=round(confidence_mean, 4),
            original_size=(original_w, original_h),
            inference_time_ms=round(inference_time_ms, 1),
        )

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to RGB numpy array."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Ensure it is a valid PNG/JPG file.")
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and convert to tensor."""
        resized = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        # Normalize with ImageNet stats
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)


def mask_to_geojson(
    mask: np.ndarray,
    simplify_tolerance: float = 2.0,
    min_branch_length: int = 20,
    min_component_area: int = 100,
) -> dict:
    """Convert a binary road mask to GeoJSON polylines via centerline extraction.

    Pipeline:
        1. Remove small noise components
        2. Morphological closing to fill small gaps
        3. Skeletonize to 1-pixel-wide road centerlines
        4. Extract graph (nodes + edges) from skeleton
        5. Prune short dangling branches
        6. Simplify edge geometry (Douglas-Peucker)
        7. Export as GeoJSON LineString features

    Args:
        mask: Binary mask (H, W) with road pixels = 255.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        min_branch_length: Minimum edge length to keep (pixels).
        min_component_area: Minimum connected component area (pixels).

    Returns:
        GeoJSON FeatureCollection with LineString features in pixel coordinates.
    """
    from road_segmentation.postprocessing.steps import (
        graph_to_geojson,
        morphological_close,
        prune_short_branches,
        remove_small_components,
        simplify_edges,
        skeleton_to_graph,
        skeletonize_mask,
    )

    # Step 1: Clean the mask
    cleaned = remove_small_components(mask, min_area=min_component_area)

    # Step 2: Fill small gaps
    closed = morphological_close(cleaned, kernel_size=5)

    # Step 3: Skeletonize to centerlines
    skeleton = skeletonize_mask(closed)

    # Step 4: Extract graph
    graph = skeleton_to_graph(skeleton)

    # Step 5: Prune short branches
    graph = prune_short_branches(graph, min_length_px=min_branch_length)

    # Step 6: Simplify geometry
    graph = simplify_edges(graph, tolerance=simplify_tolerance)

    # Step 7: Export as GeoJSON
    return graph_to_geojson(graph)
