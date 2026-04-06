"""FastAPI inference API for road segmentation.

Endpoints:
    POST /api/v1/segment  — accepts a satellite image, returns road mask + metadata
    GET  /health           — health check
    GET  /api/v1/model-info — model version and configuration
    GET  /metrics           — Prometheus-compatible metrics
    GET  /results/{filename} — serve saved result masks
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------

_engine = None
_model_info = {}
_results_dir = Path("results")

# Supported image formats
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_MAX_IMAGE_SIZE_MB = 50


def create_app(
    checkpoint_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    device: str = "cpu",
    results_dir: str = "results",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Loads the model at startup (once). Supports two backends:
        - PyTorch checkpoint (.pth) via InferenceEngine
        - ONNX model (.onnx) via ONNXInferenceEngine (preferred for production)
    """
    global _engine, _model_info, _results_dir

    app = FastAPI(
        title="Road Segmentation API",
        description="Extract road masks from satellite imagery",
        version="1.0.0",
    )

    _results_dir = Path(results_dir)
    _results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    if onnx_path:
        from road_segmentation.api.optimize import ONNXInferenceEngine

        _engine = ONNXInferenceEngine(onnx_path, device=device)
        _model_info = {
            "backend": "onnx",
            "model_path": onnx_path,
            "device": device,
            "model_version": _engine.model_version,
            "image_size": _engine.image_size,
        }
    elif checkpoint_path:
        from road_segmentation.api.inference import InferenceEngine

        _engine = InferenceEngine.from_checkpoint(checkpoint_path)
        _model_info = {
            "backend": "pytorch",
            "model_path": checkpoint_path,
            "device": str(_engine.device),
            "model_version": _engine.model_version,
            "image_size": _engine.image_size,
        }
    else:
        logger.warning("No model loaded. Endpoints will return 503.")
        _model_info = {"backend": "none", "status": "no model loaded"}

    # --- Mount static files for result images ---
    app.mount("/results", StaticFiles(directory=str(_results_dir)), name="results")

    # --- Observability middleware ---
    from road_segmentation.api.observability import (
        RequestIDMiddleware,
        RequestLoggingMiddleware,
        setup_structured_logging,
    )
    setup_structured_logging()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # --- Register routes ---
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/api/v1/model-info", model_info, methods=["GET"])
    app.add_api_route("/api/v1/segment", segment, methods=["POST"])
    app.add_api_route("/metrics", prometheus_metrics, methods=["GET"])

    logger.info(f"API ready. Model: {_model_info}")
    return app


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def health():
    """Health check with system details."""
    import shutil
    disk = shutil.disk_usage(_results_dir)
    info = {
        "status": "healthy" if _engine is not None else "degraded",
        "model_loaded": _engine is not None,
        "disk_free_gb": round(disk.free / 1e9, 1),
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_free_gb"] = round(
                (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9, 1
            )
    except Exception:
        pass
    return info


async def model_info():
    """Return model version and configuration."""
    return _model_info


async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from road_segmentation.api.observability import metrics
    return PlainTextResponse(metrics.to_prometheus(), media_type="text/plain")


async def segment(
    image: UploadFile = File(..., description="Satellite image (PNG, JPG, TIFF)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Binarization threshold"),
    return_geojson: bool = Query(False, description="Also return GeoJSON polylines"),
):
    """Segment roads from a satellite image.

    Accepts a multipart file upload and returns:
    - ``mask_url``: URL to the binary road mask PNG
    - ``road_coverage_pct``: percentage of road pixels
    - ``confidence_mean``: average confidence on predicted road pixels
    - ``metadata``: image size, model version, inference time
    - ``geojson`` (optional): road polylines in pixel coordinates
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # --- Validate input ---
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(image.filename).suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    # Read image bytes
    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(image_bytes) > _MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(image_bytes) / 1e6:.1f} MB). Max: {_MAX_IMAGE_SIZE_MB} MB",
        )

    # --- Run inference ---
    try:
        if hasattr(_engine, "predict") and callable(_engine.predict):
            result = _engine.predict(image_bytes)

            # Handle both InferenceEngine and ONNXInferenceEngine return types
            if hasattr(result, "mask"):
                # InferenceEngine returns a dataclass
                mask = result.mask
                road_coverage_pct = result.road_coverage_pct
                confidence_mean = result.confidence_mean
                original_size = result.original_size
                inference_time_ms = result.inference_time_ms
                prob_map = result.probability_map
            else:
                # ONNXInferenceEngine returns a dict
                mask = result["mask"]
                road_coverage_pct = result["road_coverage_pct"]
                confidence_mean = result["confidence_mean"]
                original_size = result["original_size"]
                inference_time_ms = result["inference_time_ms"]
                prob_map = result.get("probability_map")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")

    # --- Apply custom threshold if different from engine default ---
    if hasattr(_engine, "threshold") and threshold != _engine.threshold and prob_map is not None:
        mask = ((prob_map >= threshold) * 255).astype(np.uint8)
        road_pixels = (mask > 0).sum()
        total_pixels = mask.size
        road_coverage_pct = round(float(road_pixels / total_pixels * 100), 2)
        if road_pixels > 0:
            confidence_mean = round(float(prob_map[prob_map >= threshold].mean()), 4)
        else:
            confidence_mean = 0.0

    # --- Save mask to results directory ---
    result_id = hashlib.md5(image_bytes[:1024]).hexdigest()[:12]
    mask_filename = f"{result_id}_mask.png"
    mask_path = _results_dir / mask_filename
    cv2.imwrite(str(mask_path), mask)

    # --- Audit logging ---
    from road_segmentation.api.observability import compute_image_hash, log_inference
    log_inference(
        request_id=result_id,
        image_hash=compute_image_hash(image_bytes),
        image_size=original_size,
        road_coverage_pct=road_coverage_pct,
        confidence_mean=confidence_mean,
        inference_time_ms=inference_time_ms,
        threshold=threshold,
        model_version=_model_info.get("model_version", "v1.0"),
    )

    # --- Build response ---
    response = {
        "mask_url": f"/results/{mask_filename}",
        "road_coverage_pct": road_coverage_pct,
        "confidence_mean": confidence_mean,
        "metadata": {
            "image_size": list(original_size),
            "model_version": _model_info.get("model_version", "v1.0"),
            "inference_time_ms": inference_time_ms,
            "threshold": threshold,
            "request_id": result_id,
        },
    }

    # --- Optional GeoJSON ---
    if return_geojson:
        try:
            from road_segmentation.api.inference import mask_to_geojson
            response["geojson"] = mask_to_geojson(mask)
        except Exception as e:
            logger.warning(f"GeoJSON generation failed: {e}")
            response["geojson"] = None

    return JSONResponse(content=response)
