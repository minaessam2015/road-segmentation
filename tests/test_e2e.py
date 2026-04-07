"""End-to-end integration tests with real model weights.

These tests verify the full pipeline: model loading → preprocessing →
inference → postprocessing → API response. They use actual model weights
(not mocks) to catch real-world issues like:
- Checkpoint loading bugs (wrong keys, architecture mismatch)
- Preprocessing inconsistencies between training and inference
- ONNX export errors (model exports but produces wrong results)
- API response format correctness with real predictions

Usage:
    # Run with PyTorch checkpoint:
    pytest tests/test_e2e.py --checkpoint checkpoints/effecientnet_b4_1024_boundary_loss.pth

    # Run with ONNX model:
    pytest tests/test_e2e.py --onnx models/UnetPlusPlus_efficientnet-b4_int8.onnx

    # Skip if no model available (CI default):
    pytest tests/test_e2e.py  # all tests skip gracefully
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pytest_addoption(parser):
    """Add CLI options for model paths."""
    try:
        parser.addoption("--checkpoint", default=None, help="Path to PyTorch checkpoint")
        parser.addoption("--onnx", default=None, help="Path to ONNX model")
    except ValueError:
        pass  # already added by another conftest


# ---------------------------------------------------------------------------
# Auto-discover models if not passed via CLI
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "effecientnet_b4_1024_boundary_loss.pth"
_DEFAULT_ONNX = _PROJECT_ROOT / "models" / "UnetPlusPlus_efficientnet-b4_int8.onnx"
_SAMPLE_IMAGE_DIR = _PROJECT_ROOT / "data" / "raw" / "deepglobe-road-extraction-dataset" / "train"


def _get_checkpoint(request):
    cli = request.config.getoption("--checkpoint", default=None)
    if cli and Path(cli).exists():
        return Path(cli)
    if _DEFAULT_CHECKPOINT.exists():
        return _DEFAULT_CHECKPOINT
    return None


def _get_onnx(request):
    cli = request.config.getoption("--onnx", default=None)
    if cli and Path(cli).exists():
        return Path(cli)
    if _DEFAULT_ONNX.exists():
        return _DEFAULT_ONNX
    return None


def _get_sample_image_bytes():
    """Get a real satellite image as bytes, or generate a synthetic one."""
    if _SAMPLE_IMAGE_DIR.exists():
        sample = next(_SAMPLE_IMAGE_DIR.glob("*_sat.jpg"), None)
        if sample:
            return sample.read_bytes(), sample.name
    # Fallback: synthetic image
    img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "synthetic.png"


def _get_sample_gt_mask():
    """Get a real ground truth mask for IoU verification."""
    if _SAMPLE_IMAGE_DIR.exists():
        sample = next(_SAMPLE_IMAGE_DIR.glob("*_mask.png"), None)
        if sample:
            mask = np.array(Image.open(sample).convert("L"))
            return (mask > 0).astype(np.uint8)
    return None


# ---------------------------------------------------------------------------
# PyTorch checkpoint tests
# ---------------------------------------------------------------------------


class TestPyTorchInference:
    """End-to-end tests using the PyTorch InferenceEngine."""

    def test_checkpoint_loads(self, request):
        ckpt = _get_checkpoint(request)
        if ckpt is None:
            pytest.skip("No checkpoint available")

        from road_segmentation.api.inference import InferenceEngine
        engine = InferenceEngine.from_checkpoint(ckpt)
        assert engine is not None
        assert engine.model is not None

    def test_predict_returns_valid_result(self, request):
        ckpt = _get_checkpoint(request)
        if ckpt is None:
            pytest.skip("No checkpoint available")

        from road_segmentation.api.inference import InferenceEngine
        engine = InferenceEngine.from_checkpoint(ckpt)

        image_bytes, _ = _get_sample_image_bytes()
        result = engine.predict(image_bytes)

        # Check all fields
        assert result.mask.shape == result.original_size[::-1]  # (H, W)
        assert result.mask.dtype == np.uint8
        assert set(np.unique(result.mask)).issubset({0, 255})
        assert 0 <= result.road_coverage_pct <= 100
        assert 0 <= result.confidence_mean <= 1
        assert result.inference_time_ms > 0

    def test_predict_produces_reasonable_roads(self, request):
        ckpt = _get_checkpoint(request)
        if ckpt is None:
            pytest.skip("No checkpoint available")

        from road_segmentation.api.inference import InferenceEngine
        engine = InferenceEngine.from_checkpoint(ckpt)

        image_bytes, _ = _get_sample_image_bytes()
        result = engine.predict(image_bytes)

        # Road coverage should be in a reasonable range for satellite imagery
        # (not 0% and not 100% — the model should find some roads)
        if _SAMPLE_IMAGE_DIR.exists():
            assert result.road_coverage_pct > 0.1, "Model predicts no roads at all"
            assert result.road_coverage_pct < 50, "Model predicts too many roads"

    def test_predict_matches_ground_truth(self, request):
        """Verify IoU is above a minimum threshold on a real sample."""
        ckpt = _get_checkpoint(request)
        if ckpt is None:
            pytest.skip("No checkpoint available")

        gt = _get_sample_gt_mask()
        if gt is None:
            pytest.skip("No ground truth available")

        from road_segmentation.api.inference import InferenceEngine
        engine = InferenceEngine.from_checkpoint(ckpt)

        # Find the matching satellite image
        mask_files = list(_SAMPLE_IMAGE_DIR.glob("*_mask.png"))
        sat_path = str(mask_files[0]).replace("_mask.png", "_sat.jpg")
        image_bytes = Path(sat_path).read_bytes()

        result = engine.predict(image_bytes)

        # Compute IoU
        pred = (result.mask > 0).astype(bool)
        gt_bool = gt.astype(bool)
        # Resize if needed
        if pred.shape != gt_bool.shape:
            import cv2
            pred = cv2.resize(pred.astype(np.uint8), (gt_bool.shape[1], gt_bool.shape[0])) > 0

        tp = (pred & gt_bool).sum()
        union = (pred | gt_bool).sum()
        iou = tp / max(union, 1)

        assert iou > 0.3, f"IoU too low: {iou:.4f}. Model may be broken."


# ---------------------------------------------------------------------------
# ONNX model tests
# ---------------------------------------------------------------------------


class TestONNXInference:
    """End-to-end tests using the ONNX InferenceEngine."""

    def test_onnx_loads(self, request):
        onnx_path = _get_onnx(request)
        if onnx_path is None:
            pytest.skip("No ONNX model available")

        from road_segmentation.api.optimize import ONNXInferenceEngine
        engine = ONNXInferenceEngine(onnx_path, device="cpu", image_size=1024)
        assert engine is not None

    def test_onnx_predict_returns_valid_result(self, request):
        onnx_path = _get_onnx(request)
        if onnx_path is None:
            pytest.skip("No ONNX model available")

        from road_segmentation.api.optimize import ONNXInferenceEngine
        engine = ONNXInferenceEngine(onnx_path, device="cpu", image_size=1024)

        image_bytes, _ = _get_sample_image_bytes()
        result = engine.predict(image_bytes)

        assert result["mask"].shape == (result["original_size"][1], result["original_size"][0])
        assert result["mask"].dtype == np.uint8
        assert 0 <= result["road_coverage_pct"] <= 100
        assert 0 <= result["confidence_mean"] <= 1
        assert result["inference_time_ms"] > 0

    def test_onnx_matches_pytorch(self, request):
        """ONNX output should match PyTorch output within tolerance."""
        ckpt = _get_checkpoint(request)
        onnx_path = _get_onnx(request)
        if ckpt is None or onnx_path is None:
            pytest.skip("Need both checkpoint and ONNX model")

        from road_segmentation.api.inference import InferenceEngine
        from road_segmentation.api.optimize import ONNXInferenceEngine

        pytorch_engine = InferenceEngine.from_checkpoint(ckpt)
        onnx_engine = ONNXInferenceEngine(onnx_path, device="cpu", image_size=1024)

        image_bytes, _ = _get_sample_image_bytes()

        pytorch_result = pytorch_engine.predict(image_bytes)
        onnx_result = onnx_engine.predict(image_bytes)

        # Coverage should be similar (within 2%)
        coverage_diff = abs(pytorch_result.road_coverage_pct - onnx_result["road_coverage_pct"])
        assert coverage_diff < 2.0, f"Coverage mismatch: PyTorch={pytorch_result.road_coverage_pct}, ONNX={onnx_result['road_coverage_pct']}"

        # Masks should be mostly the same (>95% pixel agreement)
        pt_mask = (pytorch_result.mask > 0).astype(bool)
        onnx_mask = (onnx_result["mask"] > 0).astype(bool)
        if pt_mask.shape != onnx_mask.shape:
            import cv2
            onnx_mask = cv2.resize(onnx_result["mask"], (pt_mask.shape[1], pt_mask.shape[0])) > 0
        agreement = (pt_mask == onnx_mask).mean()
        assert agreement > 0.95, f"Pixel agreement too low: {agreement:.4f}"


# ---------------------------------------------------------------------------
# API end-to-end test
# ---------------------------------------------------------------------------


class TestAPIEndToEnd:
    """Full API test: start server with real model, send image, verify response."""

    def test_api_with_real_model(self, request, tmp_path):
        onnx_path = _get_onnx(request)
        if onnx_path is None:
            pytest.skip("No ONNX model available")

        from fastapi.testclient import TestClient
        from road_segmentation.api.app import create_app

        app = create_app(onnx_path=str(onnx_path), device="cpu", results_dir=str(tmp_path / "results"))
        client = TestClient(app)

        # Health check
        health = client.get("/health").json()
        assert health["model_loaded"] is True

        # Model info
        info = client.get("/api/v1/model-info").json()
        assert info["backend"] == "onnx"

        # Segment
        image_bytes, filename = _get_sample_image_bytes()
        response = client.post(
            "/api/v1/segment",
            files={"image": (filename, image_bytes, "image/png")},
        )
        assert response.status_code == 200

        data = response.json()
        assert "mask_url" in data
        assert "road_coverage_pct" in data
        assert "confidence_mean" in data
        assert "metadata" in data
        assert data["metadata"]["model_version"] is not None
        assert data["metadata"]["inference_time_ms"] > 0

        # Verify mask is downloadable
        mask_response = client.get(data["mask_url"])
        assert mask_response.status_code == 200

        # Metrics endpoint
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "road_seg_requests_total" in metrics.text
