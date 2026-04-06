"""Tests for the FastAPI inference API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    """Mock inference engine that returns deterministic results."""
    engine = MagicMock()
    engine.threshold = 0.5
    engine.model_version = "test_v1"
    engine.image_size = 256

    # Mock predict to return a dict (like ONNXInferenceEngine)
    engine.predict.return_value = {
        "mask": np.zeros((256, 256), dtype=np.uint8),
        "probability_map": np.random.uniform(0, 1, (256, 256)).astype(np.float32),
        "road_coverage_pct": 3.5,
        "confidence_mean": 0.82,
        "original_size": (256, 256),
        "inference_time_ms": 42.0,
    }
    return engine


@pytest.fixture
def client(mock_engine, tmp_path):
    """FastAPI test client with mocked model."""
    from road_segmentation.api.app import create_app

    # Patch the engine loading
    import road_segmentation.api.app as app_module
    app_module._engine = mock_engine
    app_module._model_info = {
        "backend": "mock",
        "model_version": "test_v1",
        "device": "cpu",
    }
    app_module._results_dir = tmp_path / "results"
    app_module._results_dir.mkdir()

    app = create_app.__wrapped__() if hasattr(create_app, "__wrapped__") else None

    # Build app manually for testing
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    app.mount("/results", StaticFiles(directory=str(app_module._results_dir)), name="results")
    app.add_api_route("/health", app_module.health, methods=["GET"])
    app.add_api_route("/api/v1/model-info", app_module.model_info, methods=["GET"])
    app.add_api_route("/api/v1/segment", app_module.segment, methods=["POST"])

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data


class TestSegmentEndpoint:
    def test_segment_with_valid_image(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/segment",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "mask_url" in data
        assert "road_coverage_pct" in data
        assert "confidence_mean" in data
        assert "metadata" in data
        assert "inference_time_ms" in data["metadata"]
        assert "model_version" in data["metadata"]
        assert "image_size" in data["metadata"]

    def test_segment_with_threshold(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/segment?threshold=0.3",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["threshold"] == 0.3

    def test_segment_with_geojson(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/segment?return_geojson=true",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "geojson" in data

    def test_segment_rejects_empty_file(self, client):
        response = client.post(
            "/api/v1/segment",
            files={"image": ("test.png", b"", "image/png")},
        )
        assert response.status_code == 400

    def test_segment_rejects_unsupported_format(self, client):
        response = client.post(
            "/api/v1/segment",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_segment_rejects_oversized_file(self, client):
        # 51 MB of zeros
        huge = b"\x00" * (51 * 1024 * 1024)
        response = client.post(
            "/api/v1/segment",
            files={"image": ("huge.png", huge, "image/png")},
        )
        assert response.status_code == 413

    def test_mask_url_is_accessible(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/segment",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        data = response.json()
        mask_url = data["mask_url"]
        mask_response = client.get(mask_url)
        assert mask_response.status_code == 200
