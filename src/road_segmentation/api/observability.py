"""Observability: structured logging, request tracing, and metrics.

Provides production-grade observability for the inference API:
    - Structured JSON logging with request context
    - Unique request ID tracing across all stages
    - Inference metrics (latency, throughput, error counts)
    - Prometheus-compatible metrics endpoint
    - Input/output audit logging for drift detection
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ---------------------------------------------------------------------------
# Structured JSON Logger
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_structured_logging(level: str = "INFO") -> None:
    """Configure the root logger with JSON-formatted output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))


# ---------------------------------------------------------------------------
# Request ID Middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request for tracing.

    The ID is:
        - Generated as a UUID4 if not provided
        - Read from the ``X-Request-ID`` header if present (for distributed tracing)
        - Added to the response headers
        - Stored in request.state for downstream access
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Inference Metrics Collector
# ---------------------------------------------------------------------------


@dataclass
class InferenceMetrics:
    """Collects inference metrics for monitoring and Prometheus export.

    Thread-safe for single-worker uvicorn (the default).
    For multi-worker, use an external metrics store (Redis, StatsD).
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_inference_time_ms: float = 0.0
    latency_histogram: list = field(default_factory=list)
    error_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Keep last N for rolling stats
    _max_history: int = 1000

    def record_success(self, inference_time_ms: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.total_inference_time_ms += inference_time_ms
        self.latency_histogram.append(inference_time_ms)
        if len(self.latency_histogram) > self._max_history:
            self.latency_histogram = self.latency_histogram[-self._max_history:]

    def record_error(self, status_code: int) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self.error_counts[status_code] += 1

    def get_summary(self) -> Dict[str, Any]:
        import numpy as np
        latencies = self.latency_histogram if self.latency_histogram else [0]
        arr = np.array(latencies)
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.failed_requests / max(self.total_requests, 1),
            "latency_ms": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            },
            "throughput_rps": (
                self.successful_requests / (self.total_inference_time_ms / 1000)
                if self.total_inference_time_ms > 0 else 0
            ),
            "error_counts": dict(self.error_counts),
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        import numpy as np
        lines = []

        lines.append("# HELP road_seg_requests_total Total inference requests")
        lines.append("# TYPE road_seg_requests_total counter")
        lines.append(f"road_seg_requests_total {self.total_requests}")

        lines.append("# HELP road_seg_requests_success Successful requests")
        lines.append("# TYPE road_seg_requests_success counter")
        lines.append(f"road_seg_requests_success {self.successful_requests}")

        lines.append("# HELP road_seg_requests_failed Failed requests")
        lines.append("# TYPE road_seg_requests_failed counter")
        lines.append(f"road_seg_requests_failed {self.failed_requests}")

        if self.latency_histogram:
            arr = np.array(self.latency_histogram)
            lines.append("# HELP road_seg_inference_latency_ms Inference latency in milliseconds")
            lines.append("# TYPE road_seg_inference_latency_ms summary")
            lines.append(f'road_seg_inference_latency_ms{{quantile="0.5"}} {np.median(arr):.1f}')
            lines.append(f'road_seg_inference_latency_ms{{quantile="0.95"}} {np.percentile(arr, 95):.1f}')
            lines.append(f'road_seg_inference_latency_ms{{quantile="0.99"}} {np.percentile(arr, 99):.1f}')
            lines.append(f"road_seg_inference_latency_ms_sum {arr.sum():.1f}")
            lines.append(f"road_seg_inference_latency_ms_count {len(arr)}")

        for code, count in self.error_counts.items():
            lines.append(f'road_seg_errors{{status_code="{code}"}} {count}')

        return "\n".join(lines) + "\n"


# Global metrics instance
metrics = InferenceMetrics()


# ---------------------------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing, status, and request ID."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        logger = logging.getLogger("road_segmentation.api")

        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        # Log the request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 1),
        }

        if response.status_code >= 400:
            logger.warning("Request failed", extra={"extra_data": log_data, "request_id": request_id})
        else:
            logger.info("Request completed", extra={"extra_data": log_data, "request_id": request_id})

        return response


# ---------------------------------------------------------------------------
# Inference Audit Logger
# ---------------------------------------------------------------------------


def log_inference(
    request_id: str,
    image_hash: str,
    image_size: tuple,
    road_coverage_pct: float,
    confidence_mean: float,
    inference_time_ms: float,
    threshold: float,
    model_version: str,
) -> None:
    """Log inference details for audit trail and drift detection.

    This creates a structured log entry that can be used to:
    - Trace specific predictions back to inputs
    - Monitor prediction distribution shifts over time
    - Detect data drift (changing coverage/confidence patterns)
    """
    logger = logging.getLogger("road_segmentation.api.audit")
    logger.info(
        "Inference completed",
        extra={
            "request_id": request_id,
            "extra_data": {
                "event": "inference",
                "image_hash": image_hash,
                "image_width": image_size[0],
                "image_height": image_size[1],
                "road_coverage_pct": road_coverage_pct,
                "confidence_mean": confidence_mean,
                "inference_time_ms": inference_time_ms,
                "threshold": threshold,
                "model_version": model_version,
            },
        },
    )

    # Update metrics
    metrics.record_success(inference_time_ms)


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute a short content hash for an image (for audit logging)."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]
