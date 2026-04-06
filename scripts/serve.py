"""Start the road segmentation inference API.

Usage:
    # From PyTorch checkpoint:
    python scripts/serve.py --checkpoint checkpoints/best.pth

    # From optimized ONNX model:
    python scripts/serve.py --onnx models/model_int8.onnx --device cpu

    # GPU serving:
    python scripts/serve.py --onnx models/model_fp16.onnx --device gpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Road Segmentation API Server")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to PyTorch checkpoint (.pth)")
    group.add_argument("--onnx", type=str, help="Path to ONNX model (.onnx)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"], help="Inference device")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--results-dir", default="results", help="Directory for result masks")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from road_segmentation.api.app import create_app

    app = create_app(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        device=args.device,
        results_dir=args.results_dir,
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
