"""Export and optimize model for production deployment.

Usage:
    # Export for both CPU and GPU:
    python scripts/optimize.py --checkpoint checkpoints/best.pth --output models/

    # CPU only:
    python scripts/optimize.py --checkpoint checkpoints/best.pth --output models/ --targets cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize model for production")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--targets", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from road_segmentation.api.optimize import optimize_for_targets

    results = optimize_for_targets(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        targets=args.targets,
    )

    print("\nOptimized models:")
    for target, path in results.items():
        size = path.stat().st_size / 1e6
        print(f"  {target:12s} → {path} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
