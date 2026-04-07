"""Upload model artifacts to W&B Model Registry.

Usage:
    python scripts/upload_model.py \
        --checkpoint checkpoints/effecientnet_b4_1024_boundary_loss.pth \
        --onnx-dir models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload models to W&B Artifacts")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx-dir", type=Path, help="Directory with ONNX models")
    parser.add_argument("--project", default="road-segmentation")
    parser.add_argument("--name", default="road-segmentation-model", help="Artifact name")
    parser.add_argument("--version", default="v1.0", help="Model version tag")
    args = parser.parse_args()

    from road_segmentation.env import load_env
    load_env()

    import wandb

    run = wandb.init(project=args.project, job_type="upload-model")

    artifact = wandb.Artifact(
        name=args.name,
        type="model",
        description=f"Road segmentation model {args.version}",
        metadata={"version": args.version},
    )

    # Add PyTorch checkpoint
    if args.checkpoint and args.checkpoint.exists():
        artifact.add_file(str(args.checkpoint), name="checkpoint.pth")
        size = args.checkpoint.stat().st_size / 1e6
        print(f"Added checkpoint: {args.checkpoint.name} ({size:.0f} MB)")

    # Add ONNX models
    if args.onnx_dir and args.onnx_dir.exists():
        for onnx_file in sorted(args.onnx_dir.glob("*.onnx")):
            artifact.add_file(str(onnx_file), name=onnx_file.name)
            size = onnx_file.stat().st_size / 1e6
            print(f"Added ONNX: {onnx_file.name} ({size:.0f} MB)")

    run.log_artifact(artifact)
    wandb.finish()

    print(f"\nUploaded artifact: {args.name}")
    print(f"Download with: python scripts/download_model.py")


if __name__ == "__main__":
    main()
