"""Download model artifacts from W&B Model Registry.

Usage:
    # Download latest model (ONNX + checkpoint):
    python scripts/download_model.py

    # Download specific version:
    python scripts/download_model.py --version v1

    # Download only ONNX models (for deployment):
    python scripts/download_model.py --onnx-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model from W&B Artifacts")
    parser.add_argument("--project", default="road-segmentation")
    parser.add_argument("--entity", default=None, help="W&B entity (username/team)")
    parser.add_argument("--name", default="road-segmentation-model", help="Artifact name")
    parser.add_argument("--version", default="latest", help="Artifact version (e.g. v0, v1, latest)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT, help="Output base directory")
    parser.add_argument("--onnx-only", action="store_true", help="Download only ONNX models")
    args = parser.parse_args()

    from road_segmentation.env import load_env
    load_env()

    import wandb

    api = wandb.Api()

    entity = args.entity
    if not entity:
        import os
        entity = os.environ.get("WANDB_ENTITY", api.default_entity)

    artifact_path = f"{entity}/{args.project}/{args.name}:{args.version}"
    print(f"Downloading: {artifact_path}")

    artifact = api.artifact(artifact_path)
    download_dir = Path(artifact.download())

    # Copy files to the right places
    models_dir = args.output / "models"
    checkpoints_dir = args.output / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for f in download_dir.iterdir():
        if f.suffix == ".onnx":
            dest = models_dir / f.name
            shutil.copy2(f, dest)
            print(f"  ONNX: {dest} ({f.stat().st_size / 1e6:.0f} MB)")
        elif f.suffix == ".pth" and not args.onnx_only:
            dest = checkpoints_dir / f.name
            shutil.copy2(f, dest)
            print(f"  Checkpoint: {dest} ({f.stat().st_size / 1e6:.0f} MB)")

    print(f"\nDone. Models at: {models_dir}")
    print(f"Start the API: python scripts/serve.py --onnx {models_dir / 'UnetPlusPlus_efficientnet-b4_int8.onnx'} --device cpu")


if __name__ == "__main__":
    main()
