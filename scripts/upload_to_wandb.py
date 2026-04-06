"""Upload training results to W&B from saved CSV metrics.

Usage:
    python scripts/upload_to_wandb.py --log-dir logs/Unet_resnet34_20260406_045917

This reads metrics.csv and config.yaml from the log directory and
creates a complete W&B run with all epoch metrics, as if W&B had
been active during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload training results to W&B")
    parser.add_argument("--log-dir", type=Path, required=True, help="Path to experiment log directory")
    parser.add_argument("--project", default="road-segmentation", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (username or team)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    csv_path = log_dir / "metrics.csv"
    config_path = log_dir / "config.yaml"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load env for API key
    from road_segmentation.env import load_env
    load_env()

    import csv
    import yaml
    import wandb

    # Load config if available
    config_dict = {}
    if config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

    # Derive experiment name from directory
    experiment_name = log_dir.name

    # Read metrics
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: metrics.csv is empty")
        sys.exit(1)

    print(f"Found {len(rows)} epochs in {csv_path}")
    print(f"Experiment: {experiment_name}")

    # Create W&B run
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=experiment_name,
        config=config_dict,
        settings=wandb.Settings(init_timeout=180),
    )

    # Log each epoch
    for row in rows:
        step = int(float(row["epoch"]))
        log_dict = {}
        for key, value in row.items():
            try:
                log_dict[key] = float(value)
            except (ValueError, TypeError):
                continue
        wandb.log(log_dict, step=step)

    # Set best metrics as summary
    best_row = max(rows, key=lambda r: float(r.get("val_iou", 0)))
    for key, value in best_row.items():
        try:
            wandb.run.summary[f"best_{key}"] = float(value)
        except (ValueError, TypeError):
            continue

    # Upload visualization images if they exist
    viz_dir = log_dir / "visualizations"
    if viz_dir.exists():
        for img_path in sorted(viz_dir.glob("*.png")):
            wandb.log({f"predictions/{img_path.stem}": wandb.Image(str(img_path))})
            print(f"  Uploaded: {img_path.name}")

    # Upload training curves
    curves_path = log_dir / "training_curves.png"
    if curves_path.exists():
        wandb.log({"training_curves": wandb.Image(str(curves_path))})
        print("  Uploaded: training_curves.png")

    wandb.finish()

    print(f"\nDone! View at: {run.url}")


if __name__ == "__main__":
    main()
