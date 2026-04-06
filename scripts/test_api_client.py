"""Test the inference API by simulating a client.

Sends images to the running API, saves the returned mask,
and creates an overlay visualization.

Usage:
    # First start the server:
    #   python scripts/serve.py --checkpoint checkpoints/best.pth
    #
    # Then run this script:
    python scripts/test_api_client.py --image data/raw/deepglobe-road-extraction-dataset/train/100034_sat.jpg
    python scripts/test_api_client.py --image-dir data/raw/deepglobe-road-extraction-dataset/train/ --n 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def send_image(api_url: str, image_path: Path, threshold: float = 0.5, geojson: bool = False) -> dict:
    """Send an image to the API and return the response."""
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        params = {"threshold": threshold, "return_geojson": geojson}
        response = requests.post(f"{api_url}/api/v1/segment", files=files, params=params)

    response.raise_for_status()
    return response.json()


def download_mask(api_url: str, mask_url: str) -> np.ndarray:
    """Download the mask image from the API."""
    response = requests.get(f"{api_url}{mask_url}")
    response.raise_for_status()
    arr = np.frombuffer(response.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create a red overlay of the road mask on the original image."""
    # Resize mask to match image if different
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    overlay = image.copy()
    road_pixels = mask > 0
    overlay[road_pixels] = (
        (1 - alpha) * overlay[road_pixels] + alpha * np.array([0, 0, 255])  # Red in BGR
    ).astype(np.uint8)
    return overlay


def process_image(
    api_url: str,
    image_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    geojson: bool = False,
) -> dict:
    """Process a single image: send to API, save mask and overlay."""
    stem = image_path.stem.replace("_sat", "")

    print(f"\n{'='*60}")
    print(f"  Processing: {image_path.name}")
    print(f"{'='*60}")

    # Send to API
    t0 = time.time()
    result = send_image(api_url, image_path, threshold=threshold, geojson=geojson)
    client_time = (time.time() - t0) * 1000

    print(f"  Road coverage: {result['road_coverage_pct']}%")
    print(f"  Confidence:    {result['confidence_mean']}")
    print(f"  Server time:   {result['metadata']['inference_time_ms']} ms")
    print(f"  Client time:   {client_time:.0f} ms (includes network + upload)")

    # Download mask
    mask = download_mask(api_url, result["mask_url"])

    # Load original image
    image = cv2.imread(str(image_path))

    # Create overlay
    overlay = create_overlay(image, mask)

    # Save all three
    image_out = output_dir / f"{stem}_image.png"
    mask_out = output_dir / f"{stem}_mask.png"
    overlay_out = output_dir / f"{stem}_overlay.png"
    side_by_side_out = output_dir / f"{stem}_comparison.png"

    cv2.imwrite(str(image_out), image)
    cv2.imwrite(str(mask_out), mask)
    cv2.imwrite(str(overlay_out), overlay)

    # Side-by-side comparison (image | mask | overlay)
    h, w = image.shape[:2]
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if mask_rgb.shape[:2] != image.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (w, h))
    comparison = np.hstack([image, mask_rgb, overlay])
    cv2.imwrite(str(side_by_side_out), comparison)

    print(f"  Saved:")
    print(f"    Image:      {image_out}")
    print(f"    Mask:       {mask_out}")
    print(f"    Overlay:    {overlay_out}")
    print(f"    Comparison: {side_by_side_out}")

    # Save GeoJSON if available
    if geojson and result.get("geojson"):
        geojson_out = output_dir / f"{stem}_roads.geojson"
        with open(geojson_out, "w") as f:
            json.dump(result["geojson"], f, indent=2)
        n_segments = result["geojson"].get("properties", {}).get("num_segments", 0)
        print(f"    GeoJSON:    {geojson_out} ({n_segments} segments)")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Road Segmentation API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", type=Path, help="Single image to process")
    parser.add_argument("--image-dir", type=Path, help="Directory of images (picks random samples)")
    parser.add_argument("--n", type=int, default=5, help="Number of random images from --image-dir")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "test_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--geojson", action="store_true", help="Also request GeoJSON output")
    args = parser.parse_args()

    # Check API is running
    try:
        health = requests.get(f"{args.api_url}/health").json()
        print(f"API status: {health['status']}")
        model_info = requests.get(f"{args.api_url}/api/v1/model-info").json()
        print(f"Model: {model_info}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to API at {args.api_url}")
        print(f"Start the server first: python scripts/serve.py --checkpoint checkpoints/best.pth")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect images to process
    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        all_images = sorted(args.image_dir.glob("*_sat.*"))
        if not all_images:
            all_images = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))
        random.seed(42)
        images = random.sample(all_images, min(args.n, len(all_images)))
    else:
        # Default: use a few samples from the dataset
        default_dir = PROJECT_ROOT / "data" / "raw" / "deepglobe-road-extraction-dataset" / "train"
        if default_dir.exists():
            all_images = sorted(default_dir.glob("*_sat.*"))
            random.seed(42)
            images = random.sample(all_images, min(args.n, len(all_images)))
        else:
            print("No images found. Use --image or --image-dir")
            sys.exit(1)

    print(f"\nProcessing {len(images)} images...")
    print(f"Output: {args.output}")

    all_results = []
    for img_path in images:
        result = process_image(
            args.api_url, img_path, args.output,
            threshold=args.threshold, geojson=args.geojson,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    times = [r["metadata"]["inference_time_ms"] for r in all_results]
    coverages = [r["road_coverage_pct"] for r in all_results]
    print(f"  Images processed: {len(all_results)}")
    print(f"  Avg inference:    {np.mean(times):.1f} ms")
    print(f"  Avg coverage:     {np.mean(coverages):.1f}%")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
