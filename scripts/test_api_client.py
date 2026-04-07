"""Test the inference API by simulating a client.

Sends images to the running API, saves the returned mask,
and creates overlay visualizations. Optionally uses validation set
images with ground truth for quality assessment.

Usage:
    # First start the server:
    #   python scripts/serve.py --checkpoint checkpoints/best.pth

    # Test with random training images (no ground truth):
    python scripts/test_api_client.py --n 5

    # Test with validation set (includes ground truth comparison):
    python scripts/test_api_client.py --val-set --n 10

    # Single image:
    python scripts/test_api_client.py --image data/raw/.../100034_sat.jpg

    # With GeoJSON:
    python scripts/test_api_client.py --val-set --n 5 --geojson
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth binary masks."""
    p = (pred > 0).astype(bool)
    g = (gt > 0).astype(bool)
    intersection = (p & g).sum()
    union = (p | g).sum()
    return float(intersection / max(union, 1))


def create_prediction_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Red overlay of predicted roads on the image."""
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    overlay = image.copy()
    road_px = mask > 0
    overlay[road_px] = ((1 - alpha) * overlay[road_px] + alpha * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


def create_error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """TP=green, FP=red, FN=blue overlay on the image.

    This is the most informative visualization — it shows exactly
    where the model is correct, where it hallucinates roads, and
    where it misses real roads.
    """
    if pred.shape[:2] != image.shape[:2]:
        pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    if gt.shape[:2] != image.shape[:2]:
        gt = cv2.resize(gt, (image.shape[1], image.shape[0]))

    p = pred > 0
    g = gt > 0
    tp = p & g
    fp = p & ~g
    fn = ~p & g

    overlay = image.astype(np.float32)
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha   # Green = correct
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([0, 0, 255]) * alpha   # Red = false alarm
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([255, 0, 0]) * alpha   # Blue = missed road
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_gt_comparison(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou: float,
    stem: str,
) -> np.ndarray:
    """Create a 2x3 panel: top row = image, GT, GT overlay; bottom = pred, error map, metrics."""
    h, w = image.shape[:2]

    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h))
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h))

    # Convert masks to BGR for stacking
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    # GT overlay (green)
    gt_overlay = image.copy()
    gt_px = gt_mask > 0
    gt_overlay[gt_px] = ((0.6) * gt_overlay[gt_px] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    # Error overlay (TP/FP/FN)
    error_overlay = create_error_overlay(image, pred_mask, gt_mask)

    # Metrics panel (black background with white text)
    metrics_panel = np.zeros((h, w, 3), dtype=np.uint8)
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)
    tp_count = int((p & g).sum())
    fp_count = int((p & ~g).sum())
    fn_count = int((~p & g).sum())
    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    gt_coverage = g.mean() * 100
    pred_coverage = p.mean() * 100

    texts = [
        f"Sample: {stem}",
        f"IoU: {iou:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"GT coverage: {gt_coverage:.1f}%",
        f"Pred coverage: {pred_coverage:.1f}%",
        f"TP: {tp_count:,}  FP: {fp_count:,}  FN: {fn_count:,}",
        "",
        "Legend:",
        "  Green = True Positive (correct road)",
        "  Red   = False Positive (hallucinated)",
        "  Blue  = False Negative (missed road)",
    ]
    for i, text in enumerate(texts):
        y = 40 + i * 35
        scale = 0.7 if i < 7 else 0.6
        color = (255, 255, 255) if i < 7 else (180, 180, 180)
        cv2.putText(metrics_panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # Assemble: top = [image, GT mask, GT overlay] ; bottom = [pred mask, error map, metrics]
    top_row = np.hstack([image, gt_bgr, gt_overlay])
    bottom_row = np.hstack([pred_bgr, error_overlay, metrics_panel])
    panel = np.vstack([top_row, bottom_row])

    # Add row labels
    label_h = 30
    labeled = np.zeros((panel.shape[0] + label_h * 2, panel.shape[1], 3), dtype=np.uint8)
    labeled[label_h:label_h + h, :] = top_row
    labeled[label_h * 2 + h:, :] = bottom_row

    col_labels = ["Input Image", "Ground Truth", "GT Overlay (green)"]
    col_labels2 = ["Prediction", "Error Map (TP/FP/FN)", "Metrics"]
    for i, label in enumerate(col_labels):
        x = i * w + w // 2 - len(label) * 5
        cv2.putText(labeled, label, (x, label_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, label in enumerate(col_labels2):
        x = i * w + w // 2 - len(label) * 5
        cv2.putText(labeled, label, (x, label_h + h + label_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return labeled


def process_image(
    api_url: str,
    image_path: Path,
    output_dir: Path,
    gt_mask_path: Path | None = None,
    threshold: float = 0.5,
    geojson: bool = False,
) -> dict:
    """Process a single image: send to API, save results and visualizations."""
    stem = image_path.stem.replace("_sat", "")

    print(f"\n{'='*60}")
    print(f"  Processing: {image_path.name}")
    if gt_mask_path:
        print(f"  Ground truth: {gt_mask_path.name}")
    print(f"{'='*60}")

    # Send to API
    t0 = time.time()
    result = send_image(api_url, image_path, threshold=threshold, geojson=geojson)
    client_time = (time.time() - t0) * 1000

    print(f"  Road coverage: {result['road_coverage_pct']}%")
    print(f"  Confidence:    {result['confidence_mean']}")
    print(f"  Server time:   {result['metadata']['inference_time_ms']} ms")
    print(f"  Client time:   {client_time:.0f} ms")

    # Download predicted mask
    pred_mask = download_mask(api_url, result["mask_url"])

    # Load original image
    image = cv2.imread(str(image_path))

    # --- Always save: image, mask, prediction overlay, side-by-side ---
    pred_overlay = create_prediction_overlay(image, pred_mask)

    cv2.imwrite(str(output_dir / f"{stem}_image.png"), image)
    cv2.imwrite(str(output_dir / f"{stem}_pred_mask.png"), pred_mask)
    cv2.imwrite(str(output_dir / f"{stem}_pred_overlay.png"), pred_overlay)

    h, w = image.shape[:2]
    pred_bgr = cv2.cvtColor(pred_mask if pred_mask.shape[:2] == (h, w) else cv2.resize(pred_mask, (w, h)), cv2.COLOR_GRAY2BGR)
    simple_comparison = np.hstack([image, pred_bgr, pred_overlay])
    cv2.imwrite(str(output_dir / f"{stem}_comparison.png"), simple_comparison)

    print(f"  Saved: {stem}_image.png, {stem}_pred_mask.png, {stem}_pred_overlay.png, {stem}_comparison.png")

    # --- If ground truth available: full evaluation panel ---
    if gt_mask_path and gt_mask_path.exists():
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)

        # Binarize GT (original masks are RGB white/black, grayscale gives 0/255)
        gt_binary = ((gt_mask > 0) * 255).astype(np.uint8)

        iou = compute_iou(pred_mask, gt_binary)
        result["iou"] = iou
        print(f"  IoU: {iou:.4f}")

        # Error overlay
        error_overlay = create_error_overlay(image, pred_mask, gt_binary)
        cv2.imwrite(str(output_dir / f"{stem}_error_map.png"), error_overlay)

        # Full comparison panel
        panel = create_gt_comparison(image, pred_mask, gt_binary, iou, stem)
        cv2.imwrite(str(output_dir / f"{stem}_full_panel.png"), panel)

        # GT mask and overlay
        cv2.imwrite(str(output_dir / f"{stem}_gt_mask.png"), gt_binary)

        print(f"  Saved: {stem}_error_map.png, {stem}_full_panel.png, {stem}_gt_mask.png")

    # Save GeoJSON
    if geojson and result.get("geojson"):
        geojson_out = output_dir / f"{stem}_roads.geojson"
        with open(geojson_out, "w") as f:
            json.dump(result["geojson"], f, indent=2)
        n_seg = result["geojson"].get("properties", {}).get("num_segments", 0)
        print(f"  Saved: {stem}_roads.geojson ({n_seg} segments)")

    return result


def get_val_pairs(n: int) -> list[tuple[Path, Path]]:
    """Get validation set image-mask pairs using the same split as training."""
    from road_segmentation.data.eda import discover_image_mask_pairs
    from road_segmentation.data.split import split_pairs
    from road_segmentation.paths import DEEPGLOBE_DATASET_DIR

    pairs = discover_image_mask_pairs(DEEPGLOBE_DATASET_DIR)
    _, val_pairs = split_pairs(pairs, val_ratio=0.15, seed=42)

    random.seed(42)
    selected = random.sample(val_pairs, min(n, len(val_pairs)))
    return [(p.image_path, p.mask_path) for p in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Road Segmentation API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", type=Path, help="Single image to process")
    parser.add_argument("--gt-mask", type=Path, help="Ground truth mask for --image (optional)")
    parser.add_argument("--image-dir", type=Path, help="Directory of images (picks random samples)")
    parser.add_argument("--val-set", action="store_true",
                        help="Use validation set images with ground truth masks")
    parser.add_argument("--n", type=int, default=5, help="Number of images to process")
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
        print("Start the server first: python scripts/serve.py --checkpoint checkpoints/best.pth")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect images to process
    image_gt_pairs = []  # list of (image_path, gt_mask_path_or_None)

    if args.image:
        image_gt_pairs = [(args.image, args.gt_mask)]

    elif args.val_set:
        print("\nLoading validation set (same split as training)...")
        val_pairs = get_val_pairs(args.n)
        image_gt_pairs = val_pairs
        print(f"Selected {len(image_gt_pairs)} validation samples with ground truth")

    elif args.image_dir:
        all_images = sorted(args.image_dir.glob("*_sat.*"))
        if not all_images:
            all_images = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))
        random.seed(42)
        selected = random.sample(all_images, min(args.n, len(all_images)))
        # Try to find matching masks
        for img_path in selected:
            mask_path = img_path.parent / img_path.name.replace("_sat.", "_mask.")
            gt = mask_path if mask_path.exists() else None
            image_gt_pairs.append((img_path, gt))

    else:
        # Default: random training images
        default_dir = PROJECT_ROOT / "data" / "raw" / "deepglobe-road-extraction-dataset" / "train"
        if default_dir.exists():
            all_images = sorted(default_dir.glob("*_sat.*"))
            random.seed(42)
            selected = random.sample(all_images, min(args.n, len(all_images)))
            for img_path in selected:
                mask_path = img_path.parent / img_path.name.replace("_sat.", "_mask.")
                gt = mask_path if mask_path.exists() else None
                image_gt_pairs.append((img_path, gt))
        else:
            print("No images found. Use --image, --image-dir, or --val-set")
            sys.exit(1)

    print(f"\nProcessing {len(image_gt_pairs)} images...")
    print(f"Ground truth available: {sum(1 for _, gt in image_gt_pairs if gt is not None)}")
    print(f"Output: {args.output}")

    all_results = []
    for img_path, gt_path in image_gt_pairs:
        result = process_image(
            args.api_url, img_path, args.output,
            gt_mask_path=gt_path,
            threshold=args.threshold,
            geojson=args.geojson,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    times = [r["metadata"]["inference_time_ms"] for r in all_results]
    coverages = [r["road_coverage_pct"] for r in all_results]
    print(f"  Images processed: {len(all_results)}")
    print(f"  Avg inference:    {np.mean(times):.1f} ms")
    print(f"  Avg coverage:     {np.mean(coverages):.1f}%")

    # IoU summary if ground truth was available
    ious = [r["iou"] for r in all_results if "iou" in r]
    if ious:
        print(f"  Avg IoU:          {np.mean(ious):.4f}")
        print(f"  Min IoU:          {np.min(ious):.4f}")
        print(f"  Max IoU:          {np.max(ious):.4f}")

    print(f"  Results saved to: {args.output}")

    # Save summary JSON
    summary = {
        "n_images": len(all_results),
        "avg_inference_ms": round(float(np.mean(times)), 1),
        "avg_coverage_pct": round(float(np.mean(coverages)), 2),
        "threshold": args.threshold,
    }
    if ious:
        summary["avg_iou"] = round(float(np.mean(ious)), 4)
        summary["min_iou"] = round(float(np.min(ious)), 4)
        summary["max_iou"] = round(float(np.max(ious)), 4)
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
