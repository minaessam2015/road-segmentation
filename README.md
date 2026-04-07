# Road Segmentation from Satellite Imagery

[![CI](https://github.com/minaessam2015/road-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/minaessam2015/road-segmentation/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

End-to-end road extraction system that trains a segmentation model on satellite images and serves predictions via an inference API with ONNX optimization.

| | |
|---|---|
| **Live Experiments** | [wandb.ai/minaessam/road-segmentation](https://wandb.ai/minaessam/road-segmentation) |
| **Docker Image** | `ghcr.io/minaessam2015/road-segmentation:latest` |
| **Best Val IoU** | 0.7016 (Dice: 0.8246) |
| **Inference Latency** | 78 ms (ONNX FP16, T4 GPU) |

---

## Table of Contents

- [Quick Start with Docker](#quick-start-with-docker)
- [Manual Setup](#manual-setup)
- [Training](#training)
- [Inference API](#inference-api)
- [Testing](#testing)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)
- [Results](#results)
- [Design Decisions](#design-decisions)
- [Production Considerations](#production-considerations)

---

## Quick Start with Docker

### Option A: Auto-download model (single command)

```bash
git clone https://github.com/minaessam2015/road-segmentation.git
cd road-segmentation

# Set your W&B API key (get it from https://wandb.ai/authorize)
echo "WANDB_API_KEY=your_key_here" > .env

# Build and run — model downloads automatically on first start
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Option B: Pre-downloaded model

```bash
# Download model first
pip install wandb
python scripts/download_model.py --onnx-only

# Run with mounted model
docker-compose up --build
```

### Option C: Pull pre-built image

```bash
docker pull ghcr.io/minaessam2015/road-segmentation:latest

# Run with local model directory
docker run -p 8000:8000 \
  -v ./models:/app/models \
  -e WANDB_API_KEY=your_key \
  ghcr.io/minaessam2015/road-segmentation:latest
```

### Test the running API

```bash
# Health check
curl http://localhost:8000/health

# Segment a satellite image
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@satellite.jpg" | python -m json.tool

# With GeoJSON road polylines
curl -X POST "http://localhost:8000/api/v1/segment?return_geojson=true" \
  -F "image=@satellite.jpg" | python -m json.tool

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Manual Setup

### Prerequisites

- Python 3.9+
- [Kaggle API token](https://www.kaggle.com/settings) (for dataset download)
- [W&B API key](https://wandb.ai/authorize) (for model download and experiment tracking)

### Installation

```bash
git clone https://github.com/minaessam2015/road-segmentation.git
cd road-segmentation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your credentials:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_kaggle_key
#   WANDB_API_KEY=your_wandb_key
```

### Download Dataset

```bash
python scripts/download_data.py
```

Downloads the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) (6,226 satellite images with road masks, 1024x1024 pixels).

### Download Model Weights

```bash
# All models (PyTorch checkpoint + ONNX variants):
python scripts/download_model.py

# ONNX only (for deployment):
python scripts/download_model.py --onnx-only
```

### Run the API

```bash
# CPU serving (INT8 quantized, 21 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_int8.onnx --device cpu

# GPU serving (FP16, 40 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_fp16.onnx --device gpu

# From PyTorch checkpoint:
python scripts/serve.py --checkpoint checkpoints/best.pth
```

### Test the API with validation images

```bash
# Start the server in one terminal, then:
python scripts/test_api_client.py --val-set --n 10 --geojson
```

This sends validation images (with ground truth) to the API and saves: prediction overlays, error maps (TP/FP/FN), side-by-side comparisons, and per-sample IoU scores.

---

## Training

### Quick Experiment (local, CPU)

```bash
python scripts/train.py --config configs/unet_resnet34.yaml \
    --override training.epochs=5 data.subset_size=200 data.num_workers=0
```

### Full Training (Colab with GPU)

Open the training notebook on Colab, set your credentials, and run all cells. Checkpoints save to Google Drive automatically. On disconnect, re-run to resume from the latest checkpoint.

See [Notebooks](#notebooks) section below for the Colab link.

### Config System

Each architecture has a YAML config. Override any parameter from the CLI:

```bash
python scripts/train.py --config configs/unet_resnet34.yaml \
    --override training.epochs=30 optimizer.lr=0.0005 data.batch_size=16
```

Available configs:

| Config | Architecture | Notes |
|--------|-------------|-------|
| `unet_resnet34.yaml` | U-Net + ResNet34 | Baseline |
| `unet_efficientnet_b4.yaml` | U-Net + EfficientNet-B4 | Stronger encoder |
| `unetplusplus_efficientnet_b4_1024.yaml` | UNet++ + EfficientNet-B4 @ 1024 | **Best model** |
| `deeplabv3plus_resnet50.yaml` | DeepLabV3+ + ResNet50 | Multi-scale context |
| `linknet_resnet34.yaml` | LinkNet + ResNet34 | Fastest inference |
| `segformer_mitb2.yaml` | SegFormer + MIT-B2 | Transformer architecture |
| `finetune_boundary_loss.yaml` | Boundary-weighted loss | Fine-tuning from checkpoint |

### Model Export and Optimization

```bash
# Export trained model to ONNX (FP32 + FP16 + INT8):
python scripts/optimize.py --checkpoint checkpoints/best.pth --output models/ --targets cpu gpu
```

---

## Inference API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/segment` | Upload satellite image, get road mask |
| `GET` | `/health` | Health check (model status, disk, GPU memory) |
| `GET` | `/api/v1/model-info` | Model version and configuration |
| `GET` | `/metrics` | Prometheus-compatible metrics |

### `POST /api/v1/segment`

**Parameters:**
- `image` (file, required): Satellite image (PNG, JPG, TIFF)
- `threshold` (float, optional): Binarization threshold, default 0.5
- `return_geojson` (bool, optional): Include road polylines as GeoJSON

**Response:**
```json
{
  "mask_url": "/results/abc123_mask.png",
  "road_coverage_pct": 4.2,
  "confidence_mean": 0.87,
  "metadata": {
    "image_size": [1024, 1024],
    "model_version": "UnetPlusPlus_efficientnet-b4",
    "inference_time_ms": 78.3,
    "threshold": 0.5,
    "request_id": "abc123"
  },
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "LineString", "coordinates": [[x1,y1], [x2,y2], ...] },
        "properties": { "length_px": 245.3 }
      }
    ]
  }
}
```

### Observability

- **Structured JSON logging**: Every request logged with timestamp, request ID, duration, status
- **Request tracing**: `X-Request-ID` header propagated through all stages
- **Inference audit trail**: Image hash, prediction stats, model version per request
- **Prometheus metrics**: Request counts, latency p50/p95/p99, error rates at `/metrics`
- **Health check**: Model status, disk space, GPU memory at `/health`

---

## Testing

```bash
# Unit tests only (no model weights needed, runs in CI):
pytest tests/ --ignore=tests/test_e2e.py

# Full suite including end-to-end (needs model weights in checkpoints/ and models/):
pytest tests/

# E2E with specific model paths:
pytest tests/test_e2e.py --checkpoint checkpoints/best.pth --onnx models/model_int8.onnx
```

**69 tests** covering:
- Config loading, validation, CLI overrides
- Data transforms, normalization, resize
- Model forward pass, encoder freeze/unfreeze, parameter counting
- All 7 loss functions, boundary weight computation
- Post-processing pipeline (threshold, morphology, skeletonization)
- API endpoints with mocked model
- End-to-end: real model checkpoint → ONNX → API → verify response

---

## Notebooks

All notebooks are designed to run on Google Colab with GPU. Click the Colab badge to open directly.

| # | Notebook | Open in Colab | Description |
|---|----------|---------------|-------------|
| 01 | [EDA](notebooks/01_eda.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/01_eda.ipynb) | Data exploration: class imbalance, road width, spatial patterns, mask quality. All downstream decisions (loss, metrics, augmentations) are justified here. |
| 02 | [Model Comparison](notebooks/02_model_comparison.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/02_model_comparison.ipynb) | Quick benchmark of 4 architectures (U-Net, LinkNet, SegFormer, DeepLabV3+) on 1000 samples. Produces ranked comparison table and overlaid training curves. |
| 03 | [Training](notebooks/03_training.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/03_training.ipynb) | Full training with W&B tracking and Google Drive checkpoints. Auto-resumes on Colab disconnect. |
| 04 | [Data Leakage](notebooks/04_data_leakage_check.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/04_data_leakage_check.ipynb) | Train/val similarity analysis using ResNet50 + CLIP embeddings. Quantifies metric inflation from spatial leakage. |
| 05 | [Error Analysis](notebooks/05_error_analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/05_error_analysis.ipynb) | Failure patterns, FP/FN balance, performance by road width/coverage, spatial error heatmaps, calibration (reliability diagram, ECE, temperature scaling). |
| 06 | [Post-Processing](notebooks/06_postprocessing_ablation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/06_postprocessing_ablation.ipynb) | Ablation study: per-step IoU contribution of threshold tuning, component filtering, morphology, gap bridging, TTA. |
| 07 | [Model Optimization](notebooks/07_model_optimization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/07_model_optimization.ipynb) | ONNX export, FP16/INT8 compression, speed + accuracy benchmark on GPU. |

---

## Project Structure

```
road-segmentation/
    src/road_segmentation/
        api/                        # Inference service
            app.py                  #   FastAPI endpoints
            inference.py            #   PyTorch inference engine + GeoJSON
            optimize.py             #   ONNX export, INT8/FP16
            observability.py        #   Logging, tracing, Prometheus
        data/                       # Dataset handling
            dataset.py              #   PyTorch Dataset + DataLoader
            transforms.py           #   Albumentations pipelines
            split.py                #   Stratified train/val split
            eda.py                  #   EDA utilities
            download.py             #   Kaggle dataset download
        models/
            factory.py              #   SMP model creation + encoder management
        training/
            trainer.py              #   Training loop (AMP, W&B, checkpointing)
            losses.py               #   7 loss functions incl. boundary-weighted
            metrics.py              #   IoU, Dice, Precision, Recall
            callbacks.py            #   Early stopping, Model EMA
            checkpoint.py           #   Save/load with resume
            visualization.py        #   Curves, prediction overlays
        postprocessing/
            steps.py                #   Individual steps (threshold, morph, skeleton)
            pipeline.py             #   Composable toggleable pipeline
        config.py                   #   YAML config + CLI overrides
        env.py                      #   .env credential loading
        paths.py                    #   Path constants
    configs/                        #   9 YAML architecture configs
    scripts/
        train.py                    #   Training CLI
        serve.py                    #   API server CLI
        optimize.py                 #   Model export CLI
        download_model.py           #   Download from W&B registry
        upload_model.py             #   Upload to W&B registry
        test_api_client.py          #   API test with GT comparison
        entrypoint.sh               #   Docker auto-download entrypoint
    notebooks/                      #   7 analysis notebooks (see above)
    tests/                          #   69 tests (unit + e2e)
    .github/workflows/ci.yml       #   CI: lint, test, Docker, GHCR push
    Dockerfile                      #   Multi-stage inference image
    docker-compose.yml              #   Single-command deployment
```

---

## Results

### Final Model

| Metric | Value |
|--------|-------|
| Architecture | UNet++ + EfficientNet-B4 |
| Input | 1024x1024 (full resolution) |
| **Val IoU** | **0.7016** |
| **Val Dice** | **0.8246** |
| Training | 50 epochs, 5292 train / 934 val |
| Cleaned IoU (leakage-adjusted) | 0.675 |

### Architecture Comparison (1000 samples, 15 epochs)

| Rank | Model | IoU | Dice | Params | Time |
|------|-------|-----|------|--------|------|
| 1 | **U-Net + ResNet34** | **0.528** | 0.691 | 24.4M | 10 min |
| 2 | LinkNet + ResNet34 | 0.507 | 0.673 | 21.8M | 9 min |
| 3 | SegFormer + MIT-B2 | 0.485 | 0.653 | 24.7M | 15 min |
| 4 | DeepLabV3+ + ResNet50 | 0.484 | — | 26.7M | 10 min |

### Optimization (T4 GPU)

| Variant | Latency | Speedup | IoU | Size |
|---------|---------|---------|-----|------|
| PyTorch FP32 | 145.6 ms | 1.00x | 0.6835 | — |
| PyTorch FP16/AMP | 95.5 ms | 1.52x | 0.6835 | — |
| ONNX FP32 | 135.7 ms | 1.07x | 0.6835 | 80 MB |
| **ONNX FP16** | **78.3 ms** | **1.86x** | **0.6835** | **40 MB** |
| ONNX INT8 (CPU) | — | — | — | 21 MB |

Zero IoU loss across all compression variants.

### Post-Processing Ablation

| Step | IoU | Delta | Verdict |
|------|-----|-------|---------|
| Raw threshold (0.5) | 0.7016 | baseline | — |
| Optimal threshold | 0.7017 | +0.01% | Neutral |
| + Component filter | 0.7019 | +0.02% | Neutral |
| + Morph. closing | 0.7012 | -0.04% | Neutral |
| TTA (6-fold) | 0.7014 | +0.65%* | Include |

Model produces clean predictions — minimal post-processing needed.

### Data Leakage

| Metric | Original | Cleaned | Inflation |
|--------|----------|---------|-----------|
| Mean IoU | 0.694 | 0.675 | 1.9% |

123 potentially leaky val samples identified using ResNet50 + CLIP embeddings. Metrics slightly optimistic — both values reported for transparency.

---

## Design Decisions

### Architecture

**UNet++ + EfficientNet-B4** selected via systematic 4-model comparison. UNet++ dense skip connections preserve thin road features. EfficientNet-B4 provides NAS-optimized compound scaling. Full 1024x1024 resolution preserves road detail without downsampling.

We chose not to use D-LinkNet (DeepGlobe winner) because its code requires Python 2.7 / PyTorch 0.2 and weights are on untrusted file-sharing links. SMP's architectures provide production-quality alternatives.

### Loss Function

**BCE + Dice compound loss** handles severe class imbalance (roads ~4% of pixels). BCE provides stable gradients; Dice optimizes the overlap metric directly. Boundary-weighted loss (5x on edge pixels) was tested but showed minimal gain — the model was already well-calibrated.

### Evaluation

**IoU** as primary metric (not pixel accuracy, which is meaningless at 96% background). Also report Dice, Precision, Recall. Production monitoring includes calibration tracking (ECE, reliability diagrams).

### Data Integrity

Spatial leakage analysis (ResNet50 + CLIP embeddings) found 1.9% IoU inflation from adjacent tiles in train/val split. Both original and cleaned metrics reported.

---

## Production Considerations

### Scalability
Stateless service enables horizontal scaling behind a load balancer. Batch processing via task queues (Celery/Redis). Large images tiled with overlap and stitched with central-crop blending.

### Latency
ONNX FP16 at 78ms/image on T4 GPU (1.86x speedup, zero accuracy loss). Further: TensorRT for 2-3x more, batching, INT8 for CPU (21 MB model).

### Model Lifecycle
W&B Model Registry for versioning. `python scripts/download_model.py --version v1` for any version. A/B testing via traffic routing. One-command rollback. CI/CD pushes Docker images to GHCR on every merge.

### Data Drift
Monitor input RGB statistics (KL divergence), prediction distributions (coverage, confidence), and calibration (ECE) over time. Alert and retrigger training when drift detected.

### Dataset Versioning
DeepGlobe is fixed, so formal versioning (DVC) not implemented. Reproducibility ensured via deterministic split (seed=42) recorded in every checkpoint config. Production would add content-hash tracking via W&B dataset artifacts.

---

## License

This project uses the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) which must be downloaded separately via the Kaggle API. The dataset is not included in this repository.
