# Road Segmentation from Satellite Imagery

[![CI](https://github.com/minaessam2015/road-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/minaessam2015/road-segmentation/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

End-to-end road extraction system that trains a segmentation model on satellite images and serves predictions via an inference API with ONNX optimization.

| | |
|---|---|
| **Training Report** | [W&B Report — training curves & metrics](https://api.wandb.ai/links/minaessam/ufwt8pum) |
| **All Experiments** | [wandb.ai/minaessam/road-segmentation](https://wandb.ai/minaessam/road-segmentation) |
| **Model Artifacts** | [W&B Model Registry](https://api.wandb.ai/links/minaessam/6cmg33dd) |
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
- [CI/CD Pipeline](#cicd-pipeline)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)
- [Write-Up](WRITEUP.md) — results, design decisions, and production considerations

---

## Quick Start with Docker

### Option A: Single command (model downloads automatically)

```bash
git clone https://github.com/minaessam2015/road-segmentation.git
cd road-segmentation
docker-compose up --build
```

The entrypoint downloads the model from Google Drive on first start (no account or API key needed). The API will be available at `http://localhost:8000`.

### Option B: Pre-downloaded model

```bash
git clone https://github.com/minaessam2015/road-segmentation.git
cd road-segmentation

# Download model (21 MB)
mkdir -p models
curl -L "https://drive.google.com/uc?export=download&id=1vliNHcQ2fQ13Tk9VRDRPEllmwdoVzN6X" \
  -o models/UnetPlusPlus_efficientnet-b4_int8.onnx

docker-compose up --build
```

### Option C: Pull pre-built image

```bash
docker pull ghcr.io/minaessam2015/road-segmentation:latest

# Download model
mkdir -p models
curl -L "https://drive.google.com/uc?export=download&id=1vliNHcQ2fQ13Tk9VRDRPEllmwdoVzN6X" \
  -o models/UnetPlusPlus_efficientnet-b4_int8.onnx

docker run -p 8000:8000 -v ./models:/app/models \
  ghcr.io/minaessam2015/road-segmentation:latest
```

### Test the running API

A sample satellite image is included in the repo for quick testing:

```bash
# Health check
curl http://localhost:8000/health

# Segment the sample image (returns JSON with mask_url)
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@tests/fixtures/sample_satellite.jpg" | python -m json.tool

# Download the predicted mask image using mask_url from the response
MASK_URL=$(curl -s -X POST http://localhost:8000/api/v1/segment \
  -F "image=@tests/fixtures/sample_satellite.jpg" | python -c "import json,sys; print(json.load(sys.stdin)['mask_url'])")
curl -o predicted_mask.png "http://localhost:8000${MASK_URL}"
open predicted_mask.png  # macOS — or xdg-open on Linux

# With GeoJSON road polylines
curl -X POST "http://localhost:8000/api/v1/segment?return_geojson=true" \
  -F "image=@tests/fixtures/sample_satellite.jpg" | python -m json.tool

# Prometheus metrics
curl http://localhost:8000/metrics
```

The `mask_url` is served by the same running API. The mask PNG is accessible as long as the container is running. See the [API Reference](#post-apiv1segment) section for full details on the response format and mask retrieval.

---

## Manual Setup

### Prerequisites

- Python 3.12
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

**Easiest — direct download from Google Drive (no account needed):**

```bash
mkdir -p models

# INT8 model (21 MB, CPU optimized):
curl -L "https://drive.google.com/uc?export=download&id=1vliNHcQ2fQ13Tk9VRDRPEllmwdoVzN6X" \
  -o models/UnetPlusPlus_efficientnet-b4_int8.onnx

# FP16 model (40 MB, GPU optimized):
curl -L "https://drive.google.com/uc?export=download&id=1nLfBITa-yjFZjdf_3XtEQjETn2VWG5lZ" \
  -o models/UnetPlusPlus_efficientnet-b4_fp16.onnx
```

**Alternative — from W&B Model Registry** (requires free [API key](https://wandb.ai/authorize)):

```bash
python scripts/download_model.py --wandb-key YOUR_KEY --onnx-only
```

Models are also browsable in the [W&B Model Registry report](https://api.wandb.ai/links/minaessam/6cmg33dd).

### Run the API

```bash
# CPU serving (INT8 quantized, 21 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_int8.onnx --device cpu

# GPU serving (FP16, 40 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_fp16.onnx --device gpu

# From PyTorch checkpoint:
python scripts/serve.py --checkpoint checkpoints/best.pth
```

### Test with validation images

```bash
# Start the server in one terminal, then:
python scripts/test_api_client.py --val-set --n 10 --geojson
```

Sends validation images (with ground truth) to the API and saves: prediction overlays, error maps (TP=green, FP=red, FN=blue), side-by-side comparisons, and per-sample IoU scores to `test_results/`.

---

## Training

### Quick Experiment (local, CPU)

```bash
python scripts/train.py --config configs/unet_resnet34.yaml \
    --override training.epochs=5 data.subset_size=200 data.num_workers=0
```

### Full Training (Colab with GPU)

Open the training notebook on Colab (see [Notebooks](#notebooks) section for the link), set your credentials, and run all cells. Checkpoints save to Google Drive automatically. On disconnect, re-run to resume from the latest checkpoint.

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

### Model Export

```bash
python scripts/optimize.py --checkpoint checkpoints/best.pth --output models/ --targets cpu gpu
```

Exports ONNX FP32, FP16 (GPU), and INT8 (CPU) variants.

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
- `return_geojson` (bool, optional): Include road centerline polylines

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

**Retrieving the mask image:** The `mask_url` field is a path served by the same running API instance. To download the binary mask PNG:

```bash
# Step 1: Send image, get response
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/segment -F "image=@satellite.jpg")
echo $RESPONSE | python -m json.tool

# Step 2: Download the mask image using mask_url from the response
MASK_URL=$(echo $RESPONSE | python -c "import json,sys; print(json.load(sys.stdin)['mask_url'])")
curl -o road_mask.png "http://localhost:8000${MASK_URL}"
```

The mask is stored in the container's `/app/results/` directory and served as a static file. It remains accessible as long as the container is running.

**Known limitation:** The current implementation saves every mask to disk without cleanup. In a production deployment, this would require a retention strategy — for example, a TTL-based cleanup job that deletes results older than N minutes, or replacing file storage with a pre-signed URL to object storage (S3/GCS) where lifecycle policies handle expiration, or returning the mask as base64 directly in the JSON response to avoid server-side storage entirely. I did not implement this due to time constraints but would address it before any production deployment.

To persist masks on the host filesystem, mount a volume:

```bash
docker run -p 8000:8000 -v ./models:/app/models -v ./results:/app/results ...
```

With `docker-compose`, the volume is already mapped — masks appear in `./results/` on the host.

**GeoJSON output:** The road centerline polylines are extracted via morphological cleanup, skeletonization to 1-pixel centerlines, graph extraction, branch pruning, and Douglas-Peucker line simplification. See [docs/postprocessing_review.md](docs/postprocessing_review.md) for the full pipeline rationale.

### Observability

- **Structured JSON logging** — every request logged with timestamp, request ID, duration, status code
- **Request tracing** — `X-Request-ID` header generated per request, propagated through all stages
- **Inference audit trail** — image hash, prediction stats, model version logged for each prediction
- **Prometheus metrics** — request counts, latency p50/p95/p99, error rates at `/metrics`
- **Health check** — model status, disk space, GPU memory at `/health`

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

**69 tests** covering: config loading, data transforms, model forward pass, all 7 loss functions, boundary weight computation, post-processing pipeline, API endpoints, and end-to-end inference with real model weights.

---

## CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/ci.yml`) with two stages that run on every push to `main`:

**Stage 1 — Code Quality** (runs on every push):
1. **Lint** (ruff) — enforces consistent code style across `src/`, `tests/`, `scripts/`
2. **Unit tests** (pytest) — runs the 61 unit tests (e2e tests are skipped in CI since they require model weights)
3. **Security scan** (pip-audit) — checks all dependencies for known vulnerabilities

**Stage 2 — Docker & Deployment** (runs on push to main, after Stage 1 passes):
1. **Build Docker image** — verifies the multi-stage Dockerfile builds correctly
2. **Import verification** — runs all key imports inside the container to catch missing dependencies
3. **E2E test with real model** — downloads the ONNX model from W&B artifacts inside the container, runs inference on a test image, and verifies the output shape and values
4. **Push to GHCR** — tags and pushes the verified image to `ghcr.io/minaessam2015/road-segmentation:latest` so the team can pull and run it directly

The pipeline requires a `WANDB_API_KEY` secret in the repository settings (Settings > Secrets > Actions) for the model download step.

---

## Notebooks

All notebooks run on Google Colab with GPU. Click the badge to open directly.

| # | Notebook | Colab | Description |
|---|----------|-------|-------------|
| 01 | [EDA](notebooks/01_eda.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/01_eda.ipynb) | Data exploration: class imbalance (roads ~4% of pixels), road width distribution, spatial patterns, mask quality. Every downstream decision is justified here. |
| 02 | [Model Comparison](notebooks/02_model_comparison.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/02_model_comparison.ipynb) | Benchmark of 4 architectures on 1000 samples / 15 epochs. U-Net+ResNet34 won, leading to UNet++ + EfficientNet-B4 as the final choice. |
| 03 | [Training](notebooks/03_training.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/03_training.ipynb) | Full training with W&B tracking and Drive checkpoints. Auto-resumes on Colab disconnect. |
| 04 | [Data Leakage](notebooks/04_data_leakage_check.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/04_data_leakage_check.ipynb) | Train/val similarity analysis using ResNet50 + CLIP embeddings. Finds 123 potentially leaky samples, quantifies 1.9% IoU inflation. |
| 05 | [Error Analysis](notebooks/05_error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/05_error_analysis.ipynb) | Per-sample failure patterns, FP/FN balance, performance by road width, calibration (reliability diagram, ECE, temperature scaling). |
| 06 | [Post-Processing](notebooks/06_postprocessing_ablation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/06_postprocessing_ablation.ipynb) | Ablation study measuring each step's IoU contribution. See [docs/postprocessing_review.md](docs/postprocessing_review.md) for the literature-backed rationale. |
| 07 | [Optimization](notebooks/07_model_optimization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/road-segmentation/blob/main/notebooks/07_model_optimization.ipynb) | ONNX export, FP16/INT8 compression, speed + accuracy benchmark on T4 GPU. |

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
    configs/                        #   9 YAML architecture configs
    scripts/                        #   CLI tools (train, serve, optimize, download)
    notebooks/                      #   7 analysis notebooks (see above)
    tests/                          #   69 tests (unit + e2e)
        fixtures/                   #   Sample satellite image for testing
    docs/                           #   Literature review, postprocessing review, images
    .github/workflows/ci.yml       #   CI: lint, test, Docker, GHCR push
    Dockerfile                      #   Multi-stage inference image
    docker-compose.yml              #   Single-command deployment
```

---

## Write-Up

For the detailed write-up covering results, design decisions, and production considerations, see **[WRITEUP.md](WRITEUP.md)**.

---

## License

This project uses the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) which must be downloaded separately via the Kaggle API. The dataset is not included in this repository.
