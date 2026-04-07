# Road Segmentation from Satellite Imagery

[![CI](https://github.com/minaessam2015/road-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/minaessam2015/road-segmentation/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

End-to-end road extraction system that trains a segmentation model on satellite images and serves predictions via an inference API with ONNX optimization.

| | |
|---|---|
| **Training Report** | [W&B Report — training curves & metrics](https://api.wandb.ai/links/minaessam/ufwt8pum) |
| **All Experiments** | [wandb.ai/minaessam/road-segmentation](https://wandb.ai/minaessam/road-segmentation) |
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

docker run -p 8000:8000 \
  -v ./models:/app/models \
  -e WANDB_API_KEY=your_key \
  ghcr.io/minaessam2015/road-segmentation:latest
```

### Test the running API

A sample satellite image is included in the repo for quick testing:

```bash
# Health check
curl http://localhost:8000/health

# Segment the sample image
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@tests/fixtures/sample_satellite.jpg" | python -m json.tool

# With GeoJSON road polylines
curl -X POST "http://localhost:8000/api/v1/segment?return_geojson=true" \
  -F "image=@tests/fixtures/sample_satellite.jpg" | python -m json.tool

# Prometheus metrics
curl http://localhost:8000/metrics
```

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

Models are stored in the [W&B Model Registry](https://wandb.ai/minaessam/road-segmentation):

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

## Results

### Step 1: Architecture Selection

I first compared 4 architectures on a small subset (1000 samples, 15 epochs) to select the best family before investing in full training. All models used ImageNet-pretrained encoders and identical augmentations for a fair comparison.

| Rank | Model | IoU | Dice | Params | Time |
|------|-------|-----|------|--------|------|
| 1 | **U-Net + ResNet34** | **0.528** | 0.691 | 24.4M | 10 min |
| 2 | LinkNet + ResNet34 | 0.507 | 0.673 | 21.8M | 9 min |
| 3 | SegFormer + MIT-B2 | 0.485 | 0.653 | 24.7M | 15 min |
| 4 | DeepLabV3+ + ResNet50 | 0.484 | — | 26.7M | 10 min |

U-Net won convincingly. The full details are in [notebook 02](notebooks/02_model_comparison.ipynb).

### Step 2: Scale Up to Best Model

Based on the comparison, I selected the U-Net family and scaled up in three ways:
- **Better decoder**: UNet++ (dense nested skip connections) instead of vanilla U-Net, which better preserves thin road features
- **Stronger encoder**: EfficientNet-B4 (NAS-optimized compound scaling) instead of ResNet34
- **Full resolution**: 1024x1024 instead of 512x512, so thin roads (~10px) aren't lost to downsampling

I also experimented with boundary-weighted loss (5x weight on road edge pixels) as a fine-tuning step, but the model was already well-calibrated and the gain was minimal.

### Step 3: Final Model

| Metric | Value |
|--------|-------|
| Architecture | UNet++ + EfficientNet-B4 |
| Input | 1024x1024 (full resolution) |
| **Val IoU** | **0.7016** |
| **Val Dice** | **0.8246** |
| Training | 50 epochs, 5292 train / 934 val |

Full training logs and curves are in the [W&B training report](https://api.wandb.ai/links/minaessam/ufwt8pum).

#### Sample Predictions

Input image | prediction overlay (red = road) | error map (green = TP, red = FP, blue = FN):

![Sample prediction 307908](docs/images/sample_307908.jpg)

![Sample prediction 529643](docs/images/sample_529643.jpg)

### Step 4: Data Leakage Analysis

DeepGlobe tiles are cut from larger satellite images, so adjacent tiles in train and val sets may share geographic context (same road continuing across tile boundaries). I analyzed this using two embedding methods:

- **ResNet50 features** — catches visual near-duplicates (same pixels). Threshold: **0.93** cosine similarity.
- **CLIP ViT-B/32** — catches semantic similarity (same area, different appearance). Threshold: **0.96** cosine similarity.

I applied both thresholds to the top-500 highest-IoU samples from each set, then took the union of flagged samples:

| | Count | Mean IoU |
|---|---|---|
| **Total val samples** | 934 | 0.694 |
| **Flagged as leaky** | 123 (13%) | 0.829 |
| **Clean (after removal)** | 811 | **0.675** |
| **IoU inflation** | | **1.9%** |

The leaky samples have significantly higher IoU (0.829 vs 0.675) because the model effectively saw their geographic context during training. The 1.9% inflation is minor, but I report both metrics for transparency.

#### Most Similar Val-Train Pairs (ResNet embeddings)

Each row shows a validation image (left) and its most similar training image (right), with cosine similarity score:

![Leakage pairs](docs/images/leakage_pairs.jpg)

Full analysis with more visualizations in [notebook 04](notebooks/04_data_leakage_check.ipynb).

### Step 5: Post-Processing Ablation

I tested each post-processing step independently to measure its contribution. Each step is documented in detail in [docs/postprocessing_review.md](docs/postprocessing_review.md).

| Step | IoU | Delta | Verdict |
|------|-----|-------|---------|
| Raw threshold (0.5) | 0.7016 | baseline | — |
| [Optimal threshold](docs/postprocessing_review.md#2-probability-to-mask-conversion) (0.45) | 0.7017 | +0.01% | Neutral |
| [+ Component filter](docs/postprocessing_review.md#4-connected-component-and-segment-filtering) | 0.7019 | +0.02% | Neutral |
| [+ Morphological closing](docs/postprocessing_review.md#3-morphological-cleanup) | 0.7012 | -0.04% | Skip |
| [+ Morphological opening](docs/postprocessing_review.md#3-morphological-cleanup) | 0.7012 | +0.00% | Neutral |
| [+ Gap bridging](docs/postprocessing_review.md#7-connectivity-repair-via-graph-reasoning) | 0.7012 | +0.00% | Neutral |
| [TTA (6-fold)](docs/postprocessing_review.md#10-test-time-augmentation-as-prediction-refinement) | 0.7014 | +0.65%* | Include |

*TTA gain measured on a 100-sample subset where the baseline IoU was lower (0.695), so the absolute gain is real but specific to harder samples.

**Conclusion**: the model produces clean predictions at threshold 0.5. Post-processing adds marginal value. I include TTA as an optional flag in the API for offline batch processing where latency is less critical.

### Step 6: Model Optimization

I exported the trained model to ONNX and tested FP16 and INT8 compression on a T4 GPU:

| Variant | Latency | Speedup | IoU | Size |
|---------|---------|---------|-----|------|
| PyTorch FP32 | 145.6 ms | 1.00x | 0.6835 | — |
| PyTorch FP16/AMP | 95.5 ms | 1.52x | 0.6835 | — |
| ONNX FP32 | 135.7 ms | 1.07x | 0.6835 | 80 MB |
| **ONNX FP16** | **78.3 ms** | **1.86x** | **0.6835** | **40 MB** |
| ONNX INT8 (CPU) | — | — | — | 21 MB |

**Zero IoU loss** across all variants. FP16 weights stay in float16 during GPU inference (using Tensor Cores), which explains the 1.86x speedup. I verified this by running all variants on 200 validation samples and comparing outputs. Full benchmark in [notebook 07](notebooks/07_model_optimization.ipynb).

---

## Design Decisions

### Why UNet++ + EfficientNet-B4?

**Architecture selection was data-driven, not assumed.** I ran a preliminary comparison of 4 architectures (U-Net, LinkNet, SegFormer, DeepLabV3+) on 1000 samples to identify the best family, then scaled up the winner.

**U-Net family won because** skip connections concatenate encoder features directly to the decoder, preserving fine spatial detail. Roads are thin structures (some as narrow as 10 pixels at 1024x1024) where this detail matters. UNet++ extends this with dense nested skip connections that capture features at multiple semantic scales.

**EfficientNet-B4 over ResNet34 because** EfficientNet uses compound scaling (depth + width + resolution simultaneously, discovered via neural architecture search) which extracts richer features at similar parameter count (19M vs 22M). The "Battle of the Backbones" study (NeurIPS 2023) found EfficientNet to be the best overall backbone for efficiency-accuracy tradeoff.

**I chose not to use D-LinkNet** (the original DeepGlobe challenge winner) because: (1) its code requires Python 2.7 and PyTorch 0.2, (2) the pretrained weights are hosted on personal Dropbox/Baidu links with no integrity verification, and (3) SMP's LinkNet + ResNet34 provides a production-quality approximation that I tested in the comparison.

**Full resolution (1024x1024) instead of 512x512** because training at native resolution gave a **5% IoU boost, 3% recall improvement, and 2.5% precision improvement** over 512x512 — a significant gain that justifies the slower training. The EDA confirmed why: road widths are as narrow as 10px at 1024, which halve to 5px at 512 where they become unreliable to segment.

### Why BCE + Dice Loss?

The EDA revealed that roads cover only **4.3% of pixels on average** (median 2.15%). This severe class imbalance means:

- **Plain BCE fails** — gradients are dominated by the 96% background pixels, so the model learns to predict "no road" everywhere.
- **Pixel accuracy is useless** — a model predicting all-background achieves 96% accuracy.
- **BCE + Dice compound loss** combines the best of both: BCE provides stable per-pixel gradients for learning, while Dice directly optimizes the overlap metric and is inherently imbalance-resistant (it only considers the ratio of overlap to total predicted+actual, not the background).

I also implemented and tested **boundary-weighted BCE+Dice** (5x weight on road edge pixels) for fine-tuning. The literature reports +2-5 IoU points from this technique on road extraction. In my case, the gain was minimal because the base model was already well-calibrated — the post-processing ablation confirmed this.

### Why IoU as the Primary Metric?

**IoU (Intersection over Union)** penalizes both false positives and false negatives equally, making it the standard for segmentation tasks with class imbalance. I also report:

- **Dice** — monotonically related to IoU, easier to interpret (harmonic mean of precision and recall)
- **Precision / Recall** — to understand whether the model over-predicts (high FP) or under-predicts (high FN)
- **Calibration (ECE)** — to verify that confidence scores are trustworthy, which matters for the gap-bridging post-processing and the `confidence_mean` field in the API response

I intentionally **do not report pixel accuracy** because it's meaningless at 96% background.

### How I Split the Data

Only the train set (6,226 samples) has ground truth masks — the valid and test sets have no masks. I split train into **85% train (5,292) / 15% val (934)** using stratified sampling by road coverage percentage. This ensures both splits have similar distributions of sparse and dense road images.

The stratification uses `sklearn.train_test_split` with `stratify=coverage_bins` (quantile-based bins). The split seed (42) is separate from the training seed, so changing training randomness doesn't change which samples are in validation — this makes model comparison fair.

I then verified split integrity via the data leakage analysis (see Results section).

---

## Production Considerations

### Scalability

The inference service is **stateless** — no session state between requests. This means scaling horizontally is straightforward:

- **Multiple replicas** behind a load balancer (Kubernetes Deployment, AWS ECS). Each replica loads the ONNX model into memory independently. No shared state needed.
- **Batch processing**: For processing thousands of tiles, decouple submission from inference using a task queue (Celery + Redis, or AWS SQS). Workers pull tiles from the queue and write results to object storage.
- **Large image tiling**: Satellite images larger than 1024x1024 would be tiled with 128-256px overlap, each tile processed independently, then stitched back using only the central region of each prediction (discarding the "halo" border to avoid boundary artifacts).

I have not implemented a task queue or tiling service — the API handles single-image requests only. These would be the first additions for production scale.

### Latency

Current latency on a T4 GPU is **78ms per 1024x1024 image** (ONNX FP16). For stricter budgets:

- **TensorRT** — layer fusion and kernel auto-tuning on top of ONNX could provide an additional 2-3x speedup. I have not implemented this because it requires NVIDIA-specific tooling and a GPU build environment, but the ONNX model is TensorRT-compatible.
- **Dynamic batching** — when serving multiple concurrent requests, accumulate them into a batch (e.g., 8 images) and run a single forward pass. ONNX Runtime and Triton Inference Server both support this natively.
- **INT8 on CPU** — the 21 MB INT8 model enables inference on CPU-only instances (no GPU required). Latency is higher (~3 seconds at 1024x1024) but the model size and cost are minimal.

The ONNX export uses fixed input dimensions (1024x1024) rather than dynamic shapes because UNet++ has shape-dependent operations that cause issues with dynamic ONNX export. This is documented in `api/optimize.py`.

### Model Lifecycle

Models are versioned in the **W&B Model Registry** as artifacts:

```bash
# Upload a new version
python scripts/upload_model.py --checkpoint checkpoints/best.pth --onnx-dir models/ --version v2.0

# Download a specific version
python scripts/download_model.py --version v1

# Download latest
python scripts/download_model.py
```

Each artifact stores: the PyTorch checkpoint (for retraining), ONNX FP32/FP16/INT8 (for deployment), and the full training config (for reproducibility).

**For A/B testing**: deploy two model versions behind a traffic-splitting proxy, route a percentage to the new version, and compare IoU on a held-out labeled set. **For rollback**: point the deployment config back to the previous artifact version — the old model is still in the registry.

I have not implemented A/B testing or automated rollback — these would require infrastructure (traffic routing, automated metric comparison) beyond the scope of this project.

### Data Drift

Satellite imagery varies by provider, resolution, season, and geography. The inference audit logging (in `api/observability.py`) already captures per-request statistics that could feed a drift detection system:

```json
{"event": "inference", "image_hash": "3f8a9bc2", "road_coverage_pct": 4.2, "confidence_mean": 0.87, "inference_time_ms": 78.3}
```

**Detection strategy** (not implemented, but designed for):

1. **Input monitoring**: Compute per-image RGB channel means and standard deviations. Compare to the **training set distribution** computed during EDA (notebook 01 reports the actual dataset channel statistics, which differ from ImageNet normalization values). Use KL divergence between the running distribution and the training baseline. Alert when divergence exceeds a threshold (e.g., > 0.1) sustained over 100 consecutive images — a persistent shift suggests a new imagery source, season, or sensor.

2. **Prediction monitoring**: Track `road_coverage_pct` and `confidence_mean` from the inference audit log (already captured per request by `api/observability.py`). Maintain a 1000-request sliding window and compare its distribution to a baseline window from initial deployment using a Kolmogorov-Smirnov test. For example: if the training set has mean road coverage of 4.3%, and the sliding window suddenly shifts to 8% or 1%, that indicates either the model is seeing a different geographic region (legitimate shift) or the model is degrading (hallucinating roads or missing them). The distinction requires human review, but the alert surfaces it early.

3. **Calibration monitoring**: Periodically collect a small labeled sample from new production data and recompute the ECE (Expected Calibration Error) using the same method as notebook 05. If ECE increases above 0.05, the model's confidence scores are no longer reliable — this directly affects the confidence-weighted gap bridging in post-processing and the `confidence_mean` value reported to users.

4. **Retraining trigger**: When drift is confirmed, retrain on a mix of original and new data using the same config system and W&B tracking. The deterministic split and versioned configs ensure reproducibility.

### Dataset Versioning

The DeepGlobe dataset is fixed (single version on Kaggle), so I did not implement formal dataset versioning (DVC, W&B dataset artifacts). Reproducibility is ensured through:

- Deterministic train/val split (seed=42, ratio=0.15) recorded in every checkpoint config
- The download script always fetches the same Kaggle dataset version
- In production, dataset changes would be tracked via content hashing and W&B artifacts, with retraining triggered when the dataset hash changes

### Docker Image and Model Weights

The Docker image does **not** include model weights. They are either:
- Mounted at runtime via volume (`-v ./models:/app/models`)
- Auto-downloaded from W&B on first start (if `WANDB_API_KEY` is set)

This is the standard production pattern because the image stays small (~2 GB), the same image works with any model version, and model updates don't require rebuilding the image.

---

## License

This project uses the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) which must be downloaded separately via the Kaggle API. The dataset is not included in this repository.
