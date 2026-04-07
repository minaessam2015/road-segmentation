# Road Segmentation from Satellite Imagery

End-to-end road extraction system: train a segmentation model on satellite images and serve predictions via an inference API. Built for the Hudhud ML Engineer take-home assignment.

**Live experiment tracking**: [wandb.ai/minaessam/road-segmentation](https://wandb.ai/minaessam/road-segmentation)

## Results

| Metric | Value |
|--------|-------|
| **Best Val IoU** | 0.7016 |
| **Best Val Dice** | 0.8246 |
| Architecture | UNet++ + EfficientNet-B4 |
| Input resolution | 1024x1024 |
| Training data | 5,292 samples (85% of DeepGlobe train) |
| Inference latency (ONNX FP16, T4 GPU) | 78 ms |

### Model Comparison (preliminary, 1000 samples, 15 epochs)

| Model | IoU | Dice | Params | Time |
|-------|-----|------|--------|------|
| **U-Net + ResNet34** | 0.528 | 0.691 | 24.4M | 10 min |
| LinkNet + ResNet34 | 0.507 | 0.673 | 21.8M | 9 min |
| SegFormer + MIT-B2 | 0.485 | 0.653 | 24.7M | 15 min |
| DeepLabV3+ + ResNet50 | 0.484 | — | 26.7M | 10 min |

### Optimization Benchmark (T4 GPU)

| Variant | Latency | Speedup | IoU | Size |
|---------|---------|---------|-----|------|
| PyTorch FP32 | 145.6 ms | 1.00x | 0.6835 | — |
| PyTorch FP16/AMP | 95.5 ms | 1.52x | 0.6835 | — |
| ONNX FP32 | 135.7 ms | 1.07x | 0.6835 | 80 MB |
| **ONNX FP16** | **78.3 ms** | **1.86x** | **0.6835** | **40 MB** |

Zero IoU loss across all compression variants.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/minaessam2015/road-segmentation.git
cd road-segmentation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Download Data

Get a Kaggle API token from [kaggle.com/settings](https://www.kaggle.com/settings), then:

```bash
cp .env.example .env
# Edit .env with your KAGGLE_USERNAME and KAGGLE_KEY
python scripts/download_data.py
```

### 3. Download Model Weights

Models are stored in the [W&B Model Registry](https://wandb.ai/minaessam/road-segmentation):

```bash
# Add your WANDB_API_KEY to .env, then:
python scripts/download_model.py              # all models (checkpoint + ONNX)
python scripts/download_model.py --onnx-only  # just ONNX for deployment
```

### 4. Run the API

```bash
# CPU (INT8 quantized — 21 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_int8.onnx --device cpu

# GPU (FP16 — 40 MB):
python scripts/serve.py --onnx models/UnetPlusPlus_efficientnet-b4_fp16.onnx --device gpu
```

Test it:

```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@satellite.jpg" | python -m json.tool
```

### 5. Docker

```bash
docker-compose up --build
# Or pull the pre-built image:
docker pull ghcr.io/minaessam2015/road-segmentation:latest
```

## Project Structure

```
road-segmentation/
    src/road_segmentation/
        api/                    # FastAPI inference service
            app.py              # Endpoints: /segment, /health, /metrics
            inference.py        # PyTorch inference engine
            optimize.py         # ONNX export, INT8/FP16 optimization
            observability.py    # Structured logging, request tracing, Prometheus
        data/                   # Dataset loading and preprocessing
            dataset.py          # PyTorch Dataset + DataLoader
            transforms.py       # Albumentations augmentation pipelines
            split.py            # Stratified train/val split
            eda.py              # Exploratory data analysis utilities
        models/
            factory.py          # SMP model creation, encoder freeze/unfreeze
        training/
            trainer.py          # Training loop (AMP, checkpointing, W&B)
            losses.py           # BCE+Dice, Focal, Tversky, boundary-weighted
            metrics.py          # IoU, Dice, Precision, Recall
            callbacks.py        # Early stopping, Model EMA
            checkpoint.py       # Save/load with resume support
            visualization.py    # Training curves, prediction overlays
        postprocessing/
            steps.py            # Threshold, morphology, skeletonize, gap bridging
            pipeline.py         # Composable pipeline with toggleable steps
        config.py               # YAML config system with CLI overrides
        env.py                  # .env loading for credentials
        paths.py                # Project path constants
    configs/                    # YAML configs for each architecture
    scripts/
        train.py                # CLI: python scripts/train.py --config configs/...
        serve.py                # CLI: python scripts/serve.py --onnx models/...
        optimize.py             # CLI: export and compress models to ONNX
        download_model.py       # Download from W&B model registry
        upload_model.py         # Upload to W&B model registry
        upload_to_wandb.py      # Post-hoc upload of training metrics to W&B
        test_api_client.py      # API test client with GT comparison
    notebooks/
        01_eda.ipynb            # Data exploration and EDA-driven decisions
        02_model_comparison.ipynb  # Quick 4-model architecture comparison
        03_training.ipynb       # Full training (Colab, W&B, Drive checkpoints)
        04_data_leakage_check.ipynb  # Train/val similarity analysis
        05_error_analysis.ipynb # Failure patterns, calibration, spatial errors
        06_postprocessing_ablation.ipynb  # Per-step IoU contribution
        07_model_optimization.ipynb  # ONNX export + speed/accuracy benchmark
    tests/                      # 69 tests (unit + end-to-end)
    .github/workflows/ci.yml   # Lint, test, Docker build, GHCR push
    Dockerfile                  # Multi-stage build for inference
    docker-compose.yml          # Single-command deployment
```

## Training

### Quick Experiment (local, CPU)

```bash
python scripts/train.py --config configs/unet_resnet34.yaml \
    --override training.epochs=5 data.subset_size=200 data.num_workers=0
```

### Full Training (Colab)

Open `notebooks/03_training.ipynb` on [Google Colab](https://colab.research.google.com):
1. Set runtime to GPU (T4 or A100)
2. Set your credentials (Kaggle + W&B)
3. Run all cells

Checkpoints save to Google Drive automatically. On disconnect, re-run all cells to resume from the latest checkpoint.

### Config System

Each architecture has a YAML config in `configs/`. Override any parameter from the CLI:

```bash
python scripts/train.py --config configs/unet_resnet34.yaml \
    --override training.epochs=30 optimizer.lr=0.0005 data.batch_size=16
```

Available configs:
- `unet_resnet34.yaml` — baseline (closest to D-LinkNet winner)
- `unet_efficientnet_b4.yaml` — stronger encoder
- `unetplusplus_efficientnet_b4_1024.yaml` — best model (full resolution)
- `deeplabv3plus_resnet50.yaml` — multi-scale context
- `linknet_resnet34.yaml` — fastest inference
- `segformer_mitb2.yaml` — transformer architecture
- `finetune_boundary_loss.yaml` — boundary-weighted loss fine-tuning

## API Reference

### `POST /api/v1/segment`

Upload a satellite image, get a road mask.

```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@satellite.jpg" \
  -G -d "threshold=0.5" -d "return_geojson=true"
```

Response:
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
  "geojson": { "type": "FeatureCollection", "features": [...] }
}
```

### `GET /health`

Returns model status, disk space, GPU memory.

### `GET /api/v1/model-info`

Returns model backend, version, and device.

### `GET /metrics`

Prometheus-compatible metrics: request counts, latency percentiles, error rates.

## Testing

```bash
# Unit tests (no model weights needed):
pytest tests/ --ignore=tests/test_e2e.py

# Full suite including end-to-end (needs model weights):
pytest tests/

# With specific model paths:
pytest tests/test_e2e.py --checkpoint checkpoints/best.pth --onnx models/model_int8.onnx
```

69 tests covering: config loading, data transforms, model forward pass, all 7 loss functions, boundary weight computation, post-processing pipeline, API endpoints, and end-to-end inference with real model weights.

## Design Decisions and Trade-offs

### Architecture Choice

**UNet++ with EfficientNet-B4 encoder** was selected based on a systematic comparison of 4 architectures on the DeepGlobe dataset. U-Net's skip connections preserve fine spatial detail critical for thin roads. EfficientNet-B4's compound scaling (depth + width + resolution) provides richer features than ResNet at similar parameter count. UNet++ adds dense nested skip connections that improve thin road segmentation over vanilla U-Net.

We chose not to use the original D-LinkNet despite it winning the DeepGlobe challenge because: (1) the code requires Python 2.7 / PyTorch 0.2, (2) pretrained weights are hosted on untrusted personal file-sharing links, and (3) SMP's LinkNet + ResNet34 approximates the approach with production-quality code.

### Loss Function

**BCE + Dice compound loss** handles the severe class imbalance (roads are ~4% of pixels). BCE provides stable per-pixel gradients while Dice directly optimizes the overlap metric. We also implemented boundary-weighted loss (5x weight on edge pixels) for fine-tuning, though the ablation showed minimal additional gain on our well-trained model.

### Evaluation Metrics

**IoU (Jaccard index)** as the primary metric because it penalizes both false positives and false negatives equally. Pixel accuracy is useless for road segmentation — a model predicting all-background achieves 96% accuracy. We also report Dice, Precision, and Recall. For production monitoring, we track calibration (reliability diagrams, ECE) to ensure confidence scores are trustworthy.

### Data Integrity

We identified potential spatial data leakage in the random train/val split using both ResNet50 and CLIP embedding similarity analysis. Combining both methods (ResNet >= 0.93 OR CLIP >= 0.96) flagged 123 potentially leaky validation samples. After removing them, the IoU dropped from 0.694 to 0.675 — a 1.9% inflation. We report both metrics for transparency.

### Post-Processing

The ablation study showed minimal post-processing gains on our model: threshold optimization (+0.01% IoU), component filtering (+0.02%), morphological operations (-0.04%). This indicates the model produces clean predictions that don't need heavy post-processing. TTA provides +0.65% IoU on harder samples. For production, we use the raw model output with a tuned threshold.

## Production Considerations

### Scalability

The inference service is stateless — each request loads the image, runs inference, and returns results independently. To handle thousands of tiles per hour:
- **Horizontal scaling**: Deploy multiple replicas behind a load balancer (Kubernetes, ECS). Each replica loads the ONNX model independently.
- **Task queue**: For batch processing, use Celery/Redis or AWS SQS to decouple submission from processing.
- **Tiling**: Large satellite images are tiled with overlap (128-256px), processed in parallel, and stitched with central-crop blending to avoid boundary artifacts.

### Latency

ONNX FP16 achieves 78ms per 1024x1024 image on a T4 GPU (1.86x speedup over PyTorch FP32, zero IoU loss). Further optimizations:
- **TensorRT**: Potential additional 2-3x speedup with layer fusion and kernel auto-tuning.
- **Batching**: Process multiple tiles simultaneously to saturate GPU throughput.
- **INT8 quantization**: 21 MB model for CPU deployment where GPU is unavailable.
- **Model distillation**: Train a smaller student model from the current teacher for edge deployment.

### Model Lifecycle

- **Versioning**: Models are stored in the W&B Model Registry with version tags, training config, and metrics.
- **Download**: `python scripts/download_model.py --version v1` fetches a specific version.
- **A/B testing**: Route a percentage of traffic to a new model version, compare IoU on held-out samples.
- **Rollback**: Previous versions remain in the registry. Swap by changing the version tag in the deployment config.
- **CI/CD**: GitHub Actions builds and pushes Docker images to GHCR on every merge to main. The image is available at `ghcr.io/minaessam2015/road-segmentation:latest`.

### Data Drift

Satellite imagery varies by provider, resolution, season, and geography. Detection strategy:
- **Input monitoring**: Track per-batch RGB channel means and standard deviations. Alert when KL divergence from training distribution exceeds a threshold.
- **Prediction monitoring**: Track road_coverage_pct and confidence_mean distributions over time. A sudden shift indicates either data drift or model degradation.
- **Calibration monitoring**: Periodically recompute the reliability diagram on new labeled samples. ECE increasing over time signals miscalibration.
- **Retraining trigger**: When drift is detected, retrain on a mix of original and new data. The config system and W&B tracking make this reproducible.

### Observability

The API provides production-grade observability:
- **Structured JSON logging**: Every request logged with timestamp, request ID, duration, status.
- **Request tracing**: Unique `X-Request-ID` header propagated through all stages.
- **Inference audit trail**: Image hash, prediction stats, model version logged per request.
- **Prometheus metrics**: `GET /metrics` endpoint with request counts, latency p50/p95/p99, error rates.
- **Health check**: `GET /health` returns model status, disk space, GPU memory.

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | EDA | Data exploration, class imbalance analysis, EDA-driven decisions |
| 02 | Model Comparison | Quick 4-architecture comparison (1000 samples, 15 epochs) |
| 03 | Training | Full training with W&B tracking and Drive checkpoints |
| 04 | Data Leakage | Train/val similarity analysis (ResNet + CLIP embeddings) |
| 05 | Error Analysis | Failure patterns, calibration, spatial errors, temperature scaling |
| 06 | Post-Processing | Per-step ablation study of threshold, morphology, TTA, gap bridging |
| 07 | Model Optimization | ONNX export, FP16/INT8 compression, speed + accuracy benchmark |

### Dataset Versioning

The DeepGlobe dataset is fixed (single version on Kaggle), so formal dataset versioning (DVC, W&B dataset artifacts) is not implemented. However, reproducibility is ensured through:
- Deterministic train/val split (seed=42, ratio=0.15) recorded in every checkpoint's config
- The download script always fetches the same dataset version from Kaggle
- In production, dataset changes would be tracked via content hashing and W&B artifacts, with retraining triggered when the hash changes

### Docker Image and Model Weights

The Docker image does not bake in model weights — they are mounted at runtime via volume:
```bash
docker run -p 8000:8000 -v ./models:/app/models ghcr.io/minaessam2015/road-segmentation:latest
```

This is the production pattern because:
- Image stays small (~2 GB vs ~2.3 GB with model)
- Same image works with different model versions (swap the mounted file)
- Model updates don't require rebuilding or re-pushing the image
- CI tests download from W&B inside the container for E2E verification

## License

This project uses the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) which must be downloaded separately via the Kaggle API. The dataset is not included in this repository.
