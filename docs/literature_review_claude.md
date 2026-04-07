# Literature Review: Road Segmentation from Satellite Imagery

## 1. Problem Overview

Road extraction from satellite imagery is a binary semantic segmentation task: given an RGB satellite image, predict a pixel-wise mask indicating road vs. background. It is foundational to map generation, navigation, disaster response, and urban planning.

The problem presents several challenges specific to remote sensing:
- **Severe class imbalance**: Roads typically occupy 2-5% of image pixels.
- **Thin, elongated structures**: Roads are long and narrow, making them sensitive to small prediction errors that can break connectivity.
- **Visual heterogeneity**: Roads vary in width, material, lighting, and context (urban vs. rural, paved vs. unpaved).
- **Occlusion**: Trees, shadows, and buildings can obscure road surfaces.
- **Topology matters**: A mask with high pixel IoU but a broken intersection is useless for routing.

---

## 2. Architectures

### 2.1 U-Net (Ronneberger et al., 2015)

The foundational encoder-decoder architecture for dense prediction. Skip connections concatenate encoder feature maps to the decoder, preserving fine spatial detail critical for thin road boundaries.

- **Strengths**: Excellent spatial detail preservation; simple and well-understood; strong baseline for binary segmentation.
- **Weaknesses**: Limited receptive field for capturing long-range road connectivity; no built-in multi-scale reasoning.
- **Relevance**: The default starting point. With a pretrained encoder (ResNet-34), U-Net remains competitive on DeepGlobe.

### 2.2 D-LinkNet (Zhou et al., CVPR 2018 Workshop) -- DeepGlobe Challenge Winner

Built on LinkNet with a pretrained ResNet-34 encoder and a **central dilated convolution block** that uses both cascade and parallel modes to enlarge the receptive field without losing spatial resolution.

- **Key innovation**: The dilated center block fuses features at multiple dilation rates (1, 2, 4, 8), capturing both local texture and global road structure. Accepts full 1024x1024 input.
- **Why it won**: Efficient decoder (LinkNet's addition-based skip connections are faster than U-Net's concatenation), strong multi-scale fusion, and pretrained encoder.
- **DeepGlobe IoU**: ~64% (2018 baseline).

*Reference*: Zhou et al., "D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction," CVPR 2018 Workshop.

### 2.3 DeepLabV3+ (Chen et al., 2018)

Uses Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context capture via parallel dilated convolutions at different rates, combined with a decoder module for boundary refinement.

- **Strengths**: Multi-scale context via ASPP; stable training convergence; strong boundary detection.
- **Weaknesses**: Heavier compute than U-Net/LinkNet; ASPP can be redundant for binary tasks.
- **Road-specific**: Improved DeepLabV3 variants have been proposed specifically for road segmentation in remote sensing (e.g., adding edge supervision).

### 2.4 HRNet (Sun et al., 2019)

Maintains **high-resolution representations throughout** the entire network by running parallel multi-resolution streams with repeated cross-resolution fusion. Unlike U-Net, it never fully downsamples to a low-resolution bottleneck.

- **Strengths**: Natively preserves spatial detail; excellent for fine-grained segmentation of thin structures.
- **Variants**: Lite-HRNet-OCR (lightweight), HRNetV2, HRFormer (transformer hybrid).
- **Relevance**: Strong choice when thin rural roads are critical.

### 2.5 SegFormer (Xie et al., NeurIPS 2021)

Hierarchical Vision Transformer encoder (Mix Vision Transformer, MIT-B0 through B5) with a lightweight MLP decoder.

- **Strengths**: Self-attention captures long-range dependencies critical for road connectivity; no positional encoding allows flexible input resolutions; produces finer lane markings than CNN approaches.
- **Weaknesses**: Requires more data; higher latency than lightweight CNNs.
- **Encoder sizes**: MIT-B0 (3.7M params, fast) to MIT-B5 (82M params, highest accuracy).

### 2.6 Emerging: PathMamba (2025) and Mamba-based Models

Hybrid Mamba-Transformer architectures designed specifically for topologically coherent road segmentation. PathMamba achieves 72.19% IoU on DeepGlobe with an APLS of 80.03%.

- **Key insight**: Mamba's linear-time sequence modeling handles the long, continuous nature of roads better than pure attention or pure convolution.

### 2.7 Architecture Comparison on DeepGlobe

| Model | Year | IoU (%) | Notes |
|-------|------|---------|-------|
| DS-Unet | 2026 | **79.25** | Dual-encoder with complementary attention (current reported SOTA) |
| PathMamba | 2025 | 72.19 | Hybrid Mamba-Transformer |
| SWGE-Net | 2024 | 71.57 | Strip convolution + wavelet features |
| DSFC-Net | 2026 | 68.68 | Dual-encoder spatial-frequency co-awareness |
| Seg-Road | 2023 | 67.20 | Transformer + CNN with connectivity |
| D-LinkNet | 2018 | ~64 | Original challenge winner |

**Practical benchmark**: IoU > 65% is competitive; > 70% is strong; > 75% is excellent.

### 2.8 Code & Weight Availability (Practical Reality)

Most published road segmentation models **do not release usable code or weights**. This table separates what's actually usable from paper-only claims:

| Model | Public Code? | Pretrained Weights? | Usable Today? |
|-------|-------------|---------------------|---------------|
| **SMP (U-Net, LinkNet, DeepLabV3+, FPN, SegFormer, etc.)** | Yes — `pip install segmentation-models-pytorch` | ImageNet encoder weights via torchvision/timm (trusted) | **Yes** — production-quality, actively maintained |
| **SegFormer (HuggingFace)** | Yes — `pip install transformers` | Official NVIDIA MIT-B0 to B5 on ImageNet, ADE20K, Cityscapes | **Yes** — verified org, needs adaptation for binary segmentation |
| **HRNet** | Yes — official repo + SMP/timm integration | ImageNet weights via timm (trusted) | **Yes** — use as `tu-hrnet_w18` encoder in SMP |
| D-LinkNet | Legacy repo (Python 2.7, PyTorch 0.2) | Dropbox/Baidu links (untrusted, unmaintained) | **No** — ancient dependencies, unverified binaries |
| DS-Unet | GitHub link in paper points to unrelated repo | None | **No** — no code exists |
| PathMamba | No public repo | None | **No** — paper only |
| SWGE-Net | No public repo | None | **No** — paper only |
| Seg-Road | Minimal repo, no docs, no weights | None | **No** — unusable |
| CoANet | Public repo | Google Drive links (unverified) | **Caution** — custom architecture, unverified binaries |

**Key takeaway**: The only **trusted, pip-installable** option is **SMP** with official ImageNet-pretrained encoders from torchvision/timm. All other "pretrained weights" for road segmentation are hosted on personal Dropbox/Baidu/Google Drive links with no provenance guarantees. We train our own decoder on DeepGlobe using trusted encoder weights.

**Our approach**: Use `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1)`. The encoder weights come from PyTorch's official torchvision model zoo. We train the full model (encoder fine-tuning + decoder from scratch) on DeepGlobe ourselves.

---

## 3. Loss Functions

Class imbalance (roads ~2-5% of pixels) makes loss function selection critical. A comparative study by Diakogiannis et al. (International Journal of Applied Earth Observation, 2022) evaluated loss functions specifically for road segmentation.

### 3.1 Standard Losses

| Loss | Formula | Road Segmentation Notes |
|------|---------|------------------------|
| **BCE** | $-[y \log p + (1-y)\log(1-p)]$ | Baseline. Does not account for imbalance; gradients dominated by easy background pixels. |
| **Weighted BCE** | BCE with class weights | Simple fix for imbalance, but requires tuning the weight ratio. |

### 3.2 Region-Based Losses

| Loss | Key Idea | Road Segmentation Notes |
|------|----------|------------------------|
| **Dice Loss** | $1 - \frac{2|P \cap G|}{|P| + |G|}$ | Directly optimizes the F1/Dice metric. Recommended as primary loss. Handles imbalance inherently. |
| **Tversky Loss** | Dice generalization with adjustable FP/FN weights ($\alpha$, $\beta$) | Best for U-Net (up to 6.36% DSC improvement). Set $\beta > 0.5$ to penalize missed roads more than false alarms. |
| **Focal Tversky** | Tversky + focal exponent $\gamma$ | Excellent for thin structures; focuses learning on hard-to-segment small road features. |
| **Lovasz-Softmax** | Direct IoU optimization via Lovasz extension | Well-suited for irregular shapes and unclear boundaries. Directly optimizes the evaluation metric. |

### 3.3 Compound Losses (Recommended)

Combining a **distribution-based loss** (BCE/Focal) with a **region-based loss** (Dice/Tversky) provides both stable gradients and imbalance robustness:

- **BCE + Dice**: Most common and robust choice. Typically $L = \lambda_1 \cdot L_{BCE} + \lambda_2 \cdot L_{Dice}$ with equal weights.
- **Focal + Dice**: Better when many easy-negative pixels dominate.
- **BCE + Lovasz**: When directly optimizing IoU is desired.

**Recommendation**: Start with **BCE + Dice** (equal weight). If thin roads are consistently missed, switch to **Focal Tversky** with $\beta = 0.7$.

---

## 4. Data Augmentation for Satellite Imagery

### 4.1 Geometric Augmentations (High Impact)

Satellite images have **no canonical orientation** (unlike street-level photos), so rotation-based augmentations are highly effective:

- **RandomRotate90**: 0/90/180/270 degree rotations. Essentially free data. Always use.
- **HorizontalFlip / VerticalFlip**: 2x and 4x data effectively. Always use.
- **RandomScale (0.5x-2.0x) + RandomCrop**: Critical for handling roads of varying widths. Roads appear at many scales.
- **ShiftScaleRotate**: Continuous rotation + translation. Use reflection padding to avoid black border artifacts.
- **ElasticTransform**: Mild distortion to improve robustness. Use sparingly — too aggressive breaks road structure.

### 4.2 Photometric Augmentations (Moderate Impact)

- **RandomBrightnessContrast**: Simulates varying sun angles and atmospheric conditions.
- **HueSaturationValue**: Accounts for seasonal changes (green vegetation vs. brown).
- **CLAHE**: Contrast-limited adaptive histogram equalization; helps in shadowed areas.
- **GaussianBlur / GaussNoise**: Simulates sensor noise and atmospheric blur. Use mild amounts.

### 4.3 Satellite-Specific Strategies

- **Mosaic augmentation**: Combine 4 images into one grid, forcing the model to see diverse contexts simultaneously.
- **Multi-scale training**: Train with random crop sizes (256, 384, 512) from the original 1024x1024 images.
- **Flip-n-Slide** (2024): During inference, tile with augmented orientations; up to 15.8% more precise than standard 50% overlap.

### 4.4 What to Avoid

- Overly aggressive color distortion (satellite imagery has constrained color space).
- CutOut/Random Erasing on small crops (can erase entire road segments).
- Aspect ratio changes (distorts road geometry).

---

## 5. Training Strategies

### 5.1 Learning Rate Schedule

- **Cosine annealing** from initial LR=1e-3 to 1e-6 is a robust default.
- **Warm-up**: Linear ramp-up over the first 5-10% of training steps stabilizes early training.
- **Optimizer**: AdamW with weight decay 1e-4 is the standard choice for segmentation.

### 5.2 Encoder Freezing (Transfer Learning)

Two-phase training:
1. **Phase 1**: Freeze pretrained encoder, train decoder only for 5-10 epochs. This avoids destroying pretrained features before the decoder has learned basic representations.
2. **Phase 2**: Unfreeze encoder, train end-to-end with 10x lower LR for the encoder than the decoder (discriminative learning rates).

### 5.3 Deep Supervision

Add auxiliary loss heads at intermediate decoder stages. This improves gradient flow to early layers and encourages discriminative features at all depths. Speed up convergence by 20-30% in practice.

### 5.4 Test-Time Augmentation (TTA)

Predict on multiple augmented versions of the input and average:
- Horizontal flip + vertical flip + original = 4 predictions averaged.
- Typically improves IoU by 1-3% at the cost of 4x inference time.
- Standard practice in competitions; optional in production.

### 5.5 Progressive Resizing

Train on 256x256 crops first (fast iterations, learn basic features), then fine-tune on 512x512 or full 1024x1024. Reduces total training time while achieving similar final accuracy.

---

## 6. Evaluation Metrics

### 6.1 Pixel-Based Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **IoU (Jaccard)** | $\frac{TP}{TP + FP + FN}$ | Primary metric for DeepGlobe. Strict: penalizes both FP and FN equally. |
| **Dice (F1)** | $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$ | Monotonically related to IoU. More forgiving than IoU. |
| **Precision** | $\frac{TP}{TP + FP}$ | How many predicted road pixels are correct. |
| **Recall** | $\frac{TP}{TP + FN}$ | How many true road pixels are detected. |
| **Pixel Accuracy** | $\frac{TP + TN}{total}$ | **Useless** for road segmentation — a model predicting all-background gets ~96% accuracy. |

### 6.2 Topology-Aware Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **APLS** (Average Path Length Similarity) | Compares shortest paths between nodes in ground truth vs. predicted road graph. Scales 0-1. | When routing/navigation quality matters. Penalizes broken intersections heavily. |
| **TOPO** | Evaluates local subgraph similarity (~300m extent) by comparing reachable nodes. | When local connectivity matters more than global routing. |
| **Relaxed Precision/Recall** | Uses a 3-pixel buffer around ground truth for tolerance. | When mask boundaries have annotation noise. |

### 6.3 Practical Recommendation

- **Report**: IoU (primary), Dice (secondary), Precision, Recall.
- **Production monitoring**: Track IoU distribution across batches, flag samples where IoU < threshold.
- **Advanced**: APLS if the downstream application is routing.

---

## 7. Post-Processing

### 7.1 Morphological Refinement

1. **Morphological closing** (dilation then erosion): Fills small gaps in road predictions, smooths boundaries.
2. **Morphological opening** (erosion then dilation): Removes small noise blobs.
3. **Connected component filtering**: Remove components smaller than a threshold (e.g., < 100 pixels).

### 7.2 Vectorization Pipeline (Mask to GeoJSON)

1. **Binarize** the probability map at an optimal threshold (tune on validation set, typically 0.3-0.5).
2. **Morphological closing** to smooth.
3. **Skeletonize** (`skimage.morphology.skeletonize`) to extract 1-pixel-wide centerlines.
4. **Graph construction** via `sknw` (skeleton to NetworkX graph) — extracts nodes (intersections) and edges (road segments) with pixel coordinates.
5. **Line simplification** via Douglas-Peucker algorithm (`shapely.geometry.LineString.simplify()`).
6. **Gap filling**: Connect nearby endpoints (< N pixels apart) to improve connectivity.
7. **Export** as GeoJSON `LineString` features. If georeferenced, transform pixel coordinates to geographic coordinates.

### 7.3 Libraries

- `scikit-image`: Skeletonization, morphological operations.
- `sknw`: Skeleton-to-graph conversion.
- `shapely`: Geometry operations, line simplification.
- `geojson`: GeoJSON serialization.
- OpenCV: Distance transform, contour detection, morphological operations.

---

## 8. Pretrained Encoders

### 8.1 Backbone Comparison

| Backbone | Params | Road Segmentation Notes |
|----------|--------|------------------------|
| **ResNet-34** | 21.8M | D-LinkNet winner choice. Best speed/accuracy tradeoff for binary segmentation. |
| **ResNet-50** | 25.6M | Marginal improvement over R34; heavier. |
| **EfficientNet-B0** | 5.3M | Best efficiency-accuracy tradeoff ("Battle of the Backbones," NeurIPS 2023). Resource-constrained option. |
| **EfficientNet-B4** | 19.3M | Strong accuracy with moderate compute. |
| **MIT-B2** | 25.4M | Best with SegFormer decoder; captures long-range dependencies. |
| **ConvNeXt-Tiny** | 28.6M | Modern CNN; competitive with transformers on dense prediction. |

### 8.2 Transfer Learning Insights

- **ImageNet pretraining** is universally beneficial — always use it.
- **Remote sensing pretraining** (SatlasPretrain, SSL4EO-S12) can provide additional 1-3% IoU gain for satellite tasks.
- Encoder choice matters less than decoder architecture and training strategy for binary road segmentation (per "Battle of the Backbones," NeurIPS 2023).
- The SMP library provides **800+ pretrained encoders** with a unified API for easy swapping.

### 8.3 Trusted Weight Sources

Only use pretrained weights from verified, trusted sources:

| Source | What It Provides | Trust Level |
|--------|-----------------|-------------|
| **torchvision** (`torch.hub`, `torchvision.models`) | Official PyTorch model zoo — ResNet, EfficientNet, MobileNet, etc. | Official (PyTorch/Meta) |
| **timm** (`pip install timm`) | 1000+ models with ImageNet weights — EfficientNet, ConvNeXt, HRNet, etc. | Maintained by Ross Wightman, widely audited |
| **SMP** (`encoder_weights='imagenet'`) | Uses torchvision/timm under the hood | Same as above |
| **HuggingFace Hub** (verified orgs only) | NVIDIA SegFormer (MIT-B0 to B5), Microsoft models | Verified organization badges |

**Do NOT use**: Random `.pth`/`.th` files from personal Dropbox, Baidu Netdisk, or Google Drive links. These have no provenance, no integrity verification, and could contain arbitrary code (PyTorch checkpoints can execute code on load via `pickle`).

---

## 9. Production and Deployment

### 9.1 Model Optimization

| Technique | Speedup | Accuracy Impact | Notes |
|-----------|---------|-----------------|-------|
| **ONNX export** | 1.5-2x | None | Interchange format; enables downstream optimization. |
| **TensorRT FP16** | 3-5x | Negligible (<0.1% IoU) | Safe default for GPU inference. |
| **TensorRT INT8** | 5-10x | Small (0.5-1% IoU) | Requires calibration dataset. Best for high-throughput. |
| **ONNX Runtime** | 1.5-3x | None | Cross-platform; works on CPU and GPU. |
| **Model distillation** | Variable | 1-3% IoU | Train smaller student from larger teacher. |

### 9.2 Tiling for Large Images

Satellite imagery can be arbitrarily large. Tiling strategy:

1. **Sliding window with overlap**: Tile the input with 128-256px overlap. Run inference on each tile. Use only the central region of each prediction (discard the "halo" border). Stitch central regions into the full output.
2. **Halo border method**: Tile size = desired output size + 2 * (half receptive field). Tiles overlap in the halo but output regions join exactly.
3. **Batch inference**: Process multiple tiles simultaneously on GPU for throughput.

### 9.3 Scaling for High Throughput

- **Horizontal scaling**: Stateless inference service behind a load balancer. Each replica loads the model independently.
- **Task queue**: Celery/Redis or AWS SQS for async processing of large batches.
- **GPU serving**: Triton Inference Server for multi-model serving with dynamic batching.
- **Caching**: Cache results for previously processed tiles (content-hash key).

### 9.4 Model Lifecycle

- **Versioning**: Tag models with version, training date, dataset hash, and metrics.
- **A/B testing**: Route a percentage of traffic to a new model version; compare IoU on a held-out set.
- **Rollback**: Keep previous model artifacts; swap via config/env variable.
- **Data drift detection**: Monitor input image statistics (mean, std, histogram) vs. training distribution. Alert on significant divergence (KL divergence or KS test).

---

## 10. Recommended Approach for This Project

Based on the literature and what is practically available with trusted, verified sources:

1. **Architecture**: U-Net with ResNet-34 encoder via SMP (`pip install segmentation-models-pytorch`). Approximates the D-LinkNet winning formula. Clean, fast, actively maintained.
2. **Encoder weights**: ImageNet-pretrained ResNet-34 from torchvision (official PyTorch model zoo, loaded automatically by SMP with `encoder_weights='imagenet'`).
3. **Loss**: BCE + Dice compound loss (equal weight).
4. **Metrics**: IoU (primary), Dice, Precision, Recall.
5. **Input**: 512x512 resize, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
6. **Augmentations**: RandomRotate90, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate.
7. **Training**: AdamW, cosine annealing LR, 50 epochs, early stopping on val IoU. Train on DeepGlobe ourselves.
8. **Post-processing**: Threshold tuning, morphological closing, optional skeletonization for GeoJSON output.
9. **API**: FastAPI with ONNX Runtime for optimized inference.

**Why train from scratch on DeepGlobe?** No trustworthy, pre-trained road segmentation weights exist from verified sources. All published DeepGlobe checkpoints are hosted on personal file-sharing services with no integrity guarantees. ImageNet encoder pretraining provides a strong feature initialization; we only need to train the decoder and fine-tune the encoder on our target dataset.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
2. Zhou, L., Zhang, C., & Wu, M. (2018). "D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction." CVPR Workshop.
3. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV.
4. Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). "Deep High-Resolution Representation Learning for Visual Recognition." CVPR.
5. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS.
6. Batra, A., Singh, S., Pang, G., Basu, S., Jawahar, C.V., & Paluri, M. (2019). "Improved Road Connectivity by Joint Learning of Orientation and Segmentation." CVPR.
7. Salehi, S.S.M., Erdogmus, D., & Gholipour, A. (2017). "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks." MLMI Workshop.
8. Berman, M., Triki, A.R., & Blaschko, M.B. (2018). "The Lovasz-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks." CVPR.
9. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). "Focal Loss for Dense Object Detection." ICCV.
10. Diakogiannis, F.I., et al. (2022). "Comparative Study of Loss Functions for Road Segmentation." International Journal of Applied Earth Observation and Geoinformation.
11. Goldblatt, R., et al. (2023). "Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks." NeurIPS.
12. Van Etten, A. (2019). "APLS: Average Path Length Similarity Metric for Evaluating Road Network Predictions." SpaceNet.
13. He, S., Bastani, F., Jagwani, S., et al. (2020). "Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding." ECCV.
