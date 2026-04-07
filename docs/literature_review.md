# Road Extraction From Satellite Imagery: Literature Review


## Bottom line

To get a conservative but strong baseline:

1. Start with a U-Net-style encoder-decoder with a pretrained CNN encoder.
2. Train on crops, not full images, with aggressive geometric and moderate color augmentation. **(too slow for my case with no GPU)**
3. Use a loss that explicitly addresses class imbalance and overlap quality, such as `BCE + Dice` or `BCE + Jaccard/IoU`.
4. Evaluate with IoU and Dice/F1, not pixel accuracy alone.
5. Add overlap-aware tiling or full-tile inference plus light post-processing if needed. **(check later if I have time)**
6. Treat vectorization/topology as a separate layer after mask prediction unless graph extraction is the main product requirement. (not very relevant to me also)

That baseline is well supported by both the DeepGlobe challenge literature and general segmentation literature.

## 1. Problem framing

Road extraction from satellite imagery is harder than generic binary segmentation because roads are:

- thin and elongated
- sparse relative to background
- topologically constrained
- often occluded by trees, buildings, shadows, clouds, or low-contrast surfaces

The 2024 survey by Mo et al. is useful for a modern overview of the field. It highlights the same recurring issues: cluttered backgrounds, large road-shape variation, occlusion, and complex topology in dense urban scenes. It also groups work into fully supervised, semi-supervised, and weakly supervised settings. For this take-home, fully supervised segmentation is the right regime.  
Source: [Mo et al., 2024](https://www.mdpi.com/1424-8220/24/5/1708)

The 2022 survey by Chen et al. is broader and helpful when you want context beyond optical imagery. It is useful mainly for understanding where the field goes after a mask predictor: graph extraction, multi-source fusion, and 3D/point-cloud variants.  
Source: [Chen et al., 2022](https://www.sciencedirect.com/science/article/pii/S1569843222000358)

## 2. Datasets and benchmarks that matter

### DeepGlobe

DeepGlobe is directly relevant because it is the dataset used by the assignment. The DeepGlobe challenge paper reports:

- RGB imagery
- 50 cm/pixel resolution
- imagery from Thailand, Indonesia, and India
- large source images tiled into 1024x1024 crops
- road extraction posed as binary segmentation with pixel-wise IoU as the challenge metric

The paper also notes that the dataset sampling tried to cover rural and urban areas and different road surfaces. That matters because model generalization is influenced by both surface appearance and scene context.  
Source: [DeepGlobe challenge paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)

### SpaceNet roads

SpaceNet matters even if you do not train on it. It is the main benchmark the field uses when connectivity is more important than raw mask overlap. The SpaceNet road challenge emphasizes the graph-based APLS metric, which penalizes disconnected or invalid road graphs.  
Sources:
- [SpaceNet dataset overview](https://spacenet.ai/spacenet-roads-dataset/)
- [SpaceNet challenge series paper](https://arxiv.org/abs/1807.01232)

### Practical implication

For the Hudhud take-home:

- DeepGlobe tells you what to optimize first: binary mask quality.
- SpaceNet tells you what to discuss in the write-up: a good mask is not always a good road network.

## 3. Model families worth knowing

### 3.1 U-Net and U-Net-style encoder-decoders

The original U-Net remains the most important conceptual baseline: multi-scale encoder-decoder structure plus skip connections for precise localization. That idea remains dominant because roads are thin structures and benefit from mixing coarse context with high-resolution detail.  
Source: [Ronneberger et al., U-Net](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)

For this project, the most relevant descendants are not exotic variants but practical pretrained encoder-decoders:

- U-Net with ResNet encoder
- LinkNet / D-LinkNet
- UNet++
- feature-pyramid-style CNN decoders

### 3.2 Challenge-proven CNN baselines

Two DeepGlobe 2018 workshop papers are especially useful because they are close to your exact problem.

#### Buslaev et al., 2018

This is the cleanest practical baseline for the assignment:

- U-Net family network
- pretrained ResNet-34 encoder
- decoder adapted from vanilla U-Net
- combined BCE-style classification loss with Jaccard/IoU term
- heavy spatial and color augmentation
- TTA at inference

The paper is valuable because it is not just architecture choice. It gives a complete training recipe that was competitive on DeepGlobe.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

#### Zhou et al., 2018, D-LinkNet

D-LinkNet is one of the strongest road-extraction-specific CNN baselines. It keeps LinkNet’s efficient encoder-decoder structure and adds dilated convolutions in the center to widen receptive field without collapsing spatial resolution. That is a strong fit for roads: local detail matters, but context over longer linear structures matters too. The paper reports better validation performance than plain U-Net and plain LinkNet variants on DeepGlobe.  
Source: [Zhou et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)

### 3.3 DeepLabv3+

DeepLabv3+ is a general semantic segmentation milestone. The most relevant idea here is atrous convolution plus multi-scale context aggregation, which often helps when the target class varies in thickness and context. Roads do exactly that. If you need a stronger off-the-shelf architecture than U-Net and can tolerate a bit more complexity, DeepLabv3+ is a good second choice.  
Source: [Chen et al., 2018](https://arxiv.org/abs/1802.02611)

### 3.4 SegFormer and modern transformer encoders

SegFormer is one of the more practical transformer-era architectures for semantic segmentation. It gives you multiscale features with a lightweight decoder and usually better robustness to resolution mismatch than older transformer approaches. For a real product, SegFormer is a credible option. For a 6 to 8 hour take-home, it is only worth it if:

- you already know the implementation well, or
- you use a stable pretrained library implementation

Otherwise, the engineering complexity may not pay back.  
Source: [Xie et al., 2021](https://research.nvidia.com/labs/lpr/publication/xie2021segformer/)

### 3.5 Topology- or graph-first approaches

These are important to understand, but usually not the right first implementation for this assignment.

#### DeepRoadMapper

DeepRoadMapper uses segmentation first, then graph extraction with topology repair. It is the clearest example of a two-stage “mask first, graph second” pipeline. This is conceptually close to what you would likely do in the take-home if you return GeoJSON or centerlines after mask inference.  
Source: [Mattyus et al., 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

#### RoadTracer

RoadTracer is important because it argues that segmentation plus heuristics is often weak on connectivity. Instead it directly builds the graph through iterative search. This is very relevant if your real product goal is routing-quality road graphs rather than just segmentation masks. For the take-home, it is usually better to reference this paper in the write-up than to implement its full approach.  
Source: [Bastani et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)

#### Single-shot graph extraction

More recent work also tries to predict road graphs end-to-end without an explicit segmentation-then-vectorization pipeline. This is useful future-looking context but too ambitious for the assignment unless graph extraction is the main objective.  
Source: [Bahl et al., 2022](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Bahl_Single-Shot_End-to-End_Road_Graph_Extraction_CVPRW_2022_paper.pdf)

## 4. Which architectures are best for this take-home?

### Best first implementation

`U-Net + pretrained ResNet encoder`

Why:

- easiest to explain
- easiest to train reliably
- aligned with challenge-winning DeepGlobe recipes
- thin-structure segmentation benefits from skip connections
- broad library support

### Best “slightly stronger” option

`D-LinkNet`

Why:

- directly road-extraction-oriented
- challenge-proven on DeepGlobe
- better long-range context than a naive U-Net baseline
- still simpler than many transformer setups

### Best modern option if you already know it

`SegFormer-B0/B1`

Why:

- strong pretrained support
- efficient multiscale encoder
- robust segmentation backbone

Risk:

- more moving pieces for a time-boxed assignment

## 5. Loss functions and optimization

Road extraction is highly imbalanced: background dominates. That makes plain BCE or plain cross-entropy fragile, especially when roads are thin.

### Strong practical choices

#### BCE + Jaccard/IoU

Buslaev et al. use a combined classification loss with a Jaccard term. This is highly relevant because the challenge itself is scored by IoU.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

#### BCE + Dice

This is not specific to roads, but it is a very standard and effective way to stabilize learning on sparse foreground classes. Dice-style overlap terms are a good default when the foreground class is thin and rare.

#### Lovasz-Softmax / Lovasz hinge

Lovasz losses are worth knowing because they directly target IoU-like optimization. They are particularly defensible when your offline metric is IoU. In practice, for a take-home, they are a good experiment after a simpler BCE+Dice or BCE+IoU baseline.  
Source: [Berman et al., 2018](https://arxiv.org/abs/1705.08790)

#### Focal or Tversky-style losses

These exist to address class imbalance and the precision-recall tradeoff.

- Focal Loss downweights easy examples and helps when easy background overwhelms training.  
  Source: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Tversky Loss is useful when recall of thin positives matters more than absolute precision.  
  Source: [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

### Practical recommendation

For the assignment:

1. Start with `BCE + Dice` or `BCE + IoU`.
2. Only switch to focal/Tversky/Lovasz if your validation masks show:
   - broken thin roads
   - poor recall on sparse segments
   - unstable convergence

This is an inference from the literature, not a direct claim from one paper: simpler overlap-aware losses are usually easier to implement and explain under time pressure, while still addressing the main failure mode of sparse foreground segmentation.

## 6. Training strategy that literature supports

### 6.1 Crop-based training

The challenge images are 1024x1024, but both the original U-Net logic and the DeepGlobe competition solutions support training on crops. Buslaev et al. explicitly train on random 448x448 crops after geometric transforms. This is useful because it:

- increases sample diversity
- fits memory better
- lets you oversample informative regions

Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### 6.2 Strong geometric augmentation

Across road extraction papers, geometric augmentation is standard and justified:

- flips
- rotations
- scale changes
- random crops

This is especially appropriate because roads have no single canonical orientation.

Buslaev et al. use scale transforms, rotation, random crops, and color transforms. D-LinkNet also reports that stronger augmentation with color jitter and shape transformations improved validation IoU.  
Sources:
- [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)
- [Zhou et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)

### 6.3 Moderate color augmentation

Moderate color augmentation is useful because satellite imagery varies by season, sensor, illumination, haze, and geography. But overdoing color distortion can destroy useful material cues. Buslaev et al. use brightness, contrast, and HSV transforms, which is a strong precedent for mild-to-moderate radiometric augmentation.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### 6.4 Positive-aware sampling

I did not find a single canonical paper in this pass that formalizes positive-aware crop sampling for DeepGlobe, but it is strongly justified by:

- DeepGlobe’s binary road segmentation framing
- challenge-winning use of crops
- sparse positive structure
- overlap-aware loss literature

Inference from sources: a practical loader should avoid spending too many batches on nearly empty background crops. A simple rule like “sample a proportion of crops centered on non-empty masks” is likely a good engineering choice.

### 6.5 Optimizer and regularization

The competition baselines use Adam successfully. Buslaev et al. use Adam with learning rate `1e-4`, spatial dropout, and TTA; D-LinkNet also uses standard deep-learning optimization and reports gains from pretrained encoders and augmentation.  
Sources:
- [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)
- [Zhou et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)

For the take-home, AdamW is also a reasonable modern default, but that is an engineering inference rather than a claim from the challenge papers.

## 7. Data and preprocessing tricks

### 7.1 Mask binarization must be explicit

Buslaev et al. note that mask values may not be strictly `0` and `255`, and recommend thresholding at `128`. This is a concrete, directly relevant detail for DeepGlobe preprocessing.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### 7.2 Check label quality and omissions

Buslaev et al. also note that labels are imperfect and that some small roads in farmlands are intentionally not annotated. This matters for both EDA and error analysis:

- some apparent false positives may actually be label omissions
- threshold tuning should consider noisy labels

Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### 7.3 Resolution handling

D-LinkNet is explicitly designed to operate on 1024x1024 DeepGlobe images while preserving detail. This supports training and inference at native tile resolution if your hardware allows it. If not, crop-based training plus full-tile or sliding-window inference is the safer alternative.  
Source: [Zhou et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)

### 7.4 Multi-source or auxiliary data

Sun et al. show that crowdsourced GPS can improve road extraction robustness and generalization. This is not needed for the take-home, but it is an excellent paper to cite in the write-up when discussing future product improvements beyond RGB-only inference.  
Source: [Sun et al., 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Leveraging_Crowdsourced_GPS_Data_for_Road_Extraction_From_Aerial_Imagery_CVPR_2019_paper.html)

## 8. Inference and deployment implications

### 8.1 Test-time augmentation

Buslaev et al. use TTA via rotations and average the predictions. This is a practical accuracy trick if latency is not strict.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

For the assignment API:

- keep TTA optional
- disable it by default if latency matters

### 8.2 Tiling and overlap

The original U-Net paper is one of the classic references for overlap-tile reasoning in segmentation. Even if you do not implement the original overlap strategy exactly, the core lesson remains valid: border artifacts can hurt large-image segmentation, so sliding-window inference with overlap and blending is often better than naively chopping tiles.  
Source: [Ronneberger et al., 2015](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)

### 8.3 Full-mask first, vectorization second

DeepRoadMapper and related work strongly suggest a practical decomposition:

1. predict a dense road mask
2. threshold and clean it
3. skeletonize or centerline it
4. convert to graph or polyline form
5. optionally repair connectivity

This is the most defensible architecture for your take-home because it separates:

- the ML problem
- the geometry/topology problem
- the API problem

Source: [Mattyus et al., 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

### 8.4 When graph-first methods matter

RoadTracer and later graph methods become more attractive when your primary metric is routing validity or graph correctness, not mask IoU. SpaceNet’s APLS metric is the clearest sign of that distinction.  
Sources:
- [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)
- [SpaceNet roads metric](https://spacenet.ai/spacenet-roads-dataset/)

## 9. Evaluation: what to measure and why

### Offline metrics

For this assignment, the most defensible set is:

- IoU / Jaccard
- Dice / F1
- precision
- recall

Why:

- IoU aligns with DeepGlobe’s benchmark framing
- Dice is more interpretable for sparse segmentation
- precision/recall explain tradeoffs for thin roads and label noise

Pixel accuracy should not be a primary metric because a background-heavy dataset can make it look deceptively good.

### Topology-aware evaluation

If you add vectorization or discuss production monitoring, cite graph-aware metrics too:

- APLS from SpaceNet
- junction/connectivity quality from graph-extraction literature

This is particularly important because segmentation papers like RoadTracer explicitly argue that local mask quality does not guarantee globally useful connectivity.  
Sources:
- [SpaceNet roads metric](https://spacenet.ai/spacenet-roads-dataset/)
- [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)

## 10. High-value papers and what each is good for

### Core dataset and task context

- [DeepGlobe 2018 challenge paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)
  - Why read it: exact dataset/task framing, imagery properties, metric.

- [SpaceNet: A Remote Sensing Dataset and Challenge Series](https://arxiv.org/abs/1807.01232)
  - Why read it: graph-aware road evaluation context and challenge design.

### Strong practical baselines

- [Buslaev et al., Fully Convolutional Network for Automatic Road Extraction From Satellite Imagery](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)
  - Why read it: probably the single best practical paper for your implementation recipe.

- [Zhou et al., D-LinkNet](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)
  - Why read it: best road-specific CNN baseline family to borrow from.

### General segmentation architecture references

- [U-Net](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)
  - Why read it: still the foundational encoder-decoder segmentation reference.

- [DeepLabv3+](https://arxiv.org/abs/1802.02611)
  - Why read it: strongest classical multiscale CNN segmentation reference.

- [SegFormer](https://research.nvidia.com/labs/lpr/publication/xie2021segformer/)
  - Why read it: modern transformer baseline that is still practical.

- [LinkNet](https://arxiv.org/abs/1707.03718)
  - Why read it: efficiency-oriented encoder-decoder baseline behind D-LinkNet.

### Losses and class imbalance

- [Lovasz-Softmax](https://arxiv.org/abs/1705.08790)
  - Why read it: direct optimization pressure toward IoU-like objectives.

- [Focal Loss](https://arxiv.org/abs/1708.02002)
  - Why read it: canonical class-imbalance reference.

- [Tversky Loss](https://arxiv.org/abs/1706.05721)
  - Why read it: precision-recall tradeoff for sparse targets.

### Topology and graph extraction

- [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)
  - Why read it: segmentation-to-graph pipeline and connectivity repair.

- [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)
  - Why read it: graph-first argument; why segmentation can fail on connectivity.

- [Single-Shot End-to-End Road Graph Extraction](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Bahl_Single-Shot_End-to-End_Road_Graph_Extraction_CVPRW_2022_paper.pdf)
  - Why read it: modern end-to-end graph extraction direction.

### Surveys

- [Chen et al., 2022 survey](https://www.sciencedirect.com/science/article/pii/S1569843222000358)
  - Why read it: broad survey across 2D and 3D road extraction.

- [Mo et al., 2024 survey](https://www.mdpi.com/1424-8220/24/5/1708)
  - Why read it: recent deep-learning-focused road extraction survey.

## 11. Recommended reading order for this assignment

If time is limited, read in this order:

1. [DeepGlobe challenge paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)
2. [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)
3. [Zhou et al., 2018, D-LinkNet](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)
4. [U-Net](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)
5. [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)
6. [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)
7. [Mo et al., 2024 survey](https://www.mdpi.com/1424-8220/24/5/1708)

## 12. Final Roadmap


- Model: `U-Net` or `D-LinkNet` with a pretrained ResNet encoder
- Loss:  `BCE + IoU` or `BCE + IoU + Boundary_Loss`
- Data:
  - explicit mask binarization
  - crop-based training
  - flips, rotations, scale jitter
  - moderate brightness/contrast/hue jitter
  - optional positive-aware crop sampling
- Metrics:
  - IoU
  - Dice
  - precision
  - recall
- Inference:
  - full tile
  - otherwise sliding windows with overlap
  - optional TTA behind



