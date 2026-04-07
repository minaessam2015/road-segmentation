# Post-Processing for Road Extraction From Satellite Imagery


## Bottom line

- If you want the best binary mask: threshold tuning, light morphology, and component filtering are the highest-value first steps.
- If you want a usable centerline or GeoJSON polyline: thresholding + skeletonization + graph cleanup is the standard practical stack.
- If you want strong connectivity or routing-quality graphs: graph repair methods are more effective than simple mask heuristics.
- If you want the highest research-level connectivity improvements: learned post-processing or graph-first methods can help, but they are harder to justify in a time-boxed project.

## 1. Main categories of post-processing

The literature clusters post-processing into six broad families:

1. Probability-to-mask conversion
2. Morphological cleanup
3. Skeletonization / centerline extraction
4. Graph simplification and cleanup
5. Connectivity repair
6. Learned post-processing / refinement networks

The first four are the most practical. The last two matter when topology is important.

## 2. Probability-to-mask conversion

### What it is

Convert the model’s probability map to a binary road mask using a threshold, often followed by small cleanup rules.

### Why it matters

Roads are thin and sparse. A threshold that is too high breaks roads; too low creates blobs and spurious branches.

### Evidence

DeepRoadMapper explicitly thresholds the road-class probability map at `0.5` before centerline extraction and graph generation.  
Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

Buslaev et al. note a related but important preprocessing detail for DeepGlobe: mask labels are not always pure `0/255`, and recommend binarization at `128`. That is label preprocessing rather than output post-processing, but it reinforces the point that threshold decisions matter in this task.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### Effectiveness

Very high for little effort.

If you do nothing else, validate the threshold on a held-out set. For road extraction, this often changes the precision/recall balance more than people expect.

### Practical recommendation

- Tune the threshold on validation IoU and Dice.
- If connectivity matters, also inspect recall on thin roads visually.
- Consider separate thresholds for mask output and vectorization if needed.

This recommendation is an engineering inference from the literature, not a direct claim from a single paper.

## 3. Morphological cleanup

### What it is

Classic shape operations applied to the binary mask, typically:

- opening: remove isolated noise
- closing: connect small gaps
- erosion/dilation combinations
- hole filling

### Why it matters

Road masks often fail in two predictable ways:

- small false positives from buildings, shadows, or bright linear clutter
- tiny breaks in true roads

Morphology is often the cheapest way to improve visual cleanliness.

### Evidence

Cira et al. describe an initial post-processing step based on morphological operations with erosion followed by dilation to remove noise, separate isolated artifacts, and join slightly separated road pieces before their deeper inpainting stage.  
Source: [Cira et al., 2022](https://www.mdpi.com/2220-9964/11/1/43)

### Effectiveness

Good for mask appearance and local continuity.

Weak for larger occlusions or topology errors.

### Limits

- Can thicken roads unnaturally
- Can merge nearby parallel roads incorrectly
- Cannot reliably restore long missing connections

### Practical recommendation

Use morphology only lightly:

- small-kernel closing if roads have tiny breaks
- small-kernel opening or connected-component filtering if there is speckle noise

Do not rely on aggressive morphology to repair real connectivity problems.

## 4. Connected-component and segment filtering

### What it is

Remove tiny isolated components or short dangling segments after thresholding or graph extraction.

### Why it matters

Road segmentation often produces tiny orphan predictions that are visually road-like but not part of the network.

### Evidence

RoadTracer summarizes common segmentation-pipeline cleanup heuristics as:

- pruning short segments
- removing small connected components
- extending dead-end segments
- merging nearby junctions

The paper uses these heuristics as a baseline for segmentation-derived graph cleanup.  
Source: [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)

DeepRoadMapper also removes short curves after thinning, specifically pruning curves shorter than `5 m`.  
Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

### Effectiveness

High for reducing false positives and graph clutter.

This is one of the most reliable low-risk post-processing steps.

### Practical recommendation

- Filter connected components below an area threshold in pixel space for masks.
- After skeletonization, prune tiny branches below a length threshold.
- Tune thresholds by inspecting both rural and urban examples, because valid roads can be short in dense scenes.

## 5. Skeletonization / thinning

### What it is

Convert a thick binary road mask into a one-pixel-wide centerline representation.

### Why it matters

If your API needs a vectorized output, skeletonization is the standard bridge from segmentation mask to graph/polyline.

### Evidence

DeepRoadMapper applies thinning to the thresholded road mask to extract one-pixel-wide centerlines while preserving component connectivity.  
Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

The same paper then simplifies the centerline geometry with Ramer-Douglas-Peucker to obtain a cleaner piecewise-linear graph.  
Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

### Effectiveness

Essential if you need centerlines or graph output.

Moderately effective for representation, but not a repair method by itself.

### Limits

- Skeletonization amplifies noise if the mask is messy
- Small holes can create loops
- Broken masks become broken graphs

### Practical recommendation

Only skeletonize after you have:

- tuned the threshold
- removed small isolated components
- optionally applied light morphology

Then prune branches and simplify geometry.

## 6. Loop cleanup and graph simplification

### What it is

After thinning, simplify and clean the graph:

- remove tiny spurs
- simplify polylines
- collapse tiny loops
- merge nearby vertices/junctions

### Why it matters

Raw skeleton graphs are noisy and over-detailed. They often contain artifacts from mask holes or jagged edges.

### Evidence

DeepRoadMapper:

- simplifies skeletons with Ramer-Douglas-Peucker
- removes curves shorter than `5 m`
- converts loops smaller than `100 m` into a tree-like structure to preserve external connectivity while removing undesired local loops

Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

RoadTracer’s baseline also uses graph cleanup heuristics such as pruning short dangling segments and merging nearby junctions.  
Source: [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)

### Effectiveness

Very effective for making vectorized output usable.

This is often more important than morphology if your end goal is GeoJSON.

### Practical recommendation

For a time-boxed project that returns polylines:

1. threshold
2. skeletonize
3. convert to graph
4. prune short branches
5. merge nearby junctions
6. simplify edge geometry

That stack is simple, defensible, and literature-aligned.

## 7. Connectivity repair via graph reasoning

### What it is

Instead of only cleaning what is already predicted, infer missing road links between disconnected segments.

### Why it matters

This is the core failure mode of road segmentation: the mask can look visually fine but still break the network at important points.

### Evidence

DeepRoadMapper is the clearest example. After thresholding and thinning, it:

- builds a graph from centerlines
- generates connection hypotheses from leaf nodes
- uses A* search with non-road probability as a cost
- reasons over candidate connections to decide which missing links should be added

This is one of the strongest segmentation-plus-postprocessing pipelines for topology repair in the literature.  
Source: [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

### Effectiveness

High when your biggest problem is broken connectivity.

This is much more powerful than pure morphology for repairing occlusions or missed junctions.

### Limits

- significantly more implementation complexity
- can create false bridges between nearby but separate roads
- depends on graph thresholds and heuristics

### Practical recommendation

If the deliverable needs vector output and you want a serious connectivity story, this is the most valuable advanced post-processing idea to borrow.

For a time-boxed project, a simplified version is often enough:

- identify dead ends
- connect only short plausible gaps
- require directional alignment and high average road probability along the gap path

That is an inference from DeepRoadMapper and related work, not a direct reproduction.

## 8. CRF-based refinement

### What it is

Use conditional random fields to impose spatial consistency and shape priors on the road mask.

### Why it matters

Roads are thin, smooth, and connected, so a good structured prior can sharpen boundaries and discourage fragmented labeling.

### Evidence

Wegner et al. propose a higher-order CRF where long superpixel chains act as cliques that prefer consistent road assignment along thin linear structures. They report improvements in both per-pixel quality and topological correctness over simpler priors and heuristic completion.  
Source: [Wegner et al., 2013](https://openaccess.thecvf.com/content_cvpr_2013/html/Wegner_A_Higher-Order_CRF_2013_CVPR_paper.html)

Mnih et al.’s style of deep-road segmentation plus CRF-like refinement also appears in later road segmentation literature; for example, Tong et al. combine DCNN outputs with landscape metrics and CRF sharpening and report improved precision, recall, and F1 over the base segmentation model.  
Source: [Tong et al., 2017](https://www.mdpi.com/2072-4292/9/7/680)

### Effectiveness

Potentially useful, but less common in modern practical pipelines than it used to be.

Once the base network is strong, lightweight morphological or graph cleanup often gives better engineering payoff than adding CRF inference.

### Practical recommendation

For a time-boxed project, I would usually skip CRFs unless:

- you already have an implementation ready, or
- your masks have visibly ragged boundaries but decent connectivity

## 9. Learned post-processing / refinement networks

### What it is

Train a second network specifically to refine initial road predictions.

### Why it matters

If the base segmentation model consistently misses occluded or fractured roads, a learned refiner can model spatial priors better than hand-written heuristics.

### Evidence

RIRNet is a recent example of a dedicated post-processing network designed to repair fractures, omissions, misclassifications, and noise in initial road predictions using directional reasoning.  
Source: [RIRNet, 2024](https://www.mdpi.com/2072-4292/16/14/2666)

Cira et al. explore conditional generative inpainting as a post-processing stage and report up to `1.3%` IoU improvement over the original segmentation output, mainly by improving discontinuities and overlooked connections.  
Source: [Cira et al., 2022](https://www.mdpi.com/2220-9964/11/1/43)

### Effectiveness

Can help, especially on broken connectivity.

But for a practical project, the cost-benefit ratio is worse than simpler graph repair unless:

- you already have a stable training pipeline
- your base model is strong
- you specifically want to demonstrate advanced research awareness

### Practical recommendation

Good future work item. Usually not first-priority implementation work.

## 10. Test-time augmentation as prediction refinement

### What it is

Average predictions from rotated or flipped versions of the same tile.

### Why it matters

Not a post-processing method in the classical image-processing sense, but it is often used at inference to stabilize the probability map before thresholding.

### Evidence

Buslaev et al. improve robustness by averaging predictions over four 90-degree rotations.  
Source: [Buslaev et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)

### Effectiveness

Often useful for small quality gains.

### Limits

- increases inference latency roughly linearly
- does not solve structural failures on its own

### Practical recommendation

Treat it as optional. Good for offline experiments; usually off by default in an API.

## 11. What is most effective in practice?

If the output is a binary mask only, the highest-value stack is:

1. threshold tuning
2. connected-component filtering
3. light morphology
4. optional TTA

If the output is centerlines / polylines / GeoJSON, the highest-value stack is:

1. threshold tuning
2. connected-component filtering
3. skeletonization
4. graph pruning and simplification
5. junction merge / short-gap heuristics

If the output must preserve road-network connectivity as much as possible, the highest-value stack is:

1. threshold tuning
2. skeletonization + graph extraction
3. graph cleanup
4. explicit connectivity repair

The literature suggests that explicit graph repair is the strongest classical post-processing family for topology. DeepRoadMapper is the clearest supporting reference. In contrast, RoadTracer’s argument is that even strong segmentation-plus-heuristics pipelines still struggle with topology, which is why graph-first methods exist at all.  
Sources:
- [DeepRoadMapper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)
- [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)




## Sources

- [DeepRoadMapper: Extracting Road Topology From Aerial Images](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)
- [RoadTracer: Automatic Extraction of Road Networks From Aerial Images](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)
- [Fully Convolutional Network for Automatic Road Extraction From Satellite Imagery](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf)
- [A Higher-Order CRF Model for Road Network Extraction](https://openaccess.thecvf.com/content_cvpr_2013/html/Wegner_A_Higher-Order_CRF_2013_CVPR_paper.html)
- [Road Segmentation of Remotely-Sensed Images Using Deep Convolutional Neural Networks with Landscape Metrics and Conditional Random Fields](https://www.mdpi.com/2072-4292/9/7/680)
- [Improving Road Surface Area Extraction via Semantic Segmentation with Conditional Generative Learning for Deep Inpainting Operations](https://www.mdpi.com/2220-9964/11/1/43)
- [RIRNet: A Direction-Guided Post-Processing Network for Road Information Reasoning](https://www.mdpi.com/2072-4292/16/14/2666)
