# Literature Review: Instance Segmentation via Learned Condensation

## 1. Introduction

Instance segmentation — the task of identifying and delineating each individual object in a scene — is a core problem in both computer vision and scientific computing. Approaches broadly fall into three paradigms:

1. **Mask prediction**: Learn to predict an explicit binary mask per instance (Mask R-CNN, MaskFormer).
2. **Embedding clustering**: Learn an embedding space where instances form compact clusters (Discriminative Loss, Object Condensation, Influencer Loss).
3. **Center-based grouping**: Predict instance centers and group nearby points via offsets (CenterNet, VoteNet, PointGroup).

The **Influencer Loss** (Murnane, 2024) belongs to paradigm (2), extending the Object Condensation framework with full differentiability and formal optimality guarantees. This review surveys all major related techniques, with particular attention to how they compare with the embedding/condensation approach.

---

## 2. Embedding and Metric Learning Approaches

These methods learn a per-point (or per-pixel) embedding such that same-instance points cluster together and different-instance points are separated. The Influencer Loss is the most recent entry in this lineage.

### 2.1 Discriminative Loss (De Brabandere et al., 2017)

- **Venue:** CVPR 2017 Workshop
- **Key idea:** A CNN maps each pixel to an embedding vector. A three-term loss (a) pulls same-instance embeddings toward their mean, (b) pushes different-instance means apart, and (c) regularizes cluster means toward the origin. Post-processing uses mean-shift clustering.
- **Loss:** $L = \alpha L_\text{var} + \beta L_\text{dist} + \gamma L_\text{reg}$
- **Relation to Influencer Loss:** Direct ancestor. The Influencer Loss generalizes pull/push to graphs and point clouds, replaces the mean-based cluster center with a *learned* representative ("influencer") point, and achieves full differentiability.

### 2.2 Associative Embedding (Newell et al., 2017)

- **Venue:** NeurIPS 2017
- **Key idea:** Jointly predicts detection heatmaps and 1D "tag" embeddings. Tags for the same instance should match; tags across instances should differ. Applied to multi-person pose estimation and instance segmentation.
- **Loss:** Pull loss (variance of tags within an instance) + push loss (proximity penalty between instance mean tags).
- **Relation to Influencer Loss:** Shares the pull/push paradigm but uses very low-dimensional tags (often 1D). The Influencer Loss operates in a higher-dimensional learned latent space with explicit representative points.

### 2.3 Spatial Embeddings (Neven et al., 2019)

- **Venue:** CVPR 2019
- **Key idea:** Extends the discriminative loss to jointly learn spatial embedding coordinates *and* a per-instance clustering bandwidth $\sigma$. Larger objects get wider bandwidths. Uses the Lovász-hinge loss to directly maximize per-instance IoU. Achieves real-time performance.
- **Loss:** Lovász-hinge per instance, jointly optimizing embeddings and $\sigma$.
- **Results:** 27.6 AP on Cityscapes, outperforming Mask R-CNN at >10 fps.
- **Relation to Influencer Loss:** The learned per-instance bandwidth is analogous to the per-object condensation weight ($\beta$) in condensation approaches. Both allow the model to adapt cluster tightness per instance. Spatial Embeddings is the 2D predecessor to the 3D condensation paradigm.

### 2.4 Object Condensation (Kieseler, 2020)

- **Venue:** European Physical Journal C 80, 886
- **Key idea:** A general framework for clustering variable numbers of objects from point clouds, graphs, or images. Each vertex predicts: (1) a clustering coordinate $\mathbf{x}$ in latent space, (2) a condensation weight $\beta \in (0,1)$, and (3) object properties. The vertex with the highest $\beta$ per object becomes the condensation point. Physics-inspired attractive/repulsive potentials drive same-object points together and different-object points apart.
- **Loss:**
  - Attractive: $V_\text{att}(\mathbf{x}) = \|\mathbf{x} - \mathbf{x}_\alpha\|^2 \cdot q_\alpha$
  - Repulsive: $V_\text{rep}(\mathbf{x}) = \max(0, 1 - \|\mathbf{x} - \mathbf{x}_\alpha\|) \cdot q_\alpha$
  - Beta loss: encourages one high-$\beta$ point per object
  - Property loss: predicts object properties at condensation points, weighted by $\beta$
- **Relation to Influencer Loss:** Direct predecessor. The Influencer Loss extends Object Condensation by (1) making the full pipeline differentiable (OC's greedy inference-time assignment is not), (2) learning influencer points jointly with the embedding, and (3) proving formal global optima matching the reconstruction task.

### 2.5 Influencer Loss (Murnane, 2024)

- **Venue:** EPJ Web of Conferences 295, 09016 (CHEP 2024)
- **Key idea:** Extends geometric attention to fully differentiable, end-to-end instance segmentation in a single step. The "influencer" is the most representative point in each instance, learned jointly with the embedding. The loss has global optima that formally guarantee smooth condensation.
- **Loss:** Novel condensation loss with:
  - Follower–influencer attraction: pulls instance points toward their influencer in embedding space
  - Influencer–influencer repulsion: pushes influencers of different instances apart
  - Full differentiability throughout (no non-differentiable argmax or greedy steps)
- **Results:** Significantly outperforms baselines on TrackML; up to 10× faster inference than multi-stage pipelines.
- **Key advance over Object Condensation:** Standard OC requires non-differentiable condensation point selection at inference. The Influencer Loss makes representative-point selection part of the differentiable computation graph.

### 2.6 Mean-Shift Clustering Approaches

- **Key idea:** Several methods (Liang et al., 2017; Kong & Fowlkes, 2018) use mean-shift or its differentiable approximations as a post-processing step on learned embeddings to recover instance masks.
- **Relation to Influencer Loss:** Mean-shift discovers cluster modes (representative points) iteratively. The Influencer Loss learns them directly via the condensation weight, avoiding iterative inference-time procedures.

---

## 3. Mask Prediction Approaches

### 3.1 The MaskFormer Family

#### MaskFormer (Cheng et al., 2021)
- **Venue:** NeurIPS 2021 (Spotlight)
- **Key idea:** Reframes segmentation from per-pixel classification to **mask classification**. A transformer decoder consumes $N$ learned queries and produces per-segment embeddings. Each embedding generates a class prediction and a mask via dot product with per-pixel features.
- **Loss:** Bipartite (Hungarian) matching between predicted mask–class pairs and ground truth. Combined cross-entropy + binary cross-entropy + dice loss on matched pairs.
- **Results:** 55.6 mIoU (ADE20K semantic); 52.7 PQ (COCO panoptic).

#### Mask2Former (Cheng et al., 2022)
- **Venue:** CVPR 2022
- **Key idea:** Three improvements over MaskFormer: (1) **masked attention** constrains cross-attention to predicted foreground regions, (2) **multi-scale features** via deformable attention at resolutions 1/8, 1/16, 1/32, (3) efficiency optimizations including point-sampled loss computation.
- **Loss:** Same Hungarian-matching framework; point-sampled binary cross-entropy + dice loss.
- **Results:** 57.8 PQ (COCO panoptic), 50.1 AP (COCO instance), 57.7 mIoU (ADE20K semantic). State-of-the-art across all three tasks with a single architecture.

#### OneFormer (Jain et al., 2023)
- **Venue:** CVPR 2023
- **Key idea:** Multi-task universal segmentation trained once on a single panoptic dataset. A task token conditions the model on the desired task at inference. A query–text contrastive loss aligns query embeddings with class-label text embeddings.
- **Results:** 58.0 PQ on COCO panoptic.

#### Mask DINO (Li et al., 2023)
- **Venue:** CVPR 2023
- **Key idea:** Extends DINO (improved DETR) by adding a mask prediction branch. Demonstrates that detection and segmentation are mutually beneficial when unified.
- **Results:** 54.5 AP (COCO instance), 59.4 PQ (COCO panoptic).

### 3.2 Detect-then-Segment

#### Mask R-CNN (He et al., 2017)
- **Venue:** ICCV 2017 (Marr Prize)
- **Key idea:** Extends Faster R-CNN with a parallel mask prediction branch per RoI. Introduces RoIAlign to eliminate quantization artifacts. Predicts per-class binary masks, decoupling mask and class prediction.
- **Loss:** Multi-task: classification CE + box regression (smooth L1) + per-pixel binary CE for masks.
- **Results:** 35.7 mask AP on COCO test (ResNet-101-FPN).

#### Hybrid Task Cascade / HTC (Chen et al., 2019)
- **Venue:** CVPR 2019
- **Key idea:** Interleaves bounding box regression and mask prediction across cascade stages. Adds mask information flow between stages and an auxiliary semantic segmentation branch.
- **Results:** 48.6 mask AP (COCO test-challenge, 1st place 2018).

### 3.3 Single-Stage Mask Prediction

#### SOLO / SOLOv2 (Wang et al., 2020)
- **Venue:** ECCV 2020 / NeurIPS 2020
- **Key idea:** Box-free, grouping-free. Classifies each pixel based on its location within a spatial grid. SOLOv2 factorizes mask generation into dynamic convolution kernels + shared mask features. Matrix NMS enables parallel post-processing.
- **Loss:** Focal loss for category + dice loss for masks.
- **Results:** 38.8 AP on COCO (ResNet-101-FPN).
- **Relation to Influencer Loss:** SOLO's location-based categories are conceptually similar to condensation's proximity-based instance assignment. Both avoid proposals. But SOLO uses a fixed spatial grid; condensation dynamically discovers representatives.

#### CondInst (Tian et al., 2020)
- **Venue:** ECCV 2020 (Oral)
- **Key idea:** Generates instance-specific dynamic convolution filters conditioned on each detected center (from FCOS). Filters are applied to a shared mask feature map. Extremely compact mask head (3 layers, 8 channels each).
- **Loss:** FCOS detection losses + dice loss for masks.
- **Relation to Influencer Loss:** CondInst's dynamic filter generation parallels how the Influencer Loss learns per-instance representative points encoding instance-specific information. Both generate instance-specific representations from shared features, but through different mechanisms (convolution kernels vs. embedding coordinates).

#### QueryInst (Fang et al., 2021)
- **Venue:** ICCV 2021
- **Key idea:** Each instance is a learnable query, extending Sparse R-CNN. Exploits one-to-one correspondence between queries across cascade stages.
- **Results:** 42.8 mask AP on COCO (ResNet-101-FPN).
- **Relation to Influencer Loss:** The "instances as queries" philosophy is the closest mask-based analogue to the influencer-point concept. Both represent each instance with a single entity. Queries decode explicit masks; influencers define implicit clusters.

### 3.4 3D / Point Cloud Mask Prediction

#### Mask3D (Schult et al., 2023)
- **Venue:** ICRA 2023
- **Key idea:** First transformer-based 3D instance segmentation. Instance queries iteratively attend to multi-scale 3D features through a transformer decoder. No voting, no center prediction, no hand-tuned grouping. Directly optimizes instance masks.
- **Loss:** Hungarian matching + binary CE + dice loss for masks.
- **Results:** SOTA on ScanNet (+6.2 mAP), S3DIS (+10.1 mAP), ScanNet200 (+12.4 mAP).

#### SPFormer (Sun et al., 2023)
- **Venue:** AAAI 2023
- **Key idea:** Groups points into superpoints, then uses transformer queries with superpoint cross-attention to predict instance masks over superpoints rather than individual points. No NMS needed.
- **Results:** +4.3 mAP over prior SOTA on ScanNetv2 hidden test.

#### OneFormer3D (Kolodiazhnyi et al., 2024)
- **Venue:** CVPR 2024
- **Key idea:** Unified framework for semantic, instance, and panoptic 3D segmentation. Extends SPFormer with unified query kernels.
- **Results:** SOTA on ScanNet, ScanNet200, and panoptic segmentation.

---

## 4. Query / Set Prediction Approaches

### 4.1 DETR (Carion et al., 2020)
- **Venue:** ECCV 2020
- **Key idea:** Frames detection as direct set prediction using $N$ learned object queries + transformer encoder-decoder + bipartite matching. Eliminates anchors and NMS.
- **Loss:** Hungarian matching with class CE + L1 + GIoU for boxes; binary mask loss for panoptic extension.
- **Relation to Influencer Loss:** DETR's queries are the conceptual ancestor of "representative points." Each query uniquely represents one instance. Critical difference: DETR queries are abstract learned vectors with no geometric meaning; influencer points are actual data points with physical coordinates.

### 4.2 Panoptic SegFormer (Li et al., 2022)
- **Venue:** CVPR 2022
- **Key idea:** Extends Deformable DETR with deeply-supervised mask decoder, query decoupling (thing vs. stuff), and improved post-processing.
- **Results:** 56.2 PQ on COCO test-dev.

---

## 5. Center-Based / Grouping Approaches

### 5.1 CenterNet / CenterMask (2019–2020)
- **Key idea:** Detect objects as center keypoint heatmaps. CenterMask adds a spatial attention-guided mask branch on top of anchor-free detection.
- **Relation to Influencer Loss:** Both identify per-instance representative points. CenterNet fixes these as geometric centers; the Influencer Loss *learns* which point best represents each instance.

### 5.2 VoteNet (Qi et al., 2019)
- **Venue:** ICCV 2019
- **Key idea:** Deep Hough voting for 3D detection. Points vote toward object centers; votes are clustered and aggregated into proposals.
- **Relation to Influencer Loss:** Both identify instance representatives from point clouds. VoteNet uses explicit geometric voting toward centers; the Influencer Loss uses learned attractive potentials in embedding space.

### 5.3 PointGroup (Jiang et al., 2020)
- **Venue:** CVPR 2020
- **Key idea:** Predicts semantic labels and offset vectors to shift points toward instance centroids. Clusters in both original and shifted coordinate spaces.
- **Results:** 35.2 / 57.1 / 71.4 mAP / mAP50 / mAP25 on ScanNetv2.

### 5.4 SoftGroup (Vu et al., 2022)
- **Venue:** CVPR 2022 (Oral)
- **Key idea:** Soft bottom-up grouping with top-down refinement. Each point can be associated with multiple semantic classes (soft assignment), mitigating error propagation from hard predictions.
- **Results:** +6.2 AP50 over prior methods on ScanNet test.

### 5.5 3D-BoNet (Yang et al., 2019)
- **Venue:** NeurIPS 2019
- **Key idea:** Directly predicts 3D bounding boxes for all instances and per-point masks within each box, using a multi-task point network. Uses Hungarian matching for box assignment.
- **Results:** 48.8 mPrec / 42.7 mRec on ScanNet.

### 5.6 OccuSeg (Han et al., 2020)
- **Venue:** CVPR 2020
- **Key idea:** Predicts per-voxel occupancy signals (expected number of points per instance) to guide clustering. Graph-based partitioning on feature similarity, guided by occupancy.
- **Results:** 44.3 / 67.2 mAP50 on ScanNetv2.

---

## 6. Graph-Based Instance Segmentation

### 6.1 Superpoint Graphs (Landrieu & Simonovsky, 2018)
- **Venue:** CVPR 2018
- **Key idea:** Partitions point clouds into geometrically homogeneous superpoints. A graph neural network operates on the superpoint adjacency graph for semantic segmentation.
- **Relation to Influencer Loss:** Both exploit graph structure over point clouds. SPG uses hand-crafted geometric features for superpoint construction; the Influencer Loss learns the graph structure end-to-end.

### 6.2 MASC (Liu & Furukawa, 2019)
- **Venue:** arXiv 2019
- **Key idea:** Multi-scale Affinity with Sparse Convolution. Learns pairwise affinity (same-instance probability) between points at multiple scales using sparse convolutions. Clusters via graph cuts on the affinity graph.
- **Relation to Influencer Loss:** Both frame instance segmentation as a clustering problem on learned affinities/embeddings. MASC learns pairwise affinities; the Influencer Loss learns per-point embeddings with representative points.

### 6.3 SSTNet (Liang et al., 2021)
- **Venue:** ICCV 2021
- **Key idea:** Semantic Superpoint Tree Network. Builds a hierarchical tree of superpoints, using tree-based refinement for instance segmentation.
- **Results:** Competitive on ScanNet and S3DIS.

---

## 7. Direct Comparison: Mask Prediction vs. Embedding Condensation

### 7.1 How Object Queries Relate to Influencer Points

| Property | Object Queries (DETR / MaskFormer) | Influencer Points (Influencer Loss) |
|---|---|---|
| **Nature** | Abstract learned vectors, no physical meaning | Actual data points with physical coordinates |
| **Count** | Fixed hyperparameter (100–300) | Dynamic, emergent from the data |
| **Instance assignment** | Decode explicit masks via dot product with pixel features | Attract nearby points via distance in embedding space |
| **Training signal** | Hungarian matching ($O(N^3)$ combinatorial) | Attractive/repulsive potentials (continuous) |
| **Post-processing** | Score thresholding ± NMS or argmax | Distance thresholding in embedding space |
| **Geometric meaning** | None inherent | Full — the influencer is a specific input point |

### 7.2 Paradigm Comparison

| Dimension | Mask Prediction | Embedding Condensation | Center-Based Grouping |
|---|---|---|---|
| **Output** | Explicit binary mask per instance | Embedding coords per point; instances via clustering | Center heatmap + offsets |
| **Instance count** | Fixed query budget | Naturally variable, emergent | Variable via detection |
| **Resolution** | Often limited (28×28 in Mask R-CNN); full-res in MaskFormer | Inherently per-point | Per-point offsets |
| **Domain generality** | Primarily 2D images | Natively handles graphs, point clouds, irregular data | Requires defined "center" |
| **Differentiability** | Hungarian matching breaks gradient flow | Fully differentiable (Influencer Loss) | Grouping step not differentiable |
| **Scalability to many instances** | Limited by query budget | No upper bound | Limited by heatmap resolution |
| **Maturity on vision benchmarks** | Dominant (50+ AP on COCO) | Emerging — validated in particle physics | Strong in 3D (PointGroup, SoftGroup) |

### 7.3 Differentiability Analysis

- **Mask-based (MaskFormer):** Mask prediction (dot product + sigmoid) is differentiable. Hungarian matching during training is combinatorial — gradients do not flow through the assignment decision. Inference argmax also breaks gradient flow.
- **Object Condensation (Kieseler 2020):** Loss terms (attractive/repulsive potentials) are differentiable. Inference-time condensation point selection (greedy assignment by $\beta$) is not.
- **Influencer Loss (Murnane 2024):** Full differentiability throughout training *and* inference. Representative-point selection is part of the continuous computation graph.

### 7.4 Scalability with Instance Count

- **Mask-based:** Fixed query budget (N=100–300). Scenes with more instances than queries → missed objects. Scaling to O(1000) instances (particle physics) is impractical.
- **Embedding/condensation:** Number of instances is emergent — no upper bound beyond the number of input points. Naturally suited to variable-multiplicity domains.

### 7.5 Training Stability

- **Hungarian matching:** Known to cause slow convergence. Original DETR needed ~500 epochs vs. ~36 for Faster R-CNN. Subsequent work (Deformable DETR, DN-DETR, DINO) proposed deformable attention, denoising training, anchor initialization to mitigate this.
- **Potential-based losses:** Continuous loss over point pairs. No combinatorial matching. More stable gradients but risk of mode collapse if repulsive term is under-weighted. The Influencer Loss addresses this with formal global-optima guarantees: the correct assignment is provably the unique minimum.

---

## 8. Theoretical Advantages of the Embedding/Condensation Approach

1. **No fixed cardinality constraint.** Unlike query-based methods, condensation discovers the number of instances from the data. Critical when instance counts vary by orders of magnitude.

2. **Full differentiability.** The Influencer Loss makes the entire pipeline differentiable, enabling end-to-end optimization of downstream tasks through the instance assignment.

3. **Native support for irregular data.** Operates on arbitrary point sets and graphs without modification. Mask-based methods are designed around regular grids and require substantial redesign for unstructured data.

4. **Geometric interpretability.** Influencer points are actual data points with physical coordinates, not abstract latent vectors. In particle physics, the influencer for a track is a specific detector hit.

5. **Unified clustering and property regression.** Object properties naturally attach to condensation points within the same loss, eliminating separate task heads.

6. **Computational scaling.** $O(N \cdot K)$ for $N$ points and $K$ instances, vs. $O(N^3)$ for Hungarian matching.

7. **Formal optimality guarantees.** Global optima provably correspond to correct instance assignment. No such guarantee for Hungarian-matching losses.

8. **Single-stage simplicity.** No proposals, no RoI operations, no NMS, no separate detection head.

### Limitations to Address

The condensation paradigm has not yet demonstrated competitive performance on standard 2D vision benchmarks (COCO, ADE20K) where mask-based methods dominate. Bridging this gap is the central goal of InfluencerFormer.

---

## 9. Summary Table

| Method | Year | Venue | Paradigm | Loss Type | Variable N? | Fully Diff.? | Domain |
|---|---|---|---|---|---|---|---|
| Discriminative Loss | 2017 | CVPR-W | Embedding | Pull/push/reg | Yes | Training only | 2D |
| Associative Embedding | 2017 | NeurIPS | Embedding | Pull/push tags | Yes | Training only | 2D |
| Mask R-CNN | 2017 | ICCV | Detect-then-mask | CE+L1+BCE | Via proposals | No | 2D |
| Superpoint Graphs | 2018 | CVPR | Graph | Graph partition | Yes | No | 3D |
| MASC | 2019 | arXiv | Graph affinity | Pairwise affinity | Yes | No | 3D |
| Spatial Embeddings | 2019 | CVPR | Embedding | Lovász-hinge | Yes | Training only | 2D |
| HTC | 2019 | CVPR | Cascade mask | Multi-stage | Via proposals | No | 2D |
| VoteNet | 2019 | ICCV | Center voting | Vote + agg. | Via voting | No | 3D |
| 3D-BoNet | 2019 | NeurIPS | Box + mask | Hungarian + mask | Fixed budget | No | 3D |
| DETR | 2020 | ECCV | Query set | Hungarian | Fixed queries | Matching no | 2D |
| Object Condensation | 2020 | EPJC | Condensation | Attract/repel | Yes (emergent) | Training only | 3D/graph |
| SOLO / SOLOv2 | 2020 | ECCV/NeurIPS | Location mask | Focal + dice | Via grid | Mostly | 2D |
| CondInst | 2020 | ECCV | Dynamic filter | FCOS + dice | Via detection | Partially | 2D |
| PointGroup | 2020 | CVPR | Center-offset | Semantic + offset | Yes | No | 3D |
| OccuSeg | 2020 | CVPR | Occupancy | Occupancy + graph | Yes | No | 3D |
| MaskFormer | 2021 | NeurIPS | Query mask | Hungarian + BCE/dice | Fixed queries | Matching no | 2D |
| QueryInst | 2021 | ICCV | Query cascade | Hungarian + cascade | Fixed queries | Matching no | 2D |
| SSTNet | 2021 | ICCV | Tree | Tree partition | Yes | No | 3D |
| Mask2Former | 2022 | CVPR | Query masked attn | Hungarian + BCE/dice | Fixed queries | Matching no | 2D |
| Panoptic SegFormer | 2022 | CVPR | Query panoptic | Hungarian + deformable | Fixed queries | Matching no | 2D |
| SoftGroup | 2022 | CVPR | Soft grouping | Semantic + mask | Yes | No | 3D |
| Mask3D | 2023 | ICRA | Query 3D mask | Hungarian + BCE/dice | Fixed queries | Matching no | 3D |
| SPFormer | 2023 | AAAI | Query superpoint | Hungarian | Fixed queries | Matching no | 3D |
| OneFormer | 2023 | CVPR | Task-cond. query | Hungarian + contrastive | Fixed queries | Matching no | 2D |
| Mask DINO | 2023 | CVPR | Query unified | Hungarian + denoise | Fixed queries | Matching no | 2D |
| **Influencer Loss** | **2024** | **CHEP/EPJC** | **Condensation** | **Attract/repel (diff.)** | **Yes (emergent)** | **Yes** | **3D/graph** |
| OneFormer3D | 2024 | CVPR | Query 3D unified | Hungarian | Fixed queries | Matching no | 3D |

---

## 10. Benchmarks

### 10.1 Image Instance Segmentation: MS-COCO

The standard benchmark. 118K training images, 5K validation, 80 classes, ~860K instance annotations. Primary metric: AP (mAP averaged over IoU 0.50:0.95). Every major method reports COCO numbers.

### 10.2 Point Cloud Instance Segmentation: ScanNet v2

The gold standard for 3D. 1,513 indoor room scans (1,201 train / 312 val / 100 test), 18 instance classes, metrics: mAP, mAP25, mAP50. Secondary benchmark: **S3DIS** (271 rooms, 13 classes, freely downloadable).

---

## 11. References

- Carion, N. et al. (2020). End-to-End Object Detection with Transformers. ECCV. arXiv:2005.12872
- Chen, K. et al. (2019). Hybrid Task Cascade for Instance Segmentation. CVPR.
- Cheng, B. et al. (2021). Per-Pixel Classification is Not All You Need for Semantic Segmentation. NeurIPS. arXiv:2107.06278
- Cheng, B. et al. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. CVPR. arXiv:2112.01527
- De Brabandere, B. et al. (2017). Semantic Instance Segmentation with a Discriminative Loss Function. CVPR Workshop. arXiv:1708.02551
- Fang, Y. et al. (2021). Instances as Queries. ICCV. arXiv:2105.01928
- Han, L. et al. (2020). OccuSeg: Occupancy-aware 3D Instance Segmentation. CVPR.
- He, K. et al. (2017). Mask R-CNN. ICCV. arXiv:1703.06870
- Jain, J. et al. (2023). OneFormer: One Transformer to Rule Universal Image Segmentation. CVPR. arXiv:2211.06220
- Jiang, L. et al. (2020). PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. CVPR.
- Kieseler, J. (2020). Object Condensation: One-Stage Grid-Free Multi-Object Reconstruction. EPJC 80, 886. arXiv:2002.03605
- Kolodiazhnyi, M. et al. (2024). OneFormer3D: One Transformer for Unified Point Cloud Segmentation. CVPR. arXiv:2311.14405
- Landrieu, L. & Simonovsky, M. (2018). Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs. CVPR.
- Li, F. et al. (2023). Mask DINO: Towards A Unified Transformer-based Framework. CVPR. arXiv:2206.02777
- Li, Z. et al. (2022). Panoptic SegFormer. CVPR. arXiv:2109.03814
- Liang, Z. et al. (2021). Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks. ICCV.
- Liu, C. & Furukawa, Y. (2019). MASC: Multi-scale Affinity with Sparse Convolution. arXiv:1902.02078
- Murnane, D. (2024). Influencer Loss: End-to-end Geometric Representation Learning for Track Reconstruction. EPJ Web Conf. 295, 09016.
- Neven, D. et al. (2019). Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth. CVPR. arXiv:1906.11109
- Newell, A. et al. (2017). Associative Embedding: End-to-End Learning for Joint Detection and Grouping. NeurIPS.
- Qi, C. R. et al. (2019). Deep Hough Voting for 3D Object Detection in Point Clouds. ICCV.
- Schult, J. et al. (2023). Mask3D: Mask Transformer for 3D Semantic Instance Segmentation. ICRA. arXiv:2210.03105
- Sun, J. et al. (2023). Superpoint Transformer for 3D Scene Instance Segmentation. AAAI. arXiv:2211.15766
- Tian, Z. et al. (2020). Conditional Convolutions for Instance Segmentation. ECCV. arXiv:2003.05664
- Vu, T. et al. (2022). SoftGroup for 3D Instance Segmentation on Point Clouds. CVPR.
- Wang, X. et al. (2020). SOLO: Segmenting Objects by Locations / SOLOv2. ECCV / NeurIPS. arXiv:2003.10152
- Yang, B. et al. (2019). Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds. NeurIPS.
