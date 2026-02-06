# Literature Review: Instance Segmentation Loss Functions and Architectures

## 1. Introduction

Instance segmentation — the task of identifying and delineating each individual object in a scene — is a core problem in both computer vision and scientific computing. Architectures broadly fall into three paradigms:

1. **Mask prediction**: Learn to predict an explicit binary mask per instance (Mask R-CNN, MaskFormer).
2. **Embedding clustering**: Learn an embedding space where instances form compact clusters (Discriminative Loss, Object Condensation).
3. **Center-based grouping**: Predict instance centers and group nearby points via offsets (CenterNet, VoteNet, PointGroup).

The dominant paradigm today is mask prediction via transformer decoders (MaskFormer, Mask2Former, Mask3D). These architectures produce dense per-pixel mask predictions from learned queries — but they all rely on **Hungarian matching + dice loss** for training. Hungarian matching is combinatorial, non-differentiable, and a known source of slow convergence.

The **Influencer Loss** (Murnane, 2024) was originally developed within the embedding/condensation paradigm for particle physics. **InfluencerFormer** proposes to use it as a **drop-in loss replacement** within the MaskFormer architecture: keep the transformer decoder and dense per-pixel outputs, but replace Hungarian matching + dice with continuous attractive/repulsive potentials. Queries become "influencer points" that claim instances through differentiable dynamics on the dense class vectors, rather than through combinatorial assignment.

This review surveys all major instance segmentation techniques and loss functions, with particular attention to (a) what loss each method uses and (b) how the Influencer Loss could replace or improve upon it.

---

## 2. Embedding and Metric Learning Approaches

These methods learn a per-point (or per-pixel) embedding such that same-instance points cluster together and different-instance points are separated. The Influencer Loss originated in this lineage — but InfluencerFormer adapts its loss function to operate on dense mask-like outputs within a MaskFormer-style architecture, rather than on low-dimensional embeddings.

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

## 7. The Loss Function Problem: Hungarian Matching

The dominant query-based architectures (DETR, MaskFormer, Mask2Former, Mask3D, SPFormer, OneFormer3D) all share one thing: **Hungarian matching + dice/BCE loss** for training. This section examines the problems with this loss and how the Influencer Loss addresses them.

### 7.1 How Hungarian Matching Works in MaskFormer

1. The model predicts $N$ (query, mask) pairs: each query produces a class logit and a dense mask (via dot product with per-pixel features).
2. The ground truth has $K$ instances ($K \leq N$, typically $K \ll N$).
3. Hungarian matching solves the optimal bipartite assignment between the $N$ predictions and $K$ ground truths (plus $N-K$ "no object" slots).
4. Dice loss + BCE is computed only on matched pairs.

### 7.2 Problems with Hungarian Matching

| Problem | Detail |
|---|---|
| **Non-differentiable** | The assignment is a discrete optimization solved outside the computation graph. Gradients don't flow through *which query gets which instance*. The model can only learn to predict better masks given a fixed assignment, not learn the assignment itself. |
| **O(N³) complexity** | The Hungarian algorithm is cubic in the number of queries. For N=300, this is non-trivial per training step. |
| **Slow convergence** | Early in training, predictions are random and matchings are unstable — a query matched to instance A in one step may be matched to instance B the next. DETR needed ~500 epochs. Mask2Former brought this down with masked attention, but the root cause remains. |
| **Convergence workarounds** | An entire line of work exists just to stabilize matching: denoising training (DN-DETR), anchor initialization (DAB-DETR, DINO), masked attention (Mask2Former), query-text contrastive loss (OneFormer). These are patches on a fundamental loss-function problem. |
| **Fixed query budget** | The number of queries N is a hard ceiling on instance count. Scenes with more than N instances lose objects. |

### 7.3 The InfluencerFormer Proposal: Replace the Loss, Keep the Architecture

InfluencerFormer keeps the MaskFormer architecture — backbone, pixel decoder, transformer decoder, learned queries producing dense per-pixel outputs — but replaces the training loss:

| Component | MaskFormer/Mask2Former | InfluencerFormer |
|---|---|---|
| **Architecture** | Transformer decoder, queries → dense masks | **Same** |
| **Output** | Per-query dense class vectors over pixels | **Same** |
| **Query–GT assignment** | Hungarian matching (combinatorial, non-diff.) | **Influencer Loss** (continuous potentials, fully diff.) |
| **Mask quality loss** | Dice + BCE on matched pairs | **Attraction/repulsion** on dense class vectors |
| **Convergence** | ~50–500 epochs depending on tricks | Expected faster (no matching instability) |
| **Gradient flow** | Broken at matching step | **Continuous** through query–instance assignment |

The queries become **influencer points**: each query's dense output vector naturally attracts the pixels of its instance and repels pixels of other instances, via continuous potentials that have provable global optima corresponding to correct assignment.

### 7.4 Differentiability Analysis

- **MaskFormer/Mask2Former:** The mask prediction itself (dot product + sigmoid) is differentiable. But the Hungarian matching step is not — it provides supervision signals, but gradients don't flow through the assignment decision. The model can't learn *which query should own which instance*.
- **Object Condensation (Kieseler 2020):** Attractive/repulsive potentials are differentiable during training. But inference-time condensation-point selection (greedy assignment by $\beta$) is not.
- **Influencer Loss (Murnane 2024):** Fully differentiable throughout. The assignment of instances to queries emerges from continuous dynamics, not discrete optimization.

### 7.5 Comparison with Other Loss Functions

| Loss | Used By | Differentiable? | Combinatorial? | Convergence |
|---|---|---|---|---|
| Hungarian + dice/BCE | DETR, MaskFormer, Mask2Former, Mask3D | No (matching) | Yes, O(N³) | Slow |
| Multi-task (CE + L1 + BCE) | Mask R-CNN | Yes | No (uses proposals) | Fast |
| Focal + dice | SOLO, SOLOv2 | Yes | No | Fast |
| Pull/push embedding | Discriminative Loss, Assoc. Embed. | Yes | No | Moderate |
| Attract/repel potentials | Object Condensation | Yes | No | Moderate |
| **Influencer Loss** | **InfluencerFormer** | **Yes** | **No** | **Expected fast** |

---

## 8. Why Replace Hungarian Matching with the Influencer Loss?

### 8.1 Core Advantages

1. **Full differentiability.** Gradients flow through the entire pipeline including query–instance assignment. The model can learn *which query should own which instance*, not just how to predict masks given a fixed assignment.

2. **No combinatorial optimization.** Replaces O(N³) Hungarian matching with O(N·K) continuous potentials. Simpler, faster, more stable.

3. **Formal optimality guarantees.** The Influencer Loss has provable global optima: the correct query–instance assignment is the unique global minimum. No such guarantee for Hungarian matching, which only provides locally optimal per-step assignments.

4. **Expected faster convergence.** Hungarian matching instability is the root cause of DETR's slow convergence and the motivation behind DN-DETR, DAB-DETR, masked attention, etc. Removing matching should remove the need for these workarounds.

5. **Natural extension to variable instance counts.** While the architecture still has a query budget, the loss doesn't require a fixed N — it could enable architectures where the number of queries adapts to the scene.

6. **Works across data modalities.** The same loss applies to both 2D images (MaskFormer-style) and 3D point clouds (Mask3D-style) without modification, since it operates on the dense output vectors, not on spatial structure.

### 8.2 Open Questions

- **Does it match MaskFormer/Mask2Former AP on COCO?** The Influencer Loss has only been validated in particle physics (TrackML). Demonstrating competitive results on COCO is the central goal.
- **How to handle the "no object" class?** Hungarian matching assigns excess queries to "no object." The Influencer Loss needs a mechanism for queries that don't correspond to any instance (background suppression via $\beta$).
- **Dice loss properties.** Dice loss has specific advantages for mask quality (scale invariance, handling class imbalance). The Influencer Loss must match or exceed these properties on dense mask outputs.
- **Integration with masked attention.** Mask2Former's masked attention constrains cross-attention to predicted foreground regions. This is architecturally independent of the loss and should be compatible with the Influencer Loss.

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
| Influencer Loss (original) | 2024 | CHEP/EPJC | Condensation | Attract/repel (diff.) | Yes (emergent) | Yes | 3D/graph |
| OneFormer3D | 2024 | CVPR | Query 3D unified | Hungarian | Fixed queries | Matching no | 3D |
| **InfluencerFormer** | **2025** | **—** | **Query mask (MaskFormer arch.)** | **Influencer Loss (replaces Hungarian)** | **Same as MaskFormer** | **Yes** | **2D + 3D** |

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
