# InfluencerFormer: First Pass

## What This Is

InfluencerFormer takes the MaskFormer/Mask2Former **architecture** — transformer decoder with learned queries producing dense per-pixel mask predictions — and replaces the **training loss**. Instead of Hungarian matching + dice/BCE, we use the Influencer Loss: continuous attractive/repulsive potentials where queries naturally "claim" instances without combinatorial assignment.

The insight: the innovation is not a new architecture or a new paradigm (embedding clustering vs. mask prediction). It's a **loss function replacement** within the existing dominant paradigm. The queries become influencer points. The output is still dense per-pixel class vectors. But the training signal is fully differentiable, with no combinatorial matching.

The repo lives at: https://github.com/murnanedaniel/InfluencerFormer

## The Core Idea

**What stays the same as MaskFormer:**
- Backbone (Swin, ResNet, etc.) + pixel decoder
- Transformer decoder with N learned queries
- Each query produces dense per-pixel logits (the "mask")
- Output format: N masks + N class predictions

**What changes:**
- ~~Hungarian matching~~ → Influencer Loss (continuous potentials)
- ~~Dice loss + BCE on matched pairs~~ → Attraction/repulsion on dense class vectors
- Queries are no longer assigned to instances via discrete optimization — they *become* influencer points that attract their instance's pixels and repel others through differentiable operations

**Why this matters:**
- Hungarian matching is non-differentiable — gradients don't flow through the assignment
- It's O(N³) and causes slow convergence (DETR: ~500 epochs)
- An entire line of work (DN-DETR, DAB-DETR, DINO, masked attention) exists just to work around matching instability
- The Influencer Loss has formal global optima guarantees — the correct assignment is provably the unique minimum

## What Was Done

### 1. Repo Creation

Created `murnanedaniel/InfluencerFormer` on GitHub via the API using credentials from the `me` monorepo. Public repo, initialized on `main`.

### 2. Literature Review (`docs/literature_review.md`)

A comprehensive survey of **25+ instance segmentation methods** organized by paradigm. The review covers:

**Embedding/condensation lineage** (where the Influencer Loss originated):
- Discriminative Loss (2017) → Associative Embedding (2017) → Spatial Embeddings (2019) → Object Condensation (2020) → Influencer Loss (2024)

**Mask prediction methods** (the architecture InfluencerFormer adopts):
- MaskFormer, Mask2Former, OneFormer, Mask DINO, Mask R-CNN, HTC, SOLO/SOLOv2, CondInst, QueryInst

**3D point cloud methods** (both mask-based and grouping-based):
- Mask3D, SPFormer, OneFormer3D, PointGroup, SoftGroup, VoteNet, 3D-BoNet, OccuSeg, SSTNet, MASC, Superpoint Graphs

**The key reframing (Section 7–8):** Instead of "embedding vs. mask prediction paradigm comparison," the review focuses on **the Hungarian matching problem** — documenting its non-differentiability, O(N³) cost, slow convergence, and the long line of workarounds — and why the Influencer Loss is a principled replacement. The architecture doesn't change; the loss does.

### 3. Benchmark Selection

**Images: MS-COCO**
- The undisputed standard. 118K train / 5K val / 80 classes / ~860K instances
- Metric: AP (mAP at IoU 0.50:0.95)
- Every major method reports COCO numbers

**Point clouds: S3DIS** (primary, frictionless) + **ScanNet v2** (gold standard, requires access agreement)
- S3DIS: 271 rooms, 13 classes, auto-downloads via PyTorch Geometric (~2 GB)
- ScanNet v2: 1,513 scenes, 18 classes, requires signing Terms of Use at scan-net.org
- S3DIS was chosen for the toy model because it has zero friction

### 4. Data Loading (`influencerformer/data/`)

**`s3dis.py`:**
- `download_s3dis()` — wraps PyTorch Geometric's `S3DIS` dataset, returns train/test splits
- `S3DISInstanceDataset` — PyTorch Dataset returning `{coords, colors, sem_labels, instance_labels}` dicts

**`coco.py`:**
- `download_coco()` — auto-downloads via `fiftyone` (supports `max_samples` for quick experiments)
- `download_coco_torchvision()` — manual download path via torchvision
- `COCOInstanceDataset` — PyTorch Dataset using `pycocotools`, returns `{image, masks, labels, boxes}`

### 5. Influencer Loss (`influencerformer/losses/influencer_loss.py`)

A simplified, general-purpose implementation of the condensation loss. Note: this current implementation operates on per-point embeddings (the original formulation). **The next step is to adapt it to operate on dense per-pixel class vectors** as produced by MaskFormer-style query outputs. The core attractive/repulsive dynamics are the same; the space they operate in changes from a low-dimensional embedding to the dense mask-logit space.

Current implementation:
- **Soft influencer selection:** Temperature-scaled softmax over β (differentiable alternative to argmax)
- **Attractive term:** Pulls same-instance embeddings toward their influencer
- **Repulsive term:** Hinge loss pushing influencers apart (margin = 1.0)
- **Beta term:** One high-β point per instance + background suppression

### 6. Toy Models (`examples/`)

These are preliminary models using the embedding formulation. They demonstrate the data pipeline and loss function work end-to-end, but don't yet use the MaskFormer architecture or the dense-class-vector formulation.

**`toy_pointcloud.py` — S3DIS:**
- `PointEmbeddingMLP`: 3-layer MLP, embedding + β heads
- Greedy clustering at inference
- Evaluation: IoU-based matching → precision/recall/F1

**`toy_image.py` — COCO:**
- `PixelEmbeddingCNN`: Encoder-decoder with skip connections, embedding + β heads
- Subsamples 2048 pixels per image for tractable loss computation

### 7. Project Structure

```
InfluencerFormer/
├── .gitignore
├── README.md                              # Overview + what changes vs. MaskFormer
├── setup.py                               # pip install -e ".[pointcloud]"
├── docs/
│   ├── first_pass.md                      # This file
│   └── literature_review.md               # 25+ method survey, loss function focus
├── examples/
│   ├── toy_pointcloud.py                  # S3DIS toy model (embedding formulation)
│   └── toy_image.py                       # COCO toy model (embedding formulation)
├── influencerformer/
│   ├── __init__.py                        # v0.1.0
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coco.py                        # COCO download + dataset
│   │   └── s3dis.py                       # S3DIS download + dataset
│   ├── losses/
│   │   ├── __init__.py
│   │   └── influencer_loss.py             # Core loss (embedding formulation)
│   ├── metrics/                           # (placeholder)
│   ├── models/                            # (placeholder)
│   ├── networks/                          # (placeholder)
│   └── utils/                             # (placeholder)
└── tests/                                 # (empty)
```

### 8. What's Placeholder / Next Steps

1. **Adapt the loss to dense class vectors.** The current `influencer_loss.py` operates on per-point embeddings. The real InfluencerFormer loss should operate on the dense per-pixel mask logits that MaskFormer queries produce — the attraction/repulsion dynamics applied to the N × H × W output space.
2. **MaskFormer integration.** Either fork Mask2Former or use the `mask2former` module from detectron2/mmdetection, swap the loss.
3. **Head-to-head comparison.** Same backbone (e.g., Swin-L), same data, same training schedule — only the loss changes. Compare AP on COCO val.
4. **3D extension.** Same loss swap in Mask3D or SPFormer for ScanNet v2 benchmarking.
5. **Ablations.** Attractive vs. repulsive weight balance, temperature for soft influencer selection, comparison with dice loss properties (scale invariance, class imbalance handling).
