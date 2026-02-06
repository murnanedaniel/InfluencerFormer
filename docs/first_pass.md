# InfluencerFormer: First Pass

## What This Is

InfluencerFormer is a new repo aimed at generalizing the Influencer Loss — originally developed for particle physics track reconstruction — into a mainstream instance segmentation method competitive with MaskFormer/Mask2Former on standard vision benchmarks.

The repo lives at: https://github.com/murnanedaniel/InfluencerFormer

## What Was Done

### 1. Repo Creation

Created `murnanedaniel/InfluencerFormer` on GitHub via the API using credentials from the `me` monorepo. Public repo, initialized on `main`.

### 2. Literature Review (`docs/literature_review.md`)

A comprehensive survey of **25+ instance segmentation methods** organized by paradigm. The goal was to map the full landscape and identify exactly where the Influencer Loss sits relative to the state of the art.

**Three paradigms identified:**

| Paradigm | Representative Methods | How It Works |
|---|---|---|
| **Mask prediction** | MaskFormer, Mask2Former, Mask R-CNN, SOLO, Mask3D | Learned queries decode explicit binary masks via dot product with pixel/point features. Hungarian matching during training. |
| **Embedding clustering** | Discriminative Loss, Spatial Embeddings, Object Condensation, **Influencer Loss** | Per-point embeddings in a latent space. Same-instance points cluster together. Representatives emerge dynamically. |
| **Center-based grouping** | CenterNet, VoteNet, PointGroup, SoftGroup | Predict instance centers + offset vectors. Group by proximity to predicted centers. |

**Key finding for positioning InfluencerFormer:** The embedding/condensation approach has clear theoretical advantages (full differentiability, no fixed query budget, native graph/point cloud support, formal optimality guarantees) but has never been validated on standard vision benchmarks. Every COCO/ScanNet leaderboard is dominated by mask-prediction methods. Bridging this gap is the entire point of this project.

**The evolution within the embedding lineage:**
- Discriminative Loss (2017) → introduced pull/push in embedding space, used mean as cluster center
- Spatial Embeddings (2019) → added learnable per-instance bandwidth
- Object Condensation (2020) → physics-inspired potentials, condensation weight β, but non-differentiable inference
- **Influencer Loss (2024)** → fully differentiable, learned representative points, formal global optima

**Direct comparison with MaskFormer (Section 7):** Detailed table comparing queries vs. influencer points on nature, count, assignment mechanism, training signal, post-processing, and geometric meaning. The core tension: MaskFormer achieves SOTA on benchmarks but uses Hungarian matching (O(N³), non-differentiable, slow to converge). Influencer Loss is fully differentiable with continuous optimization but unproven on vision benchmarks.

### 3. Benchmark Selection

**Images: MS-COCO**
- The undisputed standard. 118K train / 5K val / 80 classes / ~860K instances
- Metric: AP (mAP at IoU 0.50:0.95)
- Every major method reports COCO numbers

**Point clouds: S3DIS** (primary, frictionless) + **ScanNet v2** (gold standard, requires access agreement)
- S3DIS: 271 rooms, 13 classes, auto-downloads via PyTorch Geometric (~2 GB)
- ScanNet v2: 1,513 scenes, 18 classes, requires signing Terms of Use at scan-net.org
- S3DIS was chosen for the toy model because it has zero friction — `torch_geometric.datasets.S3DIS` handles everything

### 4. Data Loading (`influencerformer/data/`)

**`s3dis.py`:**
- `download_s3dis()` — wraps PyTorch Geometric's `S3DIS` dataset, returns train/test splits
- `S3DISInstanceDataset` — PyTorch Dataset returning `{coords, colors, sem_labels, instance_labels}` dicts. Supports `max_points` subsampling.

**`coco.py`:**
- `download_coco()` — auto-downloads via `fiftyone` (easiest path, supports `max_samples` for quick experiments)
- `download_coco_torchvision()` — for manual download + torchvision loading, with clear instructions if data is missing
- `COCOInstanceDataset` — PyTorch Dataset using `pycocotools`. Returns `{image, masks, labels, boxes}` dicts. Handles resizing, mask scaling, empty-annotation edge cases.

### 5. Influencer Loss (`influencerformer/losses/influencer_loss.py`)

A simplified, general-purpose implementation of the condensation loss:
- **Soft influencer selection:** Uses temperature-scaled softmax over β values (differentiable alternative to argmax)
- **Attractive term:** Pulls same-instance embeddings toward their influencer, weighted by condensation charge q = arctanh(β)² + 1
- **Repulsive term:** Hinge loss pushing influencers of different instances apart (margin = 1.0)
- **Beta term:** Encourages one high-β point per instance + suppresses background β
- Returns dict: `{loss, attractive, repulsive, beta}` for logging

### 6. Toy Models (`examples/`)

**`toy_pointcloud.py` — S3DIS:**
- `PointEmbeddingMLP`: 3-layer MLP (6→64→64→64), separate heads for embeddings (→8D) and β (→sigmoid)
- Input: (N, 6) = [x, y, z, r, g, b] per point, normalized
- Greedy clustering at inference: sort by β descending, greedily assign neighbors within distance threshold
- Evaluation: IoU-based matching at 0.5 threshold → precision/recall/F1
- Trains 5 epochs, evaluates on Area 5

**`toy_image.py` — COCO:**
- `PixelEmbeddingCNN`: Encoder-decoder with skip connections (3→32→64→128→64→32), heads for embeddings (→8D) and β
- Converts COCO masks to instance label maps (small-area-on-top for overlaps)
- Pads images to multiples of 4 for the encoder-decoder
- Subsamples 2048 pixels per image for tractable loss computation
- Uses 200 train / 50 test images from COCO val as a toy split

### 7. Project Structure

```
InfluencerFormer/
├── .gitignore
├── README.md                              # Overview + quick start + comparison table
├── setup.py                               # pip install -e ".[pointcloud]"
├── docs/
│   ├── first_pass.md                      # This file
│   └── literature_review.md               # 25+ method survey
├── examples/
│   ├── toy_pointcloud.py                  # S3DIS toy model
│   └── toy_image.py                       # COCO toy model
├── influencerformer/
│   ├── __init__.py                        # v0.1.0
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coco.py                        # COCO download + dataset
│   │   └── s3dis.py                       # S3DIS download + dataset
│   ├── losses/
│   │   ├── __init__.py
│   │   └── influencer_loss.py             # Core loss function
│   ├── metrics/
│   │   └── __init__.py                    # (placeholder)
│   ├── models/
│   │   └── __init__.py                    # (placeholder)
│   ├── networks/
│   │   └── __init__.py                    # (placeholder)
│   └── utils/
│       └── __init__.py                    # (placeholder)
└── tests/                                 # (empty, for future)
```

### 8. What's Placeholder / Next Steps

The `models/`, `networks/`, `metrics/`, and `utils/` packages are empty — scaffolded for the real implementation. The toy models use inline model definitions in the example scripts. Next steps would be:

1. **Real backbone:** Replace MLP/CNN with a transformer (e.g., ViT for images, Point Transformer for point clouds)
2. **Real metrics:** Implement COCO-style AP evaluation using pycocotools
3. **Training at scale:** Lightning module, multi-GPU, proper augmentation
4. **Benchmark results:** Run on full COCO val and ScanNet v2 test to get real numbers
5. **Compare head-to-head:** Same backbone (e.g., Swin-L) with Influencer Loss vs. Mask2Former loss
