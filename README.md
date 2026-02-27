# InfluencerFormer

Transformer-based instance segmentation with Influencer Loss — a drop-in replacement for Hungarian matching + dice loss.

## Overview

InfluencerFormer takes the **same architecture** as [MaskFormer](https://arxiv.org/abs/2107.06278) / [Mask2Former](https://arxiv.org/abs/2112.01527) — a transformer decoder with learned queries that produce dense per-pixel mask predictions — but replaces the **training loss**.

Instead of:
- **Hungarian matching** (O(N³) combinatorial assignment, non-differentiable, slow to converge)
- **Dice loss + binary cross-entropy** on matched pairs

We use the **[Influencer Loss](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09016/epjconf_chep2024_09016.html)**: continuous attractive/repulsive potentials where queries naturally "claim" instances without combinatorial matching. The queries become **influencer points** that attract their instance's pixels and repel other instances' pixels, all through fully differentiable operations on the dense class vectors.

Originally developed for particle physics track reconstruction, the Influencer Loss is here generalized to mainstream instance segmentation benchmarks (COCO, S3DIS/ScanNet).

## What Changes (and What Doesn't)

| Component | MaskFormer/Mask2Former | InfluencerFormer |
|---|---|---|
| **Backbone** | Swin / ResNet + pixel decoder | Same |
| **Transformer decoder** | Learned queries → dense mask logits | Same |
| **Output format** | Dense per-pixel class vectors per query | Same |
| **Training loss** | Hungarian matching + dice + BCE | **Influencer Loss** (continuous potentials) |
| **Query–GT assignment** | Bipartite matching (non-differentiable) | **Attraction/repulsion** (fully differentiable) |
| **Convergence** | ~500 epochs (DETR) / ~50 epochs (Mask2Former) | Expected faster (no matching instability) |

## Project Structure

```
influencerformer/
├── data/     # Dataset download and loading (COCO, S3DIS)
├── losses/   # InfluencerLoss (condensation) + MaskInfluencerLoss (mask-matrix)
└── models/   # InfluencerCriterion — drop-in for OneFormer3D's InstanceCriterion

examples/
├── toy_pointcloud.py   # MLP on S3DIS with InfluencerLoss (condensation approach)
└── toy_image.py        # CNN on COCO with InfluencerLoss (condensation approach)

docs/
├── literature_review.md    # Survey of related work
├── first_pass.md           # Development log
└── integration_guide.md    # OneFormer3D integration (S3DIS / ScanNet)

configs/
├── s3dis/
│   ├── oneformer3d_1xb4_s3dis-area-5.py       # Baseline: Hungarian matching
│   └── influencerformer_1xb4_s3dis-area-5.py  # InfluencerFormer: Influencer Loss
└── scannet/
    ├── oneformer3d_1xb2_scannet.py             # Baseline: Hungarian matching
    └── influencerformer_1xb2_scannet.py        # InfluencerFormer: Influencer Loss

scripts/
├── train_s3dis.sh        # Training wrapper (auto-detects OneFormer3D)
├── eval_s3dis.sh         # Evaluation wrapper
└── register_criterion.py # Registry verification / integration helper
```

## Quick Start

```bash
# Install
pip install -e ".[pointcloud]"

# Point cloud example (auto-downloads S3DIS ~2GB)
python examples/toy_pointcloud.py

# Image example (requires COCO val2017 download first)
python examples/toy_image.py
```

## Why Replace Hungarian Matching?

Hungarian matching is the universal training loss for query-based segmentation (DETR, MaskFormer, Mask2Former, Mask3D, SPFormer). It has well-known problems:

1. **Non-differentiable.** Gradients don't flow through the assignment decision — the model can't learn *which query should own which instance*, only how to predict masks once assigned.
2. **O(N³) complexity.** Cubic in the number of queries, adding training cost.
3. **Slow convergence.** DETR needed ~500 epochs. Subsequent work (DN-DETR, DINO, Mask2Former) invested heavily in tricks to stabilize matching — denoising training, anchor initialization, masked attention. These are workarounds for a fundamental loss function problem.
4. **Fixed query budget.** The number of queries is a hyperparameter ceiling on instance count.

The Influencer Loss replaces all of this with continuous potentials: queries attract their instance's pixels and repel others, with formal optimality guarantees that the correct assignment is the unique global minimum.

See [docs/literature_review.md](docs/literature_review.md) for a comprehensive survey.

## OneFormer3D Integration (3D Instance Segmentation)

`InfluencerCriterion` is a drop-in replacement for OneFormer3D's `InstanceCriterion` on S3DIS / ScanNet — no changes to OneFormer3D source required.

### Install

```bash
# 1. PyTorch + mmdet3d stack (order matters)
pip install mmengine>=0.10.0 mmdet>=3.0.0
pip install mmcv>=2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 2. OneFormer3D (not on PyPI)
git clone https://github.com/filaPro/oneformer3d && pip install -e oneformer3d/

# 3. spconv — pick the wheel matching your CUDA version
#    CUDA 11.3/11.6/11.8/12.x → spconv-cu113 / cu116 / cu118 / cu120
pip install spconv-cu118

# 4. This package
pip install -e ".[mmdet3d]"

# 5. Verify
python scripts/register_criterion.py --verify
# → PASS: InfluencerCriterion is registered in the MODELS registry.
```

### Data

Run OneFormer3D's S3DIS data prep scripts (`oneformer3d/data/s3dis/`). They produce superpoint `.pth` files and per-area `.pkl` annotation files. Then set `data_root` in `configs/s3dis/oneformer3d_1xb4_s3dis-area-5.py` to point to your processed data directory.

### Train and evaluate

```bash
# Single GPU
./scripts/train_s3dis.sh baseline            # Reproduce OneFormer3D (expected mAP 59.8)
./scripts/train_s3dis.sh influencerformer    # InfluencerFormer

# Multi-GPU (uses OneFormer3D's dist_train.sh automatically)
GPUS=4 ./scripts/train_s3dis.sh influencerformer

# Evaluate
./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth
GPUS=4 ./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth

# Resume (auto-resumes from latest.pth in work_dir)
./scripts/train_s3dis.sh influencerformer --resume
```

### Key hyperparameter

`bg_weight` in `configs/s3dis/influencerformer_1xb4_s3dis-area-5.py` defaults to `0.1`, not `1.0`. S3DIS rooms are ~75% background superpoints — `bg_weight=1.0` destabilises early training by pushing all mask logits negative before the attractive term can compensate. Increase gradually after ~50–100 epochs if background superpoints are still being activated.

Full details, hyperparameter guide, and troubleshooting: [docs/integration_guide.md](docs/integration_guide.md)
