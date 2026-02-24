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
├── data/           # Dataset download and loading (COCO, S3DIS)
├── losses/         # Influencer Loss implementation
├── models/         # Lightning training modules
├── networks/       # Backbone architectures
├── metrics/        # Evaluation metrics
└── utils/          # Visualization, clustering

examples/
├── toy_pointcloud.py   # Toy model on S3DIS (point cloud)
└── toy_image.py        # Toy model on COCO (images)

docs/
├── literature_review.md    # Comprehensive survey of related work
├── first_pass.md           # Development log
└── integration_guide.md    # OneFormer3D integration (S3DIS / ScanNet)

configs/s3dis/
├── oneformer3d_1xb4_s3dis-area-5.py      # Baseline: Hungarian matching
└── influencerformer_1xb4_s3dis-area-5.py # InfluencerFormer: Influencer Loss

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

InfluencerFormer v0.2.0 adds a drop-in `InfluencerCriterion` for
[OneFormer3D](https://github.com/filaPro/oneformer3d), enabling training on
S3DIS and ScanNet without modifying any OneFormer3D source files.

**Quick summary:**

1. Install OneFormer3D + this package (`pip install -e ".[mmdet3d]"`)
2. Both config files already include the `custom_imports` hook — no manual steps
3. Train the baseline: `./scripts/train_s3dis.sh baseline`
4. Train InfluencerFormer: `./scripts/train_s3dis.sh influencerformer`
5. Verify registry: `python scripts/register_criterion.py --verify`

To swap from Hungarian matching to Influencer Loss in any OneFormer3D config,
replace the `inst_criterion` block:

```python
# Before:
inst_criterion=dict(type='InstanceCriterion', matcher=dict(...), loss_weight=[0.5, 1.0, 1.0, 0.5], ...)

# After:
inst_criterion=dict(type='InfluencerCriterion', loss_weight=[0.5, 1.0],
                    attr_weight=1.0, rep_weight=1.0, bg_weight=0.1, ...)
```

Full installation instructions, hyperparameter guide, and troubleshooting:
[docs/integration_guide.md](docs/integration_guide.md)
