# InfluencerFormer: First Pass

## What This Is

InfluencerFormer takes the MaskFormer/Mask2Former **architecture** — transformer decoder with learned queries producing dense mask predictions — and replaces the **training loss**. Instead of Hungarian matching + dice/BCE, we use the Influencer Loss: geometric-mean attractive/repulsive potentials where the loss itself discovers query-to-instance assignments without combinatorial matching.

The repo lives at: https://github.com/murnanedaniel/InfluencerFormer

## The Architecture

```
Point Cloud (N, 6)              Learned Queries (M, D)
       │                                │
       ▼                                │
  Point Encoder                         │
       │                                │
       ▼                                ▼
  Point Embeddings (N, D) ──── Cross-Attention ────→ Mask Matrix (N, M)
                                                           │
                                                           ▼
                                                  MaskInfluencerLoss
```

1. **Embed** all N points → (N, D) point embeddings
2. **Cross-attend** between point embeddings and M learned query vectors
3. Produce an **N × M mask matrix** — each entry is how strongly point i is claimed by query m
4. **MaskInfluencerLoss** encourages all points of the same instance to unanimously claim the same query, with repulsion ensuring different instances claim different queries

## What Stays the Same as MaskFormer

- Backbone encoder for input features
- Transformer decoder with M learned queries
- Cross-attention between queries and encoded features
- Output: M mask predictions (one per query)
- Deep supervision from intermediate decoder layers

## What Changes

- ~~Hungarian matching~~ → Influencer Loss (no matcher at all)
- ~~Dice + BCE on matched query-GT pairs~~ → Geometric mean attractive + hinge repulsive on the mask matrix
- ~~SetCriterion~~ → InfluencerCriterion (drop-in replacement, same interface)

## The Loss: Mask-Matrix Influencer Loss

### Background

The original InfluencerLoss (Murnane 2024, EPJ Web Conf.) operates on two separate embedding spaces — follower and influencer embeddings — with a geometric mean formulation for the attractive loss. See [github.com/murnanedaniel/InfluencerNet](https://github.com/murnanedaniel/InfluencerNet).

The mask-matrix formulation adapts this to the MaskFormer paradigm: instead of operating on spatial embeddings, the loss operates directly on the (N, M) mask matrix.

### Attractive Loss (Follower-Influencer)

For each ground-truth instance k with points P_k:

1. Compute the **geometric mean** of mask probabilities per query:
   `geomean_k[m] = exp(mean_{i ∈ P_k} log(sigmoid(mask[i, m])))`
2. The best query for this instance is the one with the highest geometric mean
3. Loss: `-logsumexp(log_geomean[m] / T) * T` (differentiable soft-max)

The geometric mean is the key: it requires **unanimous agreement** from all points. If even one point of instance k assigns low probability to query m, the geometric mean collapses. This mirrors the `exp(mean(log(...)))` formulation in the original InfluencerNet code.

### Repulsive Loss (Influencer-Influencer)

Each instance k has a **query profile**: the M-dimensional vector of per-query geometric means.

For each pair of instances (k, l), apply a hinge loss on the L2 distance between their profiles: `max(0, margin - ||profile_k - profile_l||)`. This ensures different instances claim different queries.

### Background Suppression

Push mask probabilities toward zero for all background points (instance_label == 0).

### Comparison with Original InfluencerNet

| | Original (InfluencerNet) | Mask-Matrix (InfluencerFormer) |
|---|---|---|
| **Representation** | Two embedding spaces (follower + influencer), (N, D) each | (N, M) mask matrix |
| **Attractive** | Geometric mean of follower-influencer distances | Geometric mean of mask probabilities per query |
| **Repulsive** | Hinge on influencer-influencer distances | Hinge on instance query profile distances |
| **Matching** | None (implicit via embedding proximity) | None (implicit via geometric mean query selection) |
| **Architecture** | Any backbone → two MLP heads | MaskFormer-style: backbone → cross-attention → mask matrix |

## MaskFormer Integration

### Mask2Former Reference

Mask2Former is included as a git submodule at `third_party/Mask2Former/` for reference. The key file to compare against is `mask2former/modeling/criterion.py` (SetCriterion).

### InfluencerCriterion

`influencerformer/models/criterion.py` provides `InfluencerCriterion`, a drop-in replacement for Mask2Former's `SetCriterion`. It accepts the same `(outputs, targets)` interface:

```python
# Mask2Former's approach:
criterion = SetCriterion(matcher=HungarianMatcher(...), ...)

# InfluencerFormer's approach:
criterion = InfluencerCriterion(attr_weight=1.0, rep_weight=1.0, ...)

# Same calling convention:
losses = criterion(outputs, targets)
```

The criterion handles:
- Point clouds: `pred_masks` as (N, M) or list of (N_i, M)
- Images: `pred_masks` as (B, M, H, W), flattened to per-pixel
- Deep supervision via `aux_outputs`

## Project Structure

```
InfluencerFormer/
├── setup.py
├── docs/
│   ├── first_pass.md               # This file
│   └── literature_review.md        # 25+ method survey
├── examples/
│   ├── toy_pointcloud.py           # S3DIS example (old embedding formulation)
│   └── toy_image.py                # COCO example (old embedding formulation)
├── influencerformer/
│   ├── __init__.py                 # v0.2.0
│   ├── data/
│   │   ├── coco.py                 # COCO dataset loader
│   │   └── s3dis.py                # S3DIS dataset loader
│   ├── losses/
│   │   └── influencer_loss.py      # MaskInfluencerLoss (mask-matrix formulation)
│   ├── models/
│   │   └── criterion.py            # InfluencerCriterion (MaskFormer drop-in)
│   ├── metrics/                    # (placeholder)
│   ├── networks/                   # (placeholder)
│   └── utils/                      # (placeholder)
└── third_party/
    └── Mask2Former/                # Git submodule — reference implementation
```

## Next Steps

1. **Point cloud backbone + cross-attention model.** Wire up a point encoder (e.g. PointNet, DGCNN) with a cross-attention transformer decoder producing the (N, M) mask matrix. This is the `networks/` and `models/` gap.
2. **End-to-end training on S3DIS.** Replace the toy embedding model with the mask-matrix pipeline using InfluencerCriterion.
3. **Hyperparameter exploration.** Attractive/repulsive weights, temperature, margin, number of queries M.
4. **Image pathway.** Same loss on COCO via Mask2Former backbone, validating against the reference SetCriterion.
5. **Ablations.** Geometric mean vs arithmetic mean, sigmoid vs softmax mask probabilities, deep supervision impact.
