# InfluencerFormer: First Pass

## What This Is

InfluencerFormer replaces Hungarian matching in transformer-based 3D instance segmentation with the Influencer Loss. We build on top of [OneFormer3D](https://github.com/filaPro/oneformer3d) (CVPR 2024), which provides a SpConv backbone + transformer query decoder for point cloud segmentation on S3DIS and ScanNet.

The repo lives at: https://github.com/murnanedaniel/InfluencerFormer

## Roadmap

1. **Reproduce** OneFormer3D's published results on S3DIS (Area 5)
2. **Swap** Hungarian matching + BCE/Dice loss → Influencer Loss
3. **Evaluate** — same model, same data, only the loss changes

## OneFormer3D Architecture (What We Build On)

```
Point Cloud (N, 6)
       │
       ▼
  SpConv U-Net Backbone         Learned Queries (M, D)
       │                              │
       ▼                              │
  Superpoint Features (S, D)          │
       │                              │
       └──── Cross-Attention ─────────┘
                    │
                    ▼
              Query Features (M, D)
                    │
            ┌───────┼───────┐
            ▼       ▼       ▼
        cls_preds  masks   scores
       (M, C+1)  (M, S)   (M, 1)
```

Key details:
- **Backbone**: SpConv U-Net (pip-installable, no MinkowskiEngine)
- **Queries**: 400 instance + 13 semantic (for S3DIS)
- **Masks**: (n_queries, n_superpoints) — each query's claim over superpoints
- **Deep supervision**: Iterative prediction at each decoder layer (`iter_pred=True`)
- **Mask generation**: `pred_mask = einsum('nd,md->nm', query_feats, mask_feats)`

## The Hookpoint: InstanceCriterion → InfluencerCriterion

OneFormer3D's loss lives in `oneformer3d/instance_criterion.py`:

```
InstanceCriterion.__call__(pred, insts)
    │
    ├── HungarianMatcher(pred_instances, gt_instances)
    │       builds cost matrix: classification + BCE + Dice
    │       calls scipy.linear_sum_assignment  ← O(N³), non-differentiable
    │       returns (query_ids, gt_ids) matched pairs
    │
    ├── Cross-entropy on cls_preds (matched queries get GT class, rest → "no object")
    ├── BCE on pred_masks[matched] vs gt sp_masks[matched]
    ├── Dice on pred_masks[matched] vs gt sp_masks[matched]
    ├── MSE on objectness scores vs IoU of matched pairs
    │
    └── Repeat for each aux_outputs layer
```

Our replacement:

```
InfluencerCriterion.__call__(pred, insts)
    │
    ├── NO MATCHER
    │
    ├── MaskInfluencerLoss(masks.T, instance_labels)
    │       masks.T: (n_superpoints, n_queries) — point-centric view
    │       Attractive: geometric mean of mask probs per instance per query
    │       Repulsive: hinge loss pushing instance query profiles apart
    │       Background suppression: push bg superpoint mask probs → 0
    │
    ├── Soft classification: assign queries to instances via mask affinity
    │       (no discrete matching — argmax of mean activation per GT instance)
    │
    └── Repeat for each aux_outputs layer
```

### Config Change (S3DIS)

```python
# OneFormer3D default:
inst_criterion=dict(
    type='InstanceCriterion',
    matcher=dict(
        type='HungarianMatcher',
        costs=[
            dict(type='QueryClassificationCost', weight=0.5),
            dict(type='MaskBCECost', weight=1.0),
            dict(type='MaskDiceCost', weight=1.0)]),
    loss_weight=[0.5, 1.0, 1.0, 0.5],
    num_classes=13,
    non_object_weight=0.05,
    fix_dice_loss_weight=True,
    iter_matcher=True,
    fix_mean_loss=True)

# InfluencerFormer:
inst_criterion=dict(
    type='InfluencerCriterion',
    loss_weight=[0.5, 1.0, 1.0],
    num_classes=13,
    non_object_weight=0.05,
    attr_weight=1.0,
    rep_weight=1.0,
    bg_weight=1.0,
    rep_margin=1.0,
    temperature=1.0,
    iter_matcher=True)
```

## The Loss: Mask-Matrix Influencer Loss

### Background

The original InfluencerLoss (Murnane 2024, EPJ Web Conf.) operates on two separate embedding spaces — follower and influencer embeddings — with a geometric mean formulation. See [github.com/murnanedaniel/InfluencerNet](https://github.com/murnanedaniel/InfluencerNet).

The mask-matrix formulation adapts this to MaskFormer-style architectures: instead of spatial embeddings, the loss operates on the (N, M) mask matrix directly.

### Attractive Loss (Follower-Influencer)

For each GT instance k with superpoints S_k:

1. Transpose masks: (n_queries, n_superpoints) → (n_superpoints, n_queries)
2. Compute **geometric mean** of mask probabilities per query:
   `geomean_k[m] = exp(mean_{s ∈ S_k} log(sigmoid(mask[s, m])))`
3. Soft-select the best query via logsumexp
4. Loss: `-logsumexp(log_geomean / T) * T`

The geometric mean requires **unanimous agreement**: one dissenting superpoint tanks the score.

### Repulsive Loss (Influencer-Influencer)

Each instance's **query profile** = M-dim vector of per-query geometric means.
Hinge loss: `max(0, margin - ||profile_k - profile_l||)` for all pairs (k, l).

### Background Suppression

Mean of sigmoid(mask_logits) for background superpoints → push toward zero.

## Tensor Shapes (OneFormer3D Convention)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `masks` (pred) | List[(n_queries, n_superpoints)] | Per-query mask logits over superpoints |
| `cls_preds` | List[(n_queries, n_classes+1)] | Per-query class logits |
| `scores` | List[(n_queries, 1)] | Per-query objectness |
| `sp_masks` (GT) | (n_gts, n_superpoints) | Binary GT instance masks |
| `labels_3d` (GT) | (n_gts,) | GT class labels per instance |

Our loss transposes `masks` to (n_superpoints, n_queries) and converts `sp_masks` to per-superpoint integer instance IDs.

## Project Structure

```
InfluencerFormer/
├── setup.py
├── docs/
│   ├── first_pass.md               # This file
│   └── literature_review.md        # 25+ method survey
├── influencerformer/
│   ├── __init__.py                 # v0.2.0
│   ├── data/
│   │   ├── coco.py                 # COCO dataset loader
│   │   └── s3dis.py                # S3DIS dataset loader
│   ├── losses/
│   │   └── influencer_loss.py      # MaskInfluencerLoss (mask-matrix formulation)
│   ├── models/
│   │   └── criterion.py            # InfluencerCriterion (OneFormer3D drop-in)
│   ├── metrics/                    # (placeholder)
│   ├── networks/                   # (placeholder)
│   └── utils/                      # (placeholder)
└── examples/
    ├── toy_pointcloud.py           # S3DIS example (old formulation)
    └── toy_image.py                # COCO example (old formulation)
```

## Next Steps

1. **Set up OneFormer3D** — Docker or conda environment, preprocess S3DIS, download pretrained backbone
2. **Reproduce baseline** — Train/eval OneFormer3D on S3DIS Area 5 with InstanceCriterion (Hungarian)
3. **Swap criterion** — Register InfluencerCriterion, modify config, train from same pretrained backbone
4. **Compare** — AP, mAP at IoU 0.25/0.50/0.75 on S3DIS Area 5
5. **Ablate** — Temperature, margin, attractive/repulsive weight balance, number of queries
