# InfluencerFormer

End-to-end instance segmentation via learned condensation with Influencer Loss and transformer architectures.

## Overview

InfluencerFormer generalizes the [Influencer Loss](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09016/epjconf_chep2024_09016.html) from particle physics track reconstruction to mainstream instance segmentation benchmarks (COCO, S3DIS/ScanNet).

The core idea: instead of predicting explicit masks per instance (MaskFormer) or using fixed object queries (DETR), learn an **embedding space** where same-instance points/pixels cluster around dynamically discovered **influencer points** — actual data points that serve as instance representatives.

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
└── literature_review.md  # Comprehensive survey of related work
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

## Key Advantages Over MaskFormer/DETR

| Property | MaskFormer/DETR | InfluencerFormer |
|---|---|---|
| Instance count | Fixed query budget (100-300) | Emergent from data |
| Differentiability | Hungarian matching breaks gradients | Fully differentiable |
| Data structure | Regular grids (images/voxels) | Any point set or graph |
| Representatives | Abstract learned vectors | Actual data points |
| Post-processing | Score threshold + NMS/argmax | Distance threshold in embedding space |

See [docs/literature_review.md](docs/literature_review.md) for a comprehensive survey.
