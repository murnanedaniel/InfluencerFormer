# InfluencerFormer + OneFormer3D Integration Guide

---

## 1. Install

Installation order matters for the mmdet3d stack.

```bash
# PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# mmdet3d stack
pip install mmengine>=0.10.0
pip install mmcv>=2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet>=3.0.0

# OneFormer3D (not on PyPI — install from source)
git clone https://github.com/filaPro/oneformer3d
pip install -e oneformer3d/

# spconv — CUDA-version-specific wheel
#   CUDA 11.3 → spconv-cu113
#   CUDA 11.6 → spconv-cu116
#   CUDA 11.8 → spconv-cu118
#   CUDA 12.x → spconv-cu120
#   macOS/CPU → spconv-cpu
pip install spconv-cu118

# InfluencerFormer
pip install -e ".[mmdet3d]"

# Verify the criterion is registered before running any training jobs
python scripts/register_criterion.py --verify
# → PASS: InfluencerCriterion is registered in the MODELS registry.
```

---

## 2. Data Preparation

OneFormer3D ships data prep scripts for S3DIS under `oneformer3d/data/s3dis/`.
Follow the README there. The scripts produce:

- `super_points/` — per-scene superpoint assignments (`.pth` files)
- `points/` — processed point clouds
- `instance_mask/`, `semantic_mask/` — GT masks
- `s3dis_infos_Area_1.pkl` … `s3dis_infos_Area_6.pkl` — annotation index files

Once prepared, set `data_root` in
`configs/s3dis/oneformer3d_1xb4_s3dis-area-5.py` to the directory containing
the above. The InfluencerFormer config inherits this automatically.

---

## 3. Running Experiments

### Baseline (OneFormer3D, Hungarian matching)

Reproduce the published numbers first to confirm the environment is correct.

```bash
# Single GPU
./scripts/train_s3dis.sh baseline

# Multi-GPU
GPUS=4 ./scripts/train_s3dis.sh baseline

# Evaluate
./scripts/eval_s3dis.sh baseline work_dirs/baseline/epoch_512.pth
```

Expected results on S3DIS Area 5:

| mAP  | mAP50 | mAP25 |
|------|-------|-------|
| 59.8 | 78.0  | 83.1  |

### InfluencerFormer

```bash
# Single GPU
./scripts/train_s3dis.sh influencerformer

# Multi-GPU
GPUS=4 ./scripts/train_s3dis.sh influencerformer

# Resume (auto-resumes from latest.pth in work_dir)
./scripts/train_s3dis.sh influencerformer --resume

# Load specific weights (no optimizer state)
./scripts/train_s3dis.sh influencerformer --cfg-options load_from=path/to/epoch_256.pth

# Evaluate
./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth
GPUS=4 ./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth
```

---

## 4. Hyperparameters

Config file: `configs/s3dis/influencerformer_1xb4_s3dis-area-5.py`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `bg_weight` | **0.1** | Start low — S3DIS is ~75% background. `bg_weight=1.0` destabilises early training by pushing all logits negative before instances emerge. Increase gradually after ~50–100 epochs. |
| `attr_weight` | 1.0 | Relative weight of the attractive term. Increase if instances aren't being claimed. |
| `rep_weight` | 1.0 | Relative weight of the repulsive term. Increase if queries collapse to the same instance. |
| `rep_margin` | 1.0 | Hinge margin for query separation. Increase if repulsive loss is zero from the start. |
| `temperature` | 1.0 | Softness of query selection. Reduce to 0.5–0.1 once training stabilises for harder assignment. |
| `loss_weight[0]` | 0.5 | Classification loss scale. Keep matched to baseline. |
| `loss_weight[1]` | 1.0 | Outer scale on the total influencer loss (`attr+rep+bg` weights are already applied inside `MaskInfluencerLoss`). |

---

## 5. Troubleshooting

### `KeyError: 'InfluencerCriterion'`

The criterion wasn't registered. Check in order:

1. `PYTHONPATH` includes the repo root — handled automatically by `train_s3dis.sh`;
   if bypassing the script, run `export PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH`.
2. The config file has `custom_imports = dict(imports=['influencerformer'], allow_failed_imports=False)`.
3. `python scripts/register_criterion.py --verify` — prints PASS or a specific failure reason.

### Loss not decreasing / influencer loss is zero

The criterion must reference `influencer_losses['loss']` (the differentiable
total), not `['attractive']` or `['repulsive']` (detached logging copies).
Verify:

```bash
grep -n "influencer_losses\[" influencerformer/models/criterion.py
# All matches should show ['loss']
```

### CUDA out of memory

Reduce `train_dataloader.batch_size` in the config (default 4), or reduce
`num_instance_queries` (default 400, at the cost of recall on dense scenes).

### spconv import error

Reinstall matching your actual CUDA version (`nvcc --version`):

```bash
pip install spconv-cu113  # CUDA 11.3
pip install spconv-cu116  # CUDA 11.6
pip install spconv-cu118  # CUDA 11.8
pip install spconv-cu120  # CUDA 12.x
```

### `allow_failed_imports` hiding errors

The configs use `allow_failed_imports=False` so import failures surface
immediately. If you see a cryptic `KeyError` with no prior traceback, temporarily
change to `allow_failed_imports=True` to confirm the import itself is succeeding,
then switch back to `False`.
