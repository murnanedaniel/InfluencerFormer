# InfluencerFormer + OneFormer3D Integration Guide

This guide covers everything needed to train and evaluate InfluencerFormer on
S3DIS Area 5 using the OneFormer3D framework.

---

## 1. Environment Setup

### Hardware requirements

- GPU: 16 GB VRAM recommended (A100 / V100 / RTX 3090). OneFormer3D with SpConv
  is memory-intensive (~50K superpoints per scene at batch size 4).
- CUDA 11.8 recommended (matches the `spconv-cu118` wheel).

### Installation order (order matters)

```bash
# 1. PyTorch matching your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. mmdet3d build stack (in this order)
pip install mmengine>=0.10.0
pip install mmcv>=2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet>=3.0.0

# 3. OneFormer3D (filaPro fork — not on PyPI)
git clone https://github.com/filaPro/oneformer3d
pip install -e oneformer3d/

# 4. spconv — pick the wheel matching your CUDA version:
#   CUDA 11.3: pip install spconv-cu113
#   CUDA 11.6: pip install spconv-cu116
#   CUDA 11.8: pip install spconv-cu118
#   CUDA 12.x: pip install spconv-cu120
#   macOS/CPU: pip install spconv-cpu
pip install spconv-cu118

# 5. InfluencerFormer
pip install -e ".[mmdet3d]"
# or if you only want to develop without mmdet3d extras:
# pip install -e "."
# and ensure PYTHONPATH is set (handled automatically by the training scripts)

# 6. Verify the registry integration
python scripts/register_criterion.py --verify
# Should print: PASS: InfluencerCriterion is registered in the MODELS registry.
```

### Data preparation

Follow OneFormer3D's data preparation scripts to produce the superpoint `.pth`
files for S3DIS. The processed data should be placed at (or symlinked from)
`data/s3dis/`, matching the `data_root` in the config files.

After data prep, update `data_root` in
`configs/s3dis/oneformer3d_1xb4_s3dis-area-5.py` if your data lives elsewhere.
The InfluencerFormer config inherits this setting automatically.

---

## 2. Reproduce the Baseline (OneFormer3D with Hungarian Matching)

Before switching to InfluencerFormer, reproduce the published OneFormer3D
numbers to confirm the environment is correctly set up.

### Train

```bash
./scripts/train_s3dis.sh baseline
```

Work directory: `work_dirs/` (mmdet3d default). Checkpoints are saved every 16
epochs; the final checkpoint is at epoch 512.

### Evaluate

```bash
./scripts/eval_s3dis.sh baseline work_dirs/baseline/epoch_512.pth
```

### Expected results (S3DIS Area 5, from the OneFormer3D paper)

| Metric | Value |
|--------|-------|
| mAP    | 59.8  |
| mAP50  | 78.0  |
| mAP25  | 83.1  |

---

## 3. Train and Evaluate InfluencerFormer

### Train

```bash
./scripts/train_s3dis.sh influencerformer
```

### Resume from checkpoint

```bash
./scripts/train_s3dis.sh influencerformer --resume work_dirs/influencerformer/epoch_256.pth
```

### Evaluate

```bash
./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth
```

### What changed vs. the baseline

| Aspect | Baseline | InfluencerFormer |
|--------|----------|-----------------|
| Criterion type | `InstanceCriterion` | `InfluencerCriterion` |
| Training loss | BCE + Dice (on Hungarian-matched pairs) | Attractive + Repulsive + Background |
| Matcher | HungarianMatcher (required at train time) | None |
| `loss_weight` | `[0.5, 1.0, 1.0, 0.5]` (4 elements) | `[0.5, 1.0]` (2 elements) |
| Additional hyperparameters | — | `attr_weight`, `rep_weight`, `bg_weight`, `rep_margin`, `temperature` |
| Inference | Identical | Identical (same model class, checkpoint format) |

---

## 4. How the Criterion Swap Works

InfluencerFormer replaces the criterion without touching any OneFormer3D source
files. Here is how:

1. **Registration**: `InfluencerCriterion` is decorated with
   `MODELS.register_module()` inside `influencerformer/models/criterion.py`.
   This fires at import time — but only when `mmdet3d` is importable.

2. **Triggering the import**: Both config files contain:
   ```python
   custom_imports = dict(imports=['influencerformer'], allow_failed_imports=False)
   ```
   mmdet3d processes `custom_imports` before building the model, which causes
   `import influencerformer` to run, which runs `criterion.py`, which calls
   `MODELS.register_module()`.

3. **PYTHONPATH**: The training scripts export the repo root to `PYTHONPATH` so
   `import influencerformer` resolves correctly without requiring a `pip install`.

4. **Model build**: OneFormer3D then calls
   `MODELS.build(dict(type='InfluencerCriterion', ...))`, which constructs our
   criterion. It is called with the same `__call__(pred, insts)` interface as
   `InstanceCriterion` — no other OneFormer3D code changes.

To verify the integration is live before running a full training job:

```bash
python scripts/register_criterion.py --verify
```

---

## 5. Hyperparameter Guide for S3DIS

### `bg_weight` (default in config: 0.1)

S3DIS rooms contain roughly 70–80% background superpoints. The background
suppression term pushes mask probabilities for background superpoints toward
zero. With `bg_weight=1.0`:

- Early in training, all mask logits are near zero (random init).
- The background gradient pushes them more negative.
- This conflicts with the attractive term, which must push instance superpoints
  positive — but starts from an already-negative baseline.
- Result: unstable early training, slow convergence.

Starting at `bg_weight=0.1` avoids this. Once the model has learned basic
instance separation (typically after ~50–100 epochs), you can increase
`bg_weight` in a staged schedule if you observe that background superpoints are
being activated too readily.

This is a S3DIS-specific tuning. On more balanced datasets (e.g., ScanNet, where
foreground ratios are higher), `bg_weight=0.5`–`1.0` may be appropriate from
the start.

### `attr_weight` and `rep_weight` (default: 1.0 each)

Control the relative strength of attractive vs. repulsive terms within the
influencer loss. The default 1:1 balance is a good starting point. If queries
collapse to covering the same instances (mode collapse), increase `rep_weight`.

### `rep_margin` (default: 1.0)

Hinge margin for the repulsive term. With 400 queries and ~10–30 instances per
room, the query profile space has ample room for separation. Increase if
repulsive loss is at zero from the start (margin too tight relative to initial
query diversity).

### `temperature` (default: 1.0)

Softness of the log-sum-exp query selection. Lower values give sharper
(harder) selection. Start at 1.0; reduce to 0.5 or 0.1 once training stabilises
to encourage each instance to be dominated by a single query.

### `loss_weight[0]` — classification weight (default: 0.5)

Matches the baseline. Keep at 0.5 initially. Reduce if you observe the model
over-focusing on classification at the expense of mask quality.

### `loss_weight[1]` — influencer scale (default: 1.0)

Outer scale on the full influencer loss total. The `attr_weight`, `rep_weight`,
and `bg_weight` are already applied inside `MaskInfluencerLoss`, so
`loss_weight[1]` controls the overall influencer loss magnitude relative to the
classification loss. Treat this like a global learning rate multiplier for the
mask pathway.

---

## 6. Troubleshooting

### `KeyError: 'InfluencerCriterion'`

InfluencerCriterion was not registered in the MODELS registry. Check:

1. `PYTHONPATH` includes the repo root (the training scripts handle this
   automatically; check if you are calling the training script directly or
   bypassing it).
2. The config file contains:
   ```python
   custom_imports = dict(imports=['influencerformer'], allow_failed_imports=False)
   ```
3. Run the verification:
   ```bash
   python scripts/register_criterion.py --verify
   ```
4. If `allow_failed_imports=False` causes an error (import fails silently with
   `True`), set it to `False` to surface the root cause.

### Loss is zero or not decreasing after the first few epochs

If only `cls_loss` is contributing, the gradient bug has been reintroduced. Check
that `criterion.py` uses `influencer_losses['loss']` (not `['attractive']` or
`['repulsive']`, which are `.detach()` logging copies):

```bash
grep -n "influencer_losses\[" influencerformer/models/criterion.py
```

All occurrences should reference `['loss']`.

### CUDA out of memory

Reduce the batch size in the config (`train_dataloader.batch_size`) or reduce
`num_instance_queries` (at the cost of recall). OneFormer3D's default of 50K
superpoints per scene requires ~16 GB with `batch_size=4`.

### spconv version mismatch / import errors

Install the variant matching your CUDA version:

```bash
# Check your CUDA version
nvcc --version

# Install matching spconv
pip install spconv-cu113   # CUDA 11.3
pip install spconv-cu116   # CUDA 11.6
pip install spconv-cu118   # CUDA 11.8
pip install spconv-cu120   # CUDA 12.x
pip install spconv-cpu     # no CUDA / macOS
```

### `ImportError: No module named 'influencerformer'` during training

The training script exports `PYTHONPATH` automatically. If you are calling
`tools/train.py` directly (bypassing `train_s3dis.sh`), set it manually:

```bash
export PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH
python $ONEFORMER3D_ROOT/tools/train.py configs/s3dis/influencerformer_1xb4_s3dis-area-5.py
```

Or install the package: `pip install -e /path/to/InfluencerFormer`.
