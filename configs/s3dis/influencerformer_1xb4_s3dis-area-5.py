# InfluencerFormer on S3DIS Area 5
#
# OneFormer3D backbone + decoder + data pipeline, with InfluencerCriterion
# replacing InstanceCriterion. No HungarianMatcher during training.
#
# What changed vs. the baseline config:
#   - inst_criterion type: 'InstanceCriterion' → 'InfluencerCriterion'
#   - loss_weight: [0.5, 1.0, 1.0, 0.5] → [0.5, 1.0]  (see note below)
#   - No matcher block
#   - Added attr_weight, rep_weight, bg_weight, rep_margin, temperature
#   - bg_weight=0.1 (not 1.0): see S3DIS-specific note below
#
# Hyperparameter note — bg_weight:
#   S3DIS rooms contain ~70-80% background superpoints. With bg_weight=1.0,
#   early training pushes all mask logits strongly negative (background
#   suppression dominates before instances emerge), conflicting with the
#   attractive term. Starting at 0.1 stabilises early training. Increase
#   gradually once the model has learned basic instance separation.
#
# loss_weight note:
#   loss_weight=[cls_weight, influencer_scale]
#   - cls_weight (0.5): scales the classification cross-entropy
#   - influencer_scale (1.0): outer scale on the full influencer loss total.
#     attr_weight, rep_weight, bg_weight are already applied inside
#     MaskInfluencerLoss before being summed into 'loss'.
#   loss_weight[2] is silently ignored if present (backward compat).
#
# Usage:
#   ./scripts/train_s3dis.sh influencerformer
#   ./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth

_base_ = ['./oneformer3d_1xb4_s3dis-area-5.py']

# Override only the instance criterion — everything else (backbone, neck,
# decoder head structure, data pipeline, training schedule) is inherited.
model = dict(
    decode_head=dict(
        inst_criterion=dict(
            type='InfluencerCriterion',
            loss_weight=[0.5, 1.0],   # [cls_weight, influencer_scale]
            num_classes=13,
            non_object_weight=0.05,
            attr_weight=1.0,
            rep_weight=1.0,
            bg_weight=0.1,            # see note above — start low for S3DIS
            rep_margin=1.0,
            temperature=1.0,
            iter_matcher=True)))
