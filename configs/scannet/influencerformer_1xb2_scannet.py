# InfluencerFormer on ScanNet val
#
# OneFormer3D backbone + decoder + data pipeline, with InfluencerCriterion
# replacing InstanceCriterion. Inherits all ScanNet data/training config
# from the baseline — only the instance criterion block changes.
#
# bg_weight note:
#   ScanNet scenes are ~60-70% background superpoints. bg_weight=0.1 is a
#   safe starting point, same reasoning as for S3DIS. Tune up after ~50-100
#   epochs once instances are stably separated.
#
# Usage (from OneFormer3D tools/train.py):
#   PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH \
#   python <oneformer3d>/tools/train.py configs/scannet/influencerformer_1xb2_scannet.py
#
#   # Multi-GPU
#   PYTHONPATH=/path/to/InfluencerFormer:$PYTHONPATH \
#   bash <oneformer3d>/tools/dist_train.sh configs/scannet/influencerformer_1xb2_scannet.py 4
#
#   # Resume
#   python <oneformer3d>/tools/train.py configs/scannet/influencerformer_1xb2_scannet.py --resume
#
#   # Evaluate
#   python <oneformer3d>/tools/test.py configs/scannet/influencerformer_1xb2_scannet.py \
#       work_dirs/influencerformer_scannet/epoch_512.pth

_base_ = ['./oneformer3d_1xb2_scannet.py']

model = dict(
    decode_head=dict(
        inst_criterion=dict(
            type='InfluencerCriterion',
            loss_weight=[0.5, 1.0],   # [cls_weight, influencer_scale]
            num_classes=18,
            non_object_weight=0.05,
            attr_weight=1.0,
            rep_weight=1.0,
            bg_weight=0.1,            # start low; ScanNet is ~65% background
            rep_margin=1.0,
            temperature=1.0,
            iter_matcher=True)))
