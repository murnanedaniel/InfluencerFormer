# OneFormer3D baseline: Hungarian matching on ScanNet val
#
# Reproduces the OneFormer3D CVPR 2024 results on ScanNet.
# Reference: https://github.com/filaPro/oneformer3d
#
# Expected metrics (from paper):
#   mAP    = 49.2
#   mAP50  = 67.9
#   mAP25  = 78.8
#
# Usage:
#   ONEFORMER3D_ROOT=... python <oneformer3d>/tools/train.py \
#       configs/scannet/oneformer3d_1xb2_scannet.py
#   GPUS=4 ./scripts/train_s3dis.sh influencerformer  # if script extended for scannet

custom_imports = dict(imports=['influencerformer'], allow_failed_imports=False)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_type = 'ScanNetSegDataset'
# Update to your preprocessed ScanNet superpoint data directory.
# (OneFormer3D's data/scannet/ prep scripts produce super_points/, points/,
#  instance_mask/, semantic_mask/, scannet_infos_train.pkl, scannet_infos_val.pkl)
data_root = 'data/scannet/'

num_instance_classes = 18
num_semantic_classes = 20   # 18 things + wall + floor

# ScanNet-200 compatible class names (18 instance / thing classes)
class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'other furniture',
]
metainfo = dict(classes=class_names)

# ── Model ─────────────────────────────────────────────────────────────────────
num_instance_queries = 400
num_semantic_queries = num_semantic_classes
num_queries = num_instance_queries + num_semantic_queries

model = dict(
    type='OneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='SpConvUNet',
        num_planes=[32, 64, 128, 256, 256, 128, 96, 96],
        return_blocks=True),
    neck=dict(
        type='MinkNeck',
        in_channels=[96, 96, 128, 256, 256, 128, 96, 96],
        out_channels=128),
    decode_head=dict(
        type='UnifiedHead3D',
        in_channels=128,
        num_queries=num_queries,
        num_instance_queries=num_instance_queries,
        num_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True),
        sem_criterion=dict(
            type='SemanticCriterion',
            loss_weight=0.2,
            ignore_index=num_semantic_classes)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=400,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.45,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[18, 19]))  # wall=18, floor=19 in this mapping

# ── Training schedule ─────────────────────────────────────────────────────────
max_epochs = 512

_train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH',
         use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=False,
         with_label_3d=False, with_mask_3d=True,
         with_seg_3d=True, with_sp_mask_3d=True),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-3.14159, 3.14159],
         scale_ratio_range=[0.8, 1.2],
         translation_std=[0.1, 0.1, 0.1]),
    dict(type='NormalizePointsColor', color_mean=[127.5, 127.5, 127.5]),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
               'pts_instance_mask', 'sp_pts_mask', 'gt_sp_masks'],
         meta_keys=['box_type_3d', 'lidar_path', 'num_pts_feats',
                    'num_views'])]

_data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='scannet_infos_train.pkl',
        data_prefix=_data_prefix,
        pipeline=_train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        metainfo=metainfo,
        data_prefix=_data_prefix,
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='DEPTH',
                 use_dim=[0, 1, 2, 3, 4, 5]),
            dict(type='LoadAnnotations3D', with_bbox_3d=False,
                 with_label_3d=False, with_mask_3d=True,
                 with_seg_3d=True, with_sp_mask_3d=True),
            dict(type='PointSegClassMapping'),
            dict(type='NormalizePointsColor', color_mean=[127.5, 127.5, 127.5]),
            dict(type='Pack3DDetInputs',
                 keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
                       'pts_instance_mask', 'sp_pts_mask', 'gt_sp_masks'],
                 meta_keys=['box_type_3d', 'lidar_path', 'num_pts_feats',
                            'num_views'])]))

test_dataloader = val_dataloader

val_evaluator = dict(type='UnifiedSegMetric',
                     stuff_class_inds=[18, 19], thing_class_inds=list(range(18)),
                     min_num_points=1, id_offset=2 ** 16,
                     num_classes=num_semantic_classes, ignore_index=-1,
                     classes=class_names)
test_evaluator = val_evaluator

# ── Optimiser and LR schedule ─────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0001],
    pct_start=0.05,
    div_factor=10.0,
    final_div_factor=100.0,
    by_epoch=False,
    total_steps=None)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=512)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=16,
                    max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
