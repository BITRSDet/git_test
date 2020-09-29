# fp16 settings
# fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    rbox_head=dict(
        type='S2ANetHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        align_conv_type='AlignConv',#[AlignConv,DCN,GA_DCN]
        align_conv_size=3,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        anchor_target_type='hbb_obb_rbox_overlap',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        anchor_target_type='obb_obb_rbox_overlap',
        anchor_inside_type='center',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.1),
    max_per_img=2000)
# dataset settings
dataset_type = 'DotaOBBDataset'
# dataset_type = 'CocoDataset'
# data_root = '/home/alex/s2anet/data/finish/'
data_root = '/home/alex/mmdetection/data/GF_SAR_ship/data/'
img_norm_cfg = dict(
    mean=[0], std=[1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(1000, 1000), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    # dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 1000),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(1000, 1000), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/test.json',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='SGD', lr=0.0004, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=1e-3, momentum=0, weight_decay=0.0005)
optimizer = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[6])
checkpoint_config = dict(interval=6)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/s2anet_r50_fpn_1x/'
load_from = None
# resume_from = '/home/alex/s2anet/tools/work_dirs/s2anet_r50_fpn_1x/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
# map: 0.7412306482856393
# classaps:  [89.10943329 82.83794745 48.37226726 71.1135539  78.10860328 78.39248929
#  87.24747997 90.82796582 84.89591577 85.64272706 60.36342839 62.59772683
#  65.26277211 69.12960291 57.94405909]
