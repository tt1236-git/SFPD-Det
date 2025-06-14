# dataset settings
dataset_type = 'DOTADataset'  # 使用DOTADataset类，因为DIOR-R数据集已转换为DOTA格式
data_root_ss = 'data/split_ss_diorr/'  # 单尺度数据路径
data_root_ms = 'data/split_ms_diorr/'  # 多尺度数据路径

angle_version_le90 = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version_le90),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version_le90),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root_ms + 'trainval/annfiles/',
        img_prefix=data_root_ms + 'trainval/images/',
        pipeline=train_pipeline,
        version=angle_version_le90),
    val=dict(
        type=dataset_type,
        ann_file=data_root_ss + 'val/annfiles/',
        img_prefix=data_root_ss + 'val/images/',
        pipeline=test_pipeline,
        version=angle_version_le90),
    test=dict(
        type=dataset_type,
        ann_file=data_root_ms + 'test/images/',
        img_prefix=data_root_ms + 'test/images/',
        pipeline=test_pipeline,
        version=angle_version_le90))