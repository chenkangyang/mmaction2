#! TINY 配置
model = dict(
    # type='FastRCNN',
    type='ViDETR',
    backbone=dict(
        type='ViDETREncoder',
        det_encoder=dict(
            type="ViTDetEncoder",
            img_size=224,
            num_layers=12,
            embed_dim=24,
            num_heads=12,
            patch_size=16,
            in_channels=3,
            feedforward_channels=24,
            drop_rate=0.,
            attn_drop_rate=0.,
            hybrid_backbone=None,
            norm_cfg=dict(type='LN'),
            act_cfg=dict(type='GELU'),
            num_fcs=2,
            det_token_num=100,
            det_img_size=(800, 1344),
            mid_pe_size=None,
            use_checkpoint=False,
            pretrained=None),
        spacetime_encoder=dict(
            type="TimesFormerEncoder",
            num_frames=4, #!
            img_size=256, #!
            patch_size=16, #!
            pretrained=None,
            embed_dims=24,
            num_heads=12,
            num_transformer_layers=12,
            in_channels=3,
            dropout_ratio=0.,
            transformer_layers=None,
            attention_type='divided_space_time',
            norm_cfg=dict(type='LN', eps=1e-6),
        ),
        actor_encoder=dict(
            type="ACFormer",
            embed_dims=24,
            num_heads=12,
            transformer_layers=None,
            num_transformer_layers=12,
            num_actor_token=100,
        )
    ),
    obj_head=dict(
        type='ViDETRObjHead',
        num_classes=1,
        embed_dims=24,
        num_obj_token=100,
        sync_cls_avg_factor=False,
        reg_ffn=dict(
            type='FFN',
            embed_dims=24,
            feedforward_channels=24,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True),
            add_identity=False,
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            class_weight=1.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssignerForObjectDetetion',
                cls_cost=dict(type='ClassificationCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
    ),
    act_head=dict(
        type='ViDETRActHead',
        num_classes=81,
        embed_dims=24,
        num_actor_token=100,
        sync_cls_avg_factor=False,
        reg_ffn=dict(
            type='FFN',
            embed_dims=24,
            feedforward_channels=24,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True),
            add_identity=False,
        ),
        loss_cls=dict(
            type='MultiLabelFocalLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            class_weight=1.0,
            focal_gamma=0.,
            focal_alpha=1.,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssignerForActorDetetion',
                cls_cost=dict(type='MultiClassificationCost', weight=5.)
            ),
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
    ),
)

dataset_type = 'AVADatasetForViDETR'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'

label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

# proposal_file_train = None
# proposal_file_val = None


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='KeyframeAug', mode='train'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='FormatViDETRInput'),
    dict(type='ToTensor', keys=['clip', 'keyframe', 'gt_act_bboxes', 'gt_act_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['gt_obj_bboxes', 'gt_obj_labels', 'gt_act_bboxes', 'gt_act_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['clip', 'keyframe', 'gt_obj_bboxes', 'gt_obj_labels', 'gt_act_bboxes', 'gt_act_labels'],
        meta_keys=['entity_ids']),
    dict(type='Rename', mapping=dict(img_metas='clip_metas')),
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        data_prefix=data_root))
data['test'] = data['val']
# optimizer
optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1)
total_epochs = 10
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1)
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ViDETR_DEBUG'  # noqa: E501
# load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'  # noqa: E501
load_from = None
resume_from = None
find_unused_parameters = False
