_base_ = ["../configs/_base_/default_runtime.py"]

# misc custom setting
num_worker = 5
batch_size = 3
mix_prob = 0.8
enable_amp = True
empty_cache = False
find_unused_parameters = False

# scheduler settings
epoch = 45
eval_epoch = 45
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
param_dicts = [dict(keyword='block', lr=0.0002)]
empty_cache_per_epoch=True

# trainer/tester setting
train = dict(type='DefaultTrainer')
test = dict(type='PartSegTester', verbose=True)

# model settings
model = dict(
    type='DefaultSegmentorV2',
    num_classes=5,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=6, # 4
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])

# dataset settings
dataset_type = 'BHDataset'  # [bh_01_AJ_14, bh_01_AQ_21, bh_01_AL_14, bh_01_AL_13, bh_01_AK_14] 
data_root = '/home/leena/ptv3/Pointcept/Customed/data'
ignore_index = -1
names = ['nothing', 'ground', 'road', 'low_vegetation', 'high_vegetation']

weight = None   # "/home/leena/ptv3/Pointcept/Customed/model_best_nu.pth"
resume = False  # True
save_path = '/home/leena/ptv3/Pointcept/Customed/seg_model'
test_only = False
seed = 47

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split='train',
        data_root=data_root,
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'color', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'color', 'segment'),
                feat_keys=('grid_coord', 'color'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type=dataset_type,
        split='test',
        data_root=data_root,
        transform=[
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'color', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'color', 'segment'),
                feat_keys=('grid_coord', 'color'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type=dataset_type,
        split='test',
        data_root=data_root,
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'color', 'segment'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'color')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'color'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.9, 0.9]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.1, 1.1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]]),
        ignore_index=-1))

evaluate = True
sync_bn = False

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]

