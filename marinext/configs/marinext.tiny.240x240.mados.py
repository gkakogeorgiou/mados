# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
            type="MSCAN",
            in_chans=11,
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            norm_cfg=dict(type="SyncBN", requires_grad=True)),
    decode_head=dict(
            type='LightHamHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            ham_channels=256,
            ham_kwargs=dict(MD_R=16),
            dropout_ratio=0.1,
            num_classes=15,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            align_corners=False))
