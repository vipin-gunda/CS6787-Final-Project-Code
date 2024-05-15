_base_ = [

    '../_base_/models/seg_vit-b16.py',

    '../_base_/datasets/osmseg_generic.py', '../_base_/default_runtime.py',

    '../_base_/schedules/schedule_generic.py'

]

#checkpoint = '/home/vg245/SegVit/checkpoints/jx_vit_base_p16_384-83fb41ba.pth'
#checkpoint = "/home/vg245/SegVit/checkpoints/ade_51.3.pth"
# checkpoint = "/home/vg245/SegVit/checkpoints/graft_naip_patch.pth"
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
img_size = 224
in_channels = 768  # Adjusted input channels to match the output of the VisionTransformer backbone
out_indices = [5, 7, 11]
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,  # Changed back to 3 channels for RGB input images
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=out_indices,
        final_norm=False,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=False,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='ATMHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        num_classes=32,
        num_layers=3,
        num_heads=16,
        use_stages=len(out_indices),
        embed_dims=512,
        loss_decode=dict(
            type='ATMLoss', num_classes=32, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(),
)

data = dict(samples_per_gpu=3)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
#                                                  'ln': dict(decay_mult=0.),
#                                                  'head': dict(lr_mult=10.),
#                                                  }))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-4, by_epoch=False)