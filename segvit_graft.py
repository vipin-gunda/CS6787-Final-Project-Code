_base_ = [

    '../_base_/models/seg_vit-b16.py',

    '../_base_/datasets/osmseg.py', '../_base_/default_runtime.py',

    '../_base_/schedules/schedule_160k.py'

]

#checkpoint = '/home/vg245/SegVit/checkpoints/jx_vit_base_p16_384-83fb41ba.pth'
#checkpoint = "/home/vg245/SegVit/checkpoints/ade_51.3.pth"
checkpoint = "/home/vg245/SegVit/checkpoints/graft_naip_patch.pth"
img_size = 224
out_indices = [5, 7, 11]

model = dict(
    pretrained=checkpoint,
    decode_head=dict(
        #changed
        num_classes=32,
        loss_decode=dict(
            # Changed ATMLoss to CrossEntropyLoss
            type='ATMLoss', num_classes=32, dec_layers=len(out_indices), loss_weight=1.0), num_heads=16, embed_dims = 512

    )
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