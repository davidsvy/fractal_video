import pytorchvideo.models.x3d
import torch.nn as nn

"""'M' & 'L' are taken from: 
https://github.com/facebookresearch/SlowFast/tree/main/configs/Kinetics

'XL' is made up.
"""

cfg_X3D_global = {

    'M':
    {
        'WIDTH_FACTOR': 2.0,
        'DEPTH_FACTOR': 2.2,
        'BOTTLENECK_FACTOR': 2.25,
        'HEAD_DIM_OUT': 2048,
        'STEM_DIM_IN': 12,
        'DROPOUT_RATE': 0.5,
        'SE_RATIO': 0.0625,
        'MLP_DIM': 512,
    },

    'L':
    {
        'WIDTH_FACTOR': 2.0,
        'DEPTH_FACTOR': 5.0,
        'BOTTLENECK_FACTOR': 2.25,
        'HEAD_DIM_OUT': 2048,
        'STEM_DIM_IN': 12,
        'DROPOUT_RATE': 0.5,
        'SE_RATIO': 0.0625,
        'MLP_DIM': 512,
    },

    'XL':
    {
        'WIDTH_FACTOR': 2.0,
        'DEPTH_FACTOR': 8.0,
        'BOTTLENECK_FACTOR': 2.25,
        'HEAD_DIM_OUT': 2048,
        'STEM_DIM_IN': 12,
        'DROPOUT_RATE': 0.5,
        'SE_RATIO': 0.0625,
        'MLP_DIM': 512,
    },
}


def build(config, mlp_head=False, *args, **kwargs):
    x3d_size = config.MODEL.X3D_SIZE
    if not x3d_size in cfg_X3D_global:
        raise ValueError(f'Unknown X3D size: {x3d_size}')
        
    cfg_X3D = cfg_X3D_global[x3d_size]

    model = pytorchvideo.models.x3d.create_x3d(
        input_channel=config.DATA.N_CHANNELS,
        input_clip_length=config.DATA.CLIP_LENGTH,
        input_crop_size=config.DATA.IMG_SIZE,
        model_num_class=config.MODEL.N_CLASSES,
        dropout_rate=cfg_X3D['DROPOUT_RATE'],
        width_factor=cfg_X3D['WIDTH_FACTOR'],
        depth_factor=cfg_X3D['DEPTH_FACTOR'],
        stem_dim_in=cfg_X3D['STEM_DIM_IN'],
        bottleneck_factor=cfg_X3D['BOTTLENECK_FACTOR'],
        se_ratio=cfg_X3D['SE_RATIO'],
        head_dim_out=cfg_X3D['HEAD_DIM_OUT'],
        head_activation=None,
    )

    if mlp_head:
        in_features = model.blocks[-1].proj.in_features

        model.blocks[-1].proj = nn.Sequential(
            nn.Linear(in_features, cfg_X3D['MLP_DIM']),
            nn.SiLU(),
            nn.Linear(cfg_X3D['MLP_DIM'], config.MODEL.N_CLASSES),
        )

    return model


def init_head(model, n_classes):
    old_head = model.blocks[-1].proj

    if isinstance(old_head, nn.Linear):
        in_features = old_head.in_features

    elif isinstance(old_head, nn.Sequential):
        in_features = old_head[0].in_features

    else:
        raise ValueError(f'Class of head is invalid: {type(old_head)}')

    model.blocks[-1].proj = nn.Linear(in_features, n_classes)
    del old_head

    return model


def remove_head(model):
    old_head = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Identity()
    del old_head

    return model


def get_head_dims(model):
    head = model.blocks[-1].proj

    if isinstance(head, nn.Linear):
        d_in = head.in_features
        d_out = head.out_features

    elif isinstance(head, nn.Sequential):
        d_in = head[0].in_features
        d_out = head[-1].out_features

    else:
        raise ValueError(f'Class of head is invalid: {type(head)}')

    return d_in, d_out
