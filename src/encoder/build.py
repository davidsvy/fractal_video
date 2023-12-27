import copy
import os

import torch
import torch.nn as nn

from . import tsm, x3d


def get_arch(name):
    name = name.lower()
    if name == 'tsm':
        arch = tsm
    elif name == 'x3d':
        arch = x3d

    else:
        raise ValueError(f'Unknown architecture: {name}')

    return arch


def build_encoder(config, mlp_head=False):
    arch = get_arch(config.MODEL.ARCH)
    model = arch.build(
        config=config, arch=config.MODEL.ARCH, mlp_head=mlp_head)

    # https://github.com/dougsouza/pytorch-sync-batchnorm-example
    if hasattr(config, 'MULTI_GPU') and config.MULTI_GPU:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


def init_sup(config, logger=None):
    path_ckpt = config.TRAIN.INIT_SUP
    if not os.path.isfile(path_ckpt):
        raise ValueError(f'No supervised checkpoint found at {path_ckpt}')

    assert not config.TRAIN.INIT_SSL
    log_fn = logger.info if logger is not None else print

    ckpt = torch.load(path_ckpt, map_location='cpu')
    config_ckpt = copy.deepcopy(ckpt['config'])
    model = build_encoder(config=config_ckpt, mlp_head=False)
    msg = model.load_state_dict(state_dict=ckpt['model'], strict=False)

    log_fn(msg)
    del ckpt
    torch.cuda.empty_cache()

    arch = get_arch(config_ckpt.MODEL.ARCH)
    model = arch.init_head(model=model, n_classes=config.MODEL.N_CLASSES)

    for param in model.parameters():
        param.requires_grad = True

    return model
