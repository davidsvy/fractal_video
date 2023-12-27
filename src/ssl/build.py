import copy
import os

import torch

from . import byol, moco, simclr
from ..encoder.build import get_arch


def build_ssl_model(config):
    scheme = config.TRAIN.SSL_SCHEME.lower()

    if scheme == 'byol':
        model, fn_loss = byol.build(config)

    elif scheme == 'moco':
        model, fn_loss = moco.build(config)

    elif scheme == 'simclr':
        model, fn_loss = simclr.build(config)

    else:
        raise ValueError(f'Unknown self-supervised scheme: {scheme}')

    return model, fn_loss


def init_ssl(config, logger=None):
    path_ckpt = config.TRAIN.INIT_SSL
    if not os.path.isfile(path_ckpt):
        raise ValueError(f'No self-supervised checkpoint found at {path_ckpt}')

    assert not config.TRAIN.INIT_SUP
    log_fn = logger.info if logger is not None else print

    ckpt = torch.load(path_ckpt, map_location='cpu')
    config_ckpt = copy.deepcopy(ckpt['config'])
    model, _ = build_ssl_model(config_ckpt)
    msg = model.load_state_dict(state_dict=ckpt['model'], strict=False)

    model = model.get_encoder()

    log_fn(msg)
    del ckpt
    torch.cuda.empty_cache()

    arch = get_arch(config_ckpt.MODEL.ARCH)
    model = arch.init_head(model=model, n_classes=config.MODEL.N_CLASSES)

    for param in model.parameters():
        param.requires_grad = True

    return model
