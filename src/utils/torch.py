import numpy as np
import os

import torch
import torch.nn as nn

from .data import find_files
from .other import set_seed as set_seed_np


def set_seed(seed):
    set_seed_np(seed)
    
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class LUT_Simple(nn.Module):

    def __init__(self, labels):
        super(LUT_Simple, self).__init__()
        labels = np.array(labels)
        if np.issubdtype(type(labels.flatten()[0]), np.integer):
            labels = labels.astype(np.long)

        self.register_buffer('labels', torch.tensor(labels))

    def forward(self, idxs):
        # idxs -> [batch_size]
        return self.labels[idxs.long()]
        # output -> [batch_size]
        
        
def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler=None):
    logger.info(
        f'==============> Resuming from {config.MODEL.RESUME}....................')

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.EPOCH_START = checkpoint['epoch'] + 1
        config.freeze()

        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    del checkpoint
    torch.cuda.empty_cache()


def delete_old_checkpoints(dir, n_check):
    if not isinstance(n_check, int) or n_check <= 0:
        return

    checkpoints = find_files(dir=dir, ext='pth')
    checkpoints = sorted(checkpoints, key=os.path.getmtime)

    if len(checkpoints) <= n_check:
        return

    for path in checkpoints[:-n_check]:
        # https://github.com/minimaxir/gpt-2-simple/issues/155#issuecomment-600258362
        open(path, 'w').close()
        os.remove(path)


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, logger):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'config': config,
    }

    if scaler is not None:
        save_state['scaler'] = scaler.state_dict()

    save_path = os.path.join(config.TRAIN.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f'{save_path} saving...')
    torch.save(save_state, save_path)
    logger.info(f'{save_path} saved!')

    delete_old_checkpoints(
        dir=config.TRAIN.OUTPUT, n_check=config.TRAIN.N_CHECK)     
        

def get_device(gpu_id=None, log_fn=None):
    use_log = log_fn is not None

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        if use_log:
            log_fn('GPU unavailable.')

        return device

    if gpu_id is None:
        gpu_id = 0

    device = torch.device(f'cuda:{gpu_id}')
    if use_log:
        log_fn(f'Using GPU-{gpu_id}: {torch.cuda.get_device_name(gpu_id)}.')

    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def accuracy(output, target, topk=(1, 5), *args, **kwargs):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = torch.cat(res, dim=0)

    return res


def tensor_linspace(start, end, steps):
    """Stolen from:
    https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    # start, end -> [...]
    device = start.device

    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=device)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=device)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    # out -> [..., steps]

    return out


def sample_uniform(low, high, size, device):
    if isinstance(size, int):
        size = (size,)
    return torch.empty(
        size=size, dtype=torch.float32, device=device).uniform_(low, high)
