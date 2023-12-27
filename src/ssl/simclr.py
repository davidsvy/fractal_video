import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import build_encoder


class SimCLR(nn.Module):
    """Based on:
    https://arxiv.org/pdf/2008.03800.pdf
    """

    def __init__(self, encoder, T=0.07):
        super(SimCLR, self).__init__()

        self.T = T
        self.encoder = encoder

    def get_encoder(self):
        return copy.deepcopy(self.encoder)

    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # x -> [batch_size, ...]
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        x = x[idx_shuffle]

        return x, idx_unshuffle

    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # x -> [batch_size, ...]
        # idx_unshuffle -> [batch_size]
        return x[idx_unshuffle]

    def forward(self, q, k, **kwargs):
        # q, k -> [batch_size, ...]
        device = q.device
        bs = q.shape[0]

        q = self.encoder(q)
        # q -> [batch_size, d_out]

        k, idx_unshuffle = self._batch_shuffle(k)
        k = self.encoder(k)
        k = self._batch_unshuffle(x=k, idx_unshuffle=idx_unshuffle)
        # k -> [batch_size, d_out]

        out = F.normalize(torch.cat([q, k], dim=0), dim=1)
        # out -> [2 * batch_size, d_out]

        logits = torch.einsum('x d, z d -> x z', out, out) / self.T
        # logits -> [2 * batch_size, 2 * batch_size]

        mask = ~torch.eye(2 * bs, dtype=torch.bool, device=device)
        # mask -> [2 * batch_size, 2 * batch_size]

        logits = logits[mask].view(2 * bs, -1)
        # logits -> [2 * batch_size, 2 * batch_size - 1]

        labels = torch.cat(
            [
                torch.arange(bs - 1, 2 * bs - 1, device=device),
                torch.arange(0, bs, device=device),
            ], dim=0
        ).long()
        # labels -> [2 * batch_size]

        return logits, labels


def build(config):
    config = config.clone()
    config.defrost()
    config.MODEL.N_CLASSES = config.SIMCLR.DIM
    config.freeze()

    encoder = build_encoder(config=config, mlp_head=config.SIMCLR.MLP_HEAD)
    model = SimCLR(encoder=encoder, T=config.SIMCLR.T)

    fn_loss = nn.CrossEntropyLoss()

    return model, fn_loss
