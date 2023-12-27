import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import build_encoder


class BYOL(nn.Module):
    """Based on:
    https://arxiv.org/pdf/2006.07733.pdf

    The paper uses 2 slightly different families of augmentations,
    but this implementation only uses one.
    """

    def __init__(self, encoder, d_output, d_hidden, ema_weight, total_steps):
        super(BYOL, self).__init__()
        assert isinstance(ema_weight, float) and 0 < ema_weight < 1
        assert isinstance(total_steps, int) and total_steps > 0

        self.ema_weight, self.total_steps = ema_weight, total_steps

        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(d_output, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_output)
        )

    def get_encoder(self):
        return copy.deepcopy(self.online_encoder)

    @torch.no_grad()
    def update_ema(self, step):
        ema_weight = math.cos(math.pi * step / self.total_steps)
        ema_weight = 1 - (1 - self.ema_weight) * (ema_weight + 1) / 2

        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data = p_t.data * ema_weight + p_o.data * (1. - ema_weight)

    @torch.no_grad()
    def _batch_shuffle(self, x):
        # x -> [batch_size, ...]
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        x = x[idx_shuffle]

        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        # x -> [batch_size, ...]
        # idx_unshuffle -> [batch_size]
        return x[idx_unshuffle]

    def forward(self, q, k, step, **kwargs):
        out_online_q = self.mlp(self.online_encoder(q))
        out_online_k = self.mlp(self.online_encoder(k))

        logits = F.normalize(
            torch.cat([out_online_q, out_online_k], dim=0), dim=1)

        with torch.no_grad():
            self.update_ema(step)

            q, idx_unshuffle = self._batch_shuffle(q)
            out_target_q = self.target_encoder(q).detach()
            out_target_q = self._batch_unshuffle(
                out_target_q, idx_unshuffle=idx_unshuffle)

            k, idx_unshuffle = self._batch_shuffle(k)
            out_target_k = self.target_encoder(k).detach()
            out_target_k = self._batch_unshuffle(
                out_target_k, idx_unshuffle=idx_unshuffle)

            labels = F.normalize(
                torch.cat([out_target_k, out_target_q], dim=0), dim=1).clone().detach()

        return logits, labels


def loss_fn_byol(input, target):
    # input, target -> [B, D]
    return 2 * (1 - torch.einsum('b d, b d -> b', input, target).mean())


def build(config):
    config = config.clone()
    config.defrost()
    config.MODEL.N_CLASSES = config.BYOL.D_OUTPUT
    config.freeze()

    encoder = build_encoder(config=config, mlp_head=True)

    model = BYOL(
        encoder=encoder,
        d_output=config.BYOL.D_OUTPUT,
        d_hidden=config.BYOL.D_HIDDEN,
        ema_weight=config.BYOL.EMA_WEIGHT,
        total_steps=config.TRAIN.EPOCHS * config.STEPS_PER_EPOCH,
    )

    fn_loss = loss_fn_byol

    return model, fn_loss
