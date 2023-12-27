import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import build_encoder


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722

    Based on:
    https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """

    def __init__(self, encoder, dim=128, K=2 ** 12, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.K, self.m, self.T = K, m, T
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        queue = F.normalize(torch.randn(K, dim), dim=1)
        self.register_buffer('queue', queue)
        self.queue_ptr = 0

        queue_file = torch.full(size=(1, K), fill_value=-1, dtype=torch.int32)
        self.register_buffer('queue_file', queue_file)

    def get_encoder(self):
        return copy.deepcopy(self.encoder_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, file_idxs):
        # keys -> [batch_size, d_moco]
        batch_size = keys.shape[0]

        self.queue[self.queue_ptr: self.queue_ptr + batch_size, :] = keys
        self.queue_file[
            0, self.queue_ptr: self.queue_ptr + batch_size] = file_idxs

        self.queue_ptr = (self.queue_ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # x -> [batch_size, ...]
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        x = x[idx_shuffle]

        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # x -> [batch_size, ...]
        # idx_unshuffle -> [batch_size]
        return x[idx_unshuffle]

    @torch.no_grad()
    def apply_encoder_k(self, k):
        k, idx_unshuffle = self._batch_shuffle(k)

        out_k = self.encoder_k(k)
        out_k = F.normalize(out_k, dim=1)
        out_k = self._batch_unshuffle(out_k, idx_unshuffle=idx_unshuffle)

        return out_k

    def forward(self, q, k, file_idxs, **kwargs):
        # q, k -> [batch_size, n_channels, n_frames, height, width]
        # file_idxs -> [batch_size]
        batch_size, device = q.shape[0], q.device

        out_q = self.encoder_q(q)
        out_q = F.normalize(out_q, dim=1)
        # out_q -> [batch_size, d_moco]

        self._momentum_update_key_encoder()

        out_k = self.apply_encoder_k(k)
        # out_k -> [batch_size, d_moco]

        l_pos = torch.einsum(
            'b d, b d -> b', out_q, out_k)[:, None] / self.T
        # l_pos -> [batch_size, 1]
        l_neg = torch.einsum(
            'b d, k d -> b k', out_q, self.queue.clone().detach()) / self.T
        # l_neg -> [batch_size, len_queue]

        mask = file_idxs[:, None] == self.queue_file
        # mask -> [batch_size, len_queue]
        l_neg = l_neg.masked_fill(mask, -float('inf'))
        # l_neg -> [batch_size, len_queue]

        logits = torch.cat([l_pos, l_neg], dim=1)
        # logits -> [batch_size, len_queue + 1]

        # labels: positive key indicators
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        # labels -> [batch_size]

        # dequeue and enqueue
        self._dequeue_and_enqueue(keys=out_k, file_idxs=file_idxs)

        return logits, labels


def build(config):
    assert config.MOCO.K % config.DATA.BATCH_SIZE == 0

    config = config.clone()
    config.defrost()
    config.MODEL.N_CLASSES = config.MOCO.DIM
    config.freeze()

    encoder = build_encoder(config=config, mlp_head=config.MOCO.MLP_HEAD)

    model = MoCo(
        encoder=encoder,
        dim=config.MOCO.DIM,
        K=config.MOCO.K,
        m=config.MOCO.M,
        T=config.MOCO.T,
    )

    fn_loss = nn.CrossEntropyLoss()

    return model, fn_loss
