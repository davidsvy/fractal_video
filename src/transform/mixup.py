import random

import torch
import torch.nn as nn
import torchvision.transforms as T

from .camera import Transform_Background, Transform_Foreground
from ..utils.torch import sample_uniform


class Mixup_Background(nn.Module):

    def __init__(self, img_size, n_steps=1, prob_scale=0.0, prob_shift=0.0, prob_clone=0.0):
        super(Mixup_Background, self).__init__()

        self.transform_background_static = T.RandomResizedCrop(
            size=img_size,
            scale=(0.6, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        )

        self.use_tr_back = prob_shift > 0
        self.transform_background = Transform_Background(
            prob=prob_shift,
            n_steps=n_steps,
        )

        self.transform_foreground = Transform_Foreground(
            prob_scale=prob_scale,
            prob_shift=prob_shift,
            prob_clone=prob_clone,
            n_steps=n_steps,
        )

        self.alpha_mixup = (0.25, 0.75)
        self.alpha_back = (0.25, 0.55)
        self.exp = (0.6, 1.0)

        self.prob_back_dynamic = 0.2
        self.prob_back_max = 0.05

    def sample_background(self, video):
        # video -> [B, C, T, H, W]
        B, _, T, *_ = video.shape
        device = video.device

        mask = torch.rand(B, device=device) < self.prob_back_dynamic
        B_dynamic = mask.sum().item()
        B_static = B - B_dynamic
        back_static, back_dynamic = None, None

        if B_static > 0:
            back_static = []
            for _ in range(2):
                idxs_batch = torch.randint(
                    low=0, high=B, size=(B_static,), device=device)
                idxs_frame = torch.randint(
                    low=0, high=T, size=(B_static,), device=device)
                # idxs_batch, idxs_frame -> [B_static]
                back_static.append(video[idxs_batch, :, idxs_frame])

            back_static = torch.amax(torch.stack(back_static, dim=0), dim=0)
            # back_static -> [B_static, C, H, W]

        if B_dynamic > 0:
            idxs_batch = torch.randint(
                low=0, high=B, size=(B_dynamic,), device=device)
            # idxs_batch -> [B_dynamic]

            time_min, time_max = 2, max(2, int(0.25 * T))
            time = random.randint(time_min, time_max)
            time_start = random.randint(0, T - time)
            
            idxs_time = torch.randint(-1, 2, size=(T,))
            idxs_time[0] = 0
            idxs_time = torch.clip(idxs_time.cumsum(dim=0), 0, time - 1)
            idxs_time = idxs_time + time_start
            # idxs_time -> [T]
            back_dynamic = video[idxs_batch][:, :, idxs_time]
            # back_dynamic -> [B_dynamic, C, T, H, W]

        return back_static, back_dynamic

    def mix_background(self, video, back_static, back_dynamic):
        """Adds background frames sampled from the video batch itself.

        Inspired by:
        https://arxiv.org/pdf/2009.05769.pdf
        https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Motion-Aware_Contrastive_Video_Representation_Learning_via_Foreground-Background_Merging_CVPR_2022_paper.pdf
        """
        # video -> [B, C, T, H, W]
        # back_static -> [B_static, C, T, H, W] or None
        # back_dynamic -> [B_dynamic, C, 1, H, W] or None
        B, device = video.shape[0], video.device

        if back_static is None:
            back = back_dynamic
        elif back_dynamic is None:
            back = back_static
        else:
            back = torch.cat([back_static, back_dynamic], axis=0)
            idxs_batch = torch.randperm(B, device=device)
            back = back[idxs_batch]

        mask_max = torch.rand(B, device=device) < self.prob_back_max
        B_add = B - mask_max.sum().item()

        video = torch.clip(video, 0, 1)
        back = torch.clip(back, 0, 1)

        if B_add > 0:
            size_alpha = (B_add, 1, 1, 1, 1)
            alpha = sample_uniform(self.alpha_back[0], self.alpha_back[1], size=size_alpha, device=device)

            video[:B_add] = ((1 - alpha) * video[:B_add] ** 0.75 + alpha * back[:B_add] ** 0.75)

            exp = sample_uniform(self.exp[0], self.exp[1], size=size_alpha, device=device)
            video[:B_add] = torch.clip(video[:B_add], 0, 1) ** exp

        if B_add < B:
            video[B_add:] = torch.maximum(video[B_add:], back[B_add:])

        return video

    def forward(self, video, step):
        """Adds background frames sampled from the video batch itself.

        Inspired by:
        https://arxiv.org/pdf/2009.05769.pdf
        https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Motion-Aware_Contrastive_Video_Representation_Learning_via_Foreground-Background_Merging_CVPR_2022_paper.pdf
        """
        # video -> [B, C, T, H, W]
        T = video.shape[2]
        #video = torch.clip(video, min=0, max=1) ** 0.75
        back_static, back_dynamic = self.sample_background(video)
        # back_static -> [B_static, C, H, W] or None
        # back_dynamic -> [B_dynamic, C, T, H, W] or None

        if back_static is not None:
            back_static = self.transform_background_static(back_static)

            if self.use_tr_back:
                back_static = self.transform_background(
                    back_static, n_frames=T, step=step)
                # back_static -> [B_static, C, T, H, W]
            else:
                back_static = torch.repeat_interleave(
                    back_static[:, :, None], repeats=T, dim=-3)
                # back_static -> [B_static, C, T, H, W]

        video = self.transform_foreground(video, step=step)

        video = self.mix_background(
            video=video, back_static=back_static, back_dynamic=back_dynamic)

        return video
