import torch
import torch.nn as nn

from src.transform.compose import Transform_Outer_Train
from src.transform.mixup import Mixup_Background


class Transform_Contrastive(nn.Module):

    def __init__(
        self, img_size, easy_k=False, randaugment_m=9, randaugment_n=2, n_steps=1,
        back=True, prob_perspective=0.0, prob_scale=0.0, prob_shift=0.0, prob_clone=0.0,
        prob_zoom=0.0, prob_shake=0.0,
    ):
        super(Transform_Contrastive, self).__init__()

        self.back = back

        self.mixup_q = Mixup_Background(
            img_size=img_size,
            n_steps=n_steps,
            prob_scale=prob_scale,
            prob_shift=prob_shift,
            prob_clone=prob_clone,
        )

        self.transform_q = Transform_Outer_Train(
            randaugment_m=randaugment_m,
            randaugment_n=randaugment_n,
            n_steps=n_steps,
            hor_flip=True,
            prob_blur=0.5,
            prob_perspective=prob_perspective,
            prob_shift=prob_shift,
            prob_zoom=prob_zoom,
            prob_shake=prob_shake,
        )

        if not easy_k:
            self.mixup_k = self.mixup_q
            self.transform_k = self.transform_q

        else:
            self.mixup_k = Mixup_Background(
                img_size=img_size,
                n_steps=n_steps,
                prob_scale=prob_scale,
                prob_shift=0,
                prob_clone=0,
            )

            self.transform_k = Transform_Outer_Train(
                randaugment_m=randaugment_m,
                randaugment_n=randaugment_n,
                n_steps=n_steps,
                hor_flip=True,
                prob_blur=0.5,
                prob_perspective=0,
                prob_shift=0,
                prob_zoom=0,
                prob_shake=0,
            )

    def temporal_reverse(self, video):
        # video -> [B, 2, C, T, H, W]
        if torch.rand(1).item() < 0.5:
            video = torch.flip(video, dims=(-3,))

        return video

    def forward(self, video, step):
        # video -> [B, 2, C, T, H, W]

        video = self.temporal_reverse(video)
        # video -> [B, 2, C, T, H, W]
        q, k = video[:, 0], video[:, 1]

        if self.back:
            q = self.mixup_q(q, step=step)
            k = self.mixup_k(k, step=step)

        q = self.transform_q(q, step=step)
        k = self.transform_k(k, step=step)

        return q, k
