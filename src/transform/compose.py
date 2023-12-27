import pytorchvideo.transforms as Tv
import torch
import torch.nn as nn
import torchvision.transforms as T

from src.transform.camera import Transform_Camera
from src.transform.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.transform.spatial import Crop_Center, Random_Perspective


def transform_inner_train(crop_size=112, min_scale=0.5, interp='bicubic'):
    crop_tr = Tv.RandomResizedCrop(
        target_height=crop_size,
        target_width=crop_size,
        scale=(min_scale, 1),
        aspect_ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=interp,
    )
    
    transform = T.Compose([
        Tv.ConvertUint8ToFloat(),
        crop_tr,
    ])

    return transform


def transform_inner_val(crop_size=112, resize=True, interp='bicubic', crop_inc=True):

    return T.Compose([
        Tv.ConvertUint8ToFloat(),
        Crop_Center(
            crop_size=crop_size,
            interpolation=interp,
            resize=resize,
            crop_inc=crop_inc,
        ),
        Tv.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ])


class Transform_Outer_Train(nn.Module):

    def __init__(
        self, randaugment_m=9, randaugment_n=2, n_steps=1, hor_flip=True, prob_blur=0.5,
        prob_perspective=0.0, prob_shift=0.0, prob_zoom=0.0, prob_shake=0.0,
    ):
        super(Transform_Outer_Train, self).__init__()

        self.use_perspective = prob_perspective > 0
        self.transform_perspective = Random_Perspective(
            p=prob_perspective,
            distortion_scale=0.7,
        )

        self.use_camera = prob_shift > 0 or prob_zoom > 0 or prob_shake > 0
        self.transform_camera = Transform_Camera(
            prob_shift=prob_shift,
            prob_zoom=prob_zoom,
            prob_shake=prob_shake,
            n_steps=n_steps,
        )

        transform = []

        if hor_flip:
            transform.append(T.RandomHorizontalFlip())

        if randaugment_m > 0:
            transform += [
                Tv.Permute((1, 0, 2, 3)),
                Tv.RandAugment(
                    magnitude=randaugment_m,
                    num_layers=randaugment_n,
                    transform_hparas={'fill': (0., 0., 0.)}
                ),
                Tv.Permute((1, 0, 2, 3)),
            ]

        if prob_blur > 0:
            transform.append(T.RandomApply([T.GaussianBlur(3)], p=prob_blur))

        transform.append(
            Tv.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        )

        self.transform = T.Compose(transform)

    def forward(self, video, step):
        # video -> [B, C, T, H, W]

        # applied in parallel for all samples in batch
        if self.use_perspective:
            video = self.transform_perspective(video)

        if self.use_camera:
            video = self.transform_camera(video, step=step)

        # applied seqentially for all samples in batch
        video = torch.stack([self.transform(v) for v in video], dim=0)

        return video


def transform_image_train(crop_size=112, min_scale=0.5):

    return T.RandomResizedCrop(
        size=crop_size,
        scale=(min_scale, 1.0),
        interpolation=T.InterpolationMode.BICUBIC,
    )
    """
    return T.Compose([
        T.RandomRotation(
            degrees=360,
            interpolation=T.InterpolationMode.BILINEAR,
            expand=False,
        ),
        T.RandomResizedCrop(
            size=crop_size,
            scale=(min_scale, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
    ])
    """
