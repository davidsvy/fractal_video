import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Random_Perspective(T.RandomPerspective):

    def sample_points(self, width, height):
        h_half = height // 2
        w_half = width // 2

        h_start = int(self.distortion_scale * h_half)
        w_start = int(self.distortion_scale * w_half)

        h_end = height - int(self.distortion_scale * h_half) - 1
        w_end = width - int(self.distortion_scale * w_half) - 1

        topleft = [
            random.randint(0, w_start), random.randint(0, h_start)]
        topright = [
            random.randint(w_end, width - 1), random.randint(0, h_start)]
        botright = [
            random.randint(w_end, width - 1), random.randint(h_end, height - 1)]
        botleft = [
            random.randint(0, w_start), random.randint(h_end, height - 1)]

        points_end = [topleft, topright, botright, botleft]
        points_start = [
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

        return points_start, points_end

    def forward(self, video):
        # video -> [B, C, T, H, W]
        B, C, T, H, W = video.shape
        device = video.device
        mask = torch.rand(B, device=device) < self.p
        B_tr = mask.sum().item()

        if B_tr == 0:
            return video

        video_tr = torch.flatten(video[mask], start_dim=1, end_dim=2)
        # video_tr -> [B_tr, C * T, H, W]

        points_start, points_end = self.sample_points(width=W, height=H)
        video_tr = TF.perspective(
            video_tr, startpoints=points_start, endpoints=points_end,
            interpolation=self.interpolation, fill=0.0)
        # video_tr -> [B_tr, C * T, H, W]
        video_tr = F.interpolate(video_tr, scale_factor=1.4, mode='bilinear')
        video_tr = TF.center_crop(video_tr, output_size=(H, W))
        video_tr = video_tr.reshape(B_tr, C, T, H, W)
        # video_tr -> [B_tr, C, T, H, W]

        video[mask] = video_tr

        return video
    
    
def short_side_scale(x, size, interpolation='bilinear'):
    """Same as:
    https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html#short_side_scale
    but supports other interpolation modes.
    """

    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return nn.functional.interpolate(
        x, size=(new_h, new_w), mode=interpolation
    )


class Crop_Center(nn.Module):

    def __init__(self, crop_size, interpolation='bicubic', resize=True, crop_inc=True):
        super(Crop_Center, self).__init__()
        self.crop_size, self.interpolation = crop_size, interpolation
        self.resize_dim = int(
            (256 / 224) * crop_size) if crop_inc else crop_size
        self.resize = resize

    def forward(self, x):
        # x -> [C, T, H, W]
        if self.resize:
            x = short_side_scale(
                x, size=self.resize_dim, interpolation=self.interpolation)
            x = T.functional.center_crop(x, output_size=self.crop_size)

        else:
            height, width = x.shape[-2:]
            start_h = int(math.ceil((height - self.crop_size) / 2))
            start_w = int(math.ceil((width - self.crop_size) / 2))
            x = x[
                ..., start_h: start_h + self.crop_size, start_w: start_w + self.crop_size]

        return x
