import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


from ..utils.torch import sample_uniform


class Transform_Background(nn.Module):

    def __init__(self, prob, n_steps):
        super(Transform_Background, self).__init__()
        assert 0 <= prob <= 1
        assert n_steps > 0
        self.prob, self.n_steps = prob, n_steps

        self.speed = (0.2, 1.0)
        self.slope_speed = (self.speed[1] - self.speed[0]) / n_steps

        self.time = (0.1, 1.0)
        self.slope_time = (self.time[1] - self.time[0]) / n_steps

    def sample_params_shift(self, n_frames, step, res):
        if step >= self.n_steps:
            speed_max = self.speed[1]
            time_max = self.time[1]

        else:
            speed_max = self.speed[0] + step * self.slope_speed
            time_max = self.time[0] + step * self.slope_time

        speed_max = max(1, speed_max * res / n_frames)
        speed_min = max(1, self.speed[0] * res / n_frames)
        speed = random.uniform(speed_min, speed_max)

        time_max = min(n_frames, max(2, int(time_max * n_frames)))
        time_min = min(n_frames, max(2, int(self.time[0] * n_frames)))
        time = random.randint(time_min, time_max)

        angle = random.uniform(0, 2 * math.pi)

        return speed, time, angle

    def forward_shift(self, image, n_frames, step):
        # image -> [B, C, H, W]
        B, C, H, W = image.shape
        device = image.device

        speed, time, angle = self.sample_params_shift(
            n_frames=n_frames, step=step, res=H)
        # angle, speed, time -> [1]

        curve = torch.arange(time, dtype=torch.float32, device=device) * speed
        curve_h = (math.sin(angle) * curve).to(torch.int32)
        curve_w = (math.cos(angle) * curve).to(torch.int32)
        # curve_h, curve_w -> [n_time]

        end_h = curve_h[-1].item()
        end_w = curve_w[-1].item()

        min_h = min(0, end_h)
        min_w = min(0, end_w)

        diff_h = abs(end_h) + 1
        diff_w = abs(end_w) + 1
        diff = max(diff_h, diff_w)

        size_interp = (H + diff, W + diff)
        image = F.interpolate(image, size=size_interp, mode='bicubic')
        image = torch.clip(image, 0, 1)
        # image -> [B, C, H + D, W + D]

        curve_h = curve_h - min_h + (diff - diff_h) // 2
        curve_w = curve_w - min_w + (diff - diff_w) // 2

        video = torch.stack(
            [image[..., h: h + H, w: w + W]
                for h, w in zip(curve_h, curve_w)],
            dim=-3,
        )
        # video -> [B, C, N_time, H, W]

        n_empty = n_frames - time
        if n_empty == 0:
            return video

        pad_left = torch.randint(0, n_empty, size=(B,))
        pad_list = [
            (0, 0, 0, 0, left.item(), n_empty - left.item()) for left in pad_left]

        video = torch.stack(
            [F.pad(v, pad=pad, mode='replicate')
             for v, pad in zip(video, pad_list)],
            dim=0,
        )

        return video

    def forward(self, image, n_frames, step):
        # image -> [B, C, H, W]
        B, C, H, W = image.shape
        device = image.device
        out = torch.empty(
            size=(B, C, n_frames, H, W), device=device, dtype=image.dtype)
        mask_inactive = torch.rand(B, device=device) > self.prob
        mask_active = torch.logical_not(mask_inactive)
        B_inactive = mask_inactive.sum().item()

        if B_inactive > 0:
            out[mask_inactive] = torch.repeat_interleave(
                image[mask_inactive, :, None, ...], repeats=n_frames, dim=-3)

        if B_inactive < B:
            out[mask_active] = self.forward_shift(
                image=image[mask_active], n_frames=n_frames, step=step)

        return out


class Transform_Foreground(nn.Module):

    def __init__(self, prob_scale, prob_shift, prob_clone, n_steps=1, pad_value=0.0):
        super(Transform_Foreground, self).__init__()
        assert n_steps > 0
        assert 0 <= prob_scale <= 1
        assert 0 <= prob_shift <= 1
        assert 0 <= prob_clone <= 1

        self.prob_scale = prob_scale
        self.prob_shift, self.prob_clone = prob_shift, prob_clone
        self.n_steps = n_steps
        self.pad_value = pad_value

        self.scale = (0.3, 1.0)
        self.scale_clone = (0.2, 0.7)
        self.slope_scale = (self.scale[0] - self.scale[1]) / n_steps

        self.speed = (0.2, 1.0)
        self.slope_speed = (self.speed[1] - self.speed[0]) / n_steps

        self.time = (0.1, 1.0)
        self.slope_time = (self.time[1] - self.time[0]) / n_steps

        self.interp_rot = T.InterpolationMode.BILINEAR

        self.n_clone = 1

    def sample_params_scale(self, step, size, device):
        if step >= self.n_steps:
            scale_min = self.scale[0]

            mask_clone = torch.rand(size, device=device) < self.prob_clone
            size_clone = mask_clone.sum().item()

            scale = sample_uniform(
                scale_min, self.scale[1], size=(size, 2), device=device)

            if size_clone > 0:
                scale[mask_clone] = sample_uniform(
                    self.scale_clone[0], self.scale_clone[1], size=(size_clone, 2), device=device)

        else:
            scale_min = self.scale[1] + step * self.slope_scale

            mask_clone = torch.full(
                (size,), fill_value=False, dtype=torch.bool)

            scale = sample_uniform(
                scale_min, self.scale[1], size=(size, 2), device=device)

        return scale, mask_clone

    def transform_clone(self, video):
        # video -> [C, T, H, W]
        T, device = video.shape[-3], video.device

        if random.random() < 0.5:
            video = torch.flip(video, dims=(-1,))

        angle = random.uniform(-60, 60)
        video = TF.rotate(
            video, angle=angle, interpolation=self.interp_rot)

        offset = random.randint(0, max(1, int(0.25 * T)))

        if offset == 0:
            return video

        idxs_time = torch.empty(size=(T,), dtype=torch.int64, device=device)

        if random.random() < 0.5:
            idxs_time[:offset] = 0
            idxs_time[offset:] = torch.arange(0, T - offset, device=device)

        else:
            idxs_time[:-offset] = torch.arange(offset, T, device=device)
            idxs_time[-offset:] = T - 1

        video = video[:, idxs_time]

        return video

    def forward_scale(self, x, step):
        # x -> [B, ..., H, W]

        B, *_, H, W = x.shape
        device = x.device
        scale, mask_clone = self.sample_params_scale(
            step=step, size=B, device=device)
        # scale -> [B, 2]
        # mask_clone -> [B]
        for idx in range(B):
            _scale = (scale[idx, 0].item(), scale[idx, 1].item())

            _x = F.interpolate(
                x[idx], scale_factor=_scale, mode='bicubic')

            h, w = _x.shape[-2:]
            h_start = random.randint(0, H - h)
            w_start = random.randint(0, W - w)

            x[idx] = self.pad_value
            x[idx, ..., h_start: h_start + h, w_start: w_start + w] = _x

            if mask_clone[idx].item():
                for _ in range(random.randint(1, self.n_clone)):
                    h_start = random.randint(0, H - h)
                    w_start = random.randint(0, W - w)

                    __x = self.transform_clone(_x.clone())
                    x[idx, ..., h_start: h_start + h, w_start: w_start + w] = torch.maximum(
                        x[idx, ..., h_start: h_start + h, w_start: w_start + w],
                        __x
                    )

        return x

    def sample_params_shift(self, n_frames, step, res):
        if step >= self.n_steps:
            speed_max = self.speed[1]
            time_max = self.time[1]

        else:
            speed_max = self.speed[0] + step * self.slope_speed
            time_max = self.time[0] + step * self.slope_time

        speed_max = max(1, speed_max * res / n_frames)
        speed_min = max(1, self.speed[0] * res / n_frames)
        speed = random.uniform(speed_min, speed_max)

        time_max = min(n_frames, max(2, int(time_max * n_frames)))
        time_min = min(n_frames, max(2, int(self.time[0] * n_frames)))
        time = random.randint(time_min, time_max)

        angle = random.uniform(0, 2 * math.pi)

        return speed, time, angle

    def forward_shift(self, video, step):
        # video -> [B, C, T, H, W]
        B, C, T, H, W = video.shape
        device = video.device

        speed, time, angle = self.sample_params_shift(
            n_frames=T, res=H, step=step)
        time_before = random.randint(0, T - time)
        # angle, speed, time -> [1]

        curve = torch.arange(time, dtype=torch.float32, device=device) * speed
        curve_h = (math.sin(angle) * curve).to(torch.int32)
        curve_w = (math.cos(angle) * curve).to(torch.int32)
        # curve_h, curve_w, curve_t -> [n_time]

        end_h = curve_h[-1].item()
        end_w = curve_w[-1].item()

        min_h = min(0, end_h)
        min_w = min(0, end_w)

        diff_h = abs(end_h) + 1
        diff_w = abs(end_w) + 1
        diff = max(diff_h, diff_w)

        size_interp = (T, H + diff, W + diff)
        video = F.interpolate(video, size=size_interp, mode='trilinear')
        # enlarged -> [B, C, T, H + D, W + D]

        curve_h = curve_h - min_h + (diff - diff_h) // 2
        curve_w = curve_w - min_w + (diff - diff_w) // 2

        pad = (time_before, T - time_before - time)
        curve = torch.stack([curve_h, curve_w], dim=0).to(torch.float32)
        curve_h, curve_w = F.pad(
            curve, pad=pad, mode='replicate').to(torch.int32)

        video = torch.stack(
            [video[..., t, h: h + H, w: w + W]
                for t, (h, w) in enumerate(zip(curve_h, curve_w))],
            dim=-3,
        )
        # video -> [B, C, T, H, W]

        return video

    def forward(self, video, step):
        # video -> [B, C, T, H, W]
        B, device = video.shape[0], video.device

        mask_scale = torch.rand(B, device=device) < self.prob_scale
        B_scale = mask_scale.sum().item()

        if B_scale > 0:
            video[mask_scale] = self.forward_scale(
                video[mask_scale], step=step)

        mask_shift = torch.rand(B, device=device) < self.prob_shift
        B_shift = mask_shift.sum().item()

        if B_shift > 0:
            video[mask_shift] = self.forward_shift(
                video[mask_shift], step=step)

        return video


class Transform_Camera(nn.Module):

    def __init__(self, prob_shift, prob_zoom, prob_shake, n_steps):
        super(Transform_Camera, self).__init__()
        assert 0 <= prob_shift <= 1
        assert 0 <= prob_zoom <= 1
        assert 0 <= prob_shake <= 1
        assert n_steps > 0

        self.prob_shift, self.prob_zoom = prob_shift, prob_zoom
        self.prob_shake, self.n_steps = prob_shake, n_steps

        self.ampl_min = (1, 2)
        self.ampl_max = (2, 4)

        self.slope_ampl_min = (self.ampl_min[1] - self.ampl_min[0]) / n_steps
        self.slope_ampl_max = (self.ampl_max[1] - self.ampl_max[0]) / n_steps

        self.time = (0.1, 1.0)
        self.slope_time = (self.time[1] - self.time[0]) / n_steps

        self.speed_shift = (0.2, 1.0)
        self.slope_speed_shift = (
            self.speed_shift[1] - self.speed_shift[0]) / n_steps

        self.speed_zoom = (0.3, 1.0)
        self.slope_speed_zoom = (
            self.speed_zoom[1] - self.speed_zoom[0]) / n_steps

    ####################################
    # SHIFT
    ####################################

    def sample_params_shift(self, n_frames, step, res):
        if step >= self.n_steps:
            speed_max = self.speed_shift[1]
            time_max = self.time[1]

        else:
            speed_max = self.speed_shift[0] + step * self.slope_speed_shift
            time_max = self.time[0] + step * self.slope_time

        speed_max = max(1, speed_max * res / n_frames)
        speed_min = max(1, self.speed_shift[0] * res / n_frames)
        speed = random.uniform(speed_min, speed_max)

        time_max = min(n_frames, max(2, int(time_max * n_frames)))
        time_min = min(n_frames, max(2, int(self.time[0] * n_frames)))
        time = random.randint(time_min, time_max)

        angle = random.uniform(0, 2 * math.pi)

        return speed, time, angle

    def forward_shift(self, video, step):
        # video -> [B, C, T, H, W]
        B, C, T, H, W = video.shape
        device = video.device

        speed, time, angle = self.sample_params_shift(
            n_frames=T, res=H, step=step)
        time_before = random.randint(0, T - time)
        # angle, speed, time -> [1]

        curve = torch.arange(time, dtype=torch.float32, device=device) * speed
        curve_h = (math.sin(angle) * curve).to(torch.int32)
        curve_w = (math.cos(angle) * curve).to(torch.int32)
        # curve_h, curve_w, curve_t -> [n_time]

        end_h = curve_h[-1].item()
        end_w = curve_w[-1].item()

        min_h = min(0, end_h)
        min_w = min(0, end_w)

        diff_h = abs(end_h) + 1
        diff_w = abs(end_w) + 1
        diff = max(diff_h, diff_w)

        size_interp = (T, H + diff, W + diff)
        video = F.interpolate(video, size=size_interp, mode='trilinear')
        # enlarged -> [B, C, T, H + D, W + D]

        curve_h = curve_h - min_h + (diff - diff_h) // 2
        curve_w = curve_w - min_w + (diff - diff_w) // 2

        pad = (time_before, T - time_before - time)
        curve = torch.stack([curve_h, curve_w], dim=0).to(torch.float32)
        curve_h, curve_w = F.pad(
            curve, pad=pad, mode='replicate').to(torch.int32)

        video = torch.stack(
            [video[..., t, h: h + H, w: w + W]
                for t, (h, w) in enumerate(zip(curve_h, curve_w))],
            dim=-3,
        )
        # video -> [B, C, T, H, W]

        return video

    ####################################
    # ZOOM
    ####################################

    def sample_params_zoom(self, step, n_frames):
        if step >= self.n_steps:
            speed_max = self.speed_zoom[1]
            time_max = self.time[1]

        else:
            speed_max = self.speed_zoom[0] + step * self.slope_speed_zoom
            time_max = self.time[0] + step * self.slope_time

        speed_max = speed_max / n_frames
        speed_min = self.speed_zoom[0] / n_frames
        speed = random.uniform(speed_min, speed_max)

        time_max = min(n_frames, max(2, int(time_max * n_frames)))
        time_min = min(n_frames, max(2, int(self.time[0] * n_frames)))
        time = random.randint(time_min, time_max)

        return speed, time

    def forward_zoom(self, video, step):
        # video -> [B, C, T, H, W]
        B, C, T, H, W = video.shape
        device = video.device

        speed, time = self.sample_params_zoom(step=step, n_frames=T)
        time_before = random.randint(0, T - time)

        scale = 1 + torch.arange(
            time, device=device, dtype=torch.float32) * speed
        if random.random() < 0.5:
            scale = torch.flip(scale, dims=(0,))

        pad = (time_before, T - time_before - time)
        scale = F.pad(scale[None, :], pad=pad, mode='replicate')[0]
        # scale -> [T]

        video = torch.stack(
            [
                TF.center_crop(F.interpolate(
                    video[..., t, :, :], scale_factor=_scale.item(), mode='bilinear'), (H, W))
                for t, _scale in enumerate(scale)
            ],
            dim=-3
        )
        # video -> [B, C, T, H, W]

        return video

    ####################################
    # SHAKE
    ####################################

    def sample_params_shake(self, n_frames, ampl, device, res):
        n_sin = random.randint(2, 5)
        freq = sample_uniform(0.1, 1.2, size=n_sin, device=device)
        # freq -> [n_sin]
        phase = sample_uniform(0, 2 * math.pi, size=1, device=device)
        # phase -> [1]
        noise = sample_uniform(-0.3, 0.3, size=n_frames, device=device)
        # noise -> [n_frames]
        #noise = torch.cumsum(noise, dim=0)

        time = torch.arange(n_frames, dtype=torch.float32, device=device)
        # time -> [n_frames]
        weight = 1 / torch.arange(
            1, 1 + n_sin, dtype=torch.float32, device=device)[None, ...]
        # weight -> [1, n_sin]
        shake = (
            weight * torch.sin(torch.outer(time, freq) + phase)).sum(dim=1) + noise
        shake = shake * ampl / shake.std() * res / 112
        shake = torch.round(shake).to(torch.int32)
        # shake -> [n_frames]

        return shake

    def sample_ampl_shake(self, step, device):
        if step >= self.n_steps:
            min_ = self.ampl_min[1]
            max_ = self.ampl_max[1]

        else:
            min_ = self.ampl_min[0] + step * self.slope_ampl_min
            max_ = self.ampl_max[0] + step * self.slope_ampl_max

        ampl = sample_uniform(min_, max_, size=1, device=device)

        return ampl

    def forward_shake(self, video, step):
        """Based on:
        https://ieeexplore.ieee.org/document/6706422
        https://www.shaopinglu.net/index.files/ICIP2018.pdf
        """
        # video -> [B, C, T, H, W]
        T, H, W = video.shape[-3:]
        device = video.device

        ampl = self.sample_ampl_shake(step=step, device=device)

        shake_h = self.sample_params_shake(
            n_frames=T, ampl=ampl, device=device, res=H)
        shake_w = self.sample_params_shake(
            n_frames=T, ampl=ampl, device=device, res=H)
        # shake_h, shake_w -> [T]

        min_h, max_h = shake_h.min().item(), shake_h.max().item()
        min_w, max_w = shake_w.min().item(), shake_w.max().item()
        diff_h, diff_w = max_h - min_h + 1, max_w - min_w + 1
        diff = max(diff_h, diff_w)

        size_interp = (T, H + diff, W + diff)
        enlarged = F.interpolate(
            video, size=size_interp, mode='trilinear')
        # enlarged -> [B, C, T, H + D, W + D]
        enlarged = torch.clip(enlarged, 0, 1).permute(2, 0, 1, 3, 4)
        # enlarged -> [T, B, C, H + D, W + D]

        shake_h = shake_h - min_h + (diff - diff_h) // 2
        shake_w = shake_w - min_w + (diff - diff_w) // 2

        video = torch.stack(
            [v[..., h: h + H, w: w + W]
             for v, h, w in zip(enlarged, shake_h, shake_w)],
            dim=-3,
        )
        # video -> [B, C, T, H, W]

        return video

    def forward(self, video, step):
        # video -> [B, C, T, H, W]
        B, device = video.shape[0], video.device
        mask_shift = torch.rand(B, device=device) < self.prob_shift
        mask_zoom = torch.rand(B, device=device) < self.prob_zoom
        mask_shake = torch.rand(B, device=device) < self.prob_shake

        B_shift = mask_shift.sum().item()
        B_zoom = mask_zoom.sum().item()
        B_shake = mask_shake.sum().item()

        if B_shift > 0:
            video[mask_shift] = self.forward_shift(
                video=video[mask_shift], step=step)

        if B_zoom > 0:
            video[mask_zoom] = self.forward_zoom(
                video=video[mask_zoom], step=step)

        if B_shake > 0:
            video[mask_shake] = self.forward_shake(
                video=video[mask_shake], step=step)

        return video
