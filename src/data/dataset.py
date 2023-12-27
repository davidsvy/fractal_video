import decord
import numpy as np
import random

import torch

from ..utils.data import get_frame_count, split_segments_pad


decord.bridge.set_bridge('torch')

##########################################################################
##########################################################################
# LABELED DATASETS
##########################################################################
##########################################################################


class Dataset_Labeled_Train(torch.utils.data.Dataset):

    def __init__(self, paths, labels, clip_length=16, stride=1,
                 transform=None, n_threads=1):
        assert len(paths), 'No file paths provided'
        assert len(labels), 'No labels provided'

        self.paths, self.labels = paths, labels
        self.full_length = (clip_length - 1) * stride + 1
        self.clip_length, self.stride = clip_length, stride
        self.transform = transform
        self.n_threads = n_threads
        self.decord_device = decord.cpu(0)

        n_videos = len(paths)
        frame_count = get_frame_count(paths)
        n_short = (frame_count < self.full_length).sum()

        print(
            f'Train dataset: '
            f'{n_short}/{n_videos} ({(n_short / n_videos):.3f}) videos need padding'
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        reader = decord.VideoReader(
            uri=self.paths[idx],
            ctx=self.decord_device,
            num_threads=self.n_threads,
        )

        n_frames = len(reader)
        if n_frames < self.full_length:
            idxs_frame = list(range(0, n_frames, self.stride))
            idxs_frame += [n_frames - 1] * (self.clip_length - len(idxs_frame))

        else:
            idx_start = random.randint(0, n_frames - self.full_length)
            idxs_frame = range(
                idx_start, idx_start + self.full_length, self.stride)

        video = reader.get_batch(idxs_frame).permute(3, 0, 1, 2)
        # video -> [C, T, H, W]

        if self.transform is not None:
            video = self.transform(video)

        label = self.labels[idx]

        return video, label


class Dataset_Labeled_Val(torch.utils.data.Dataset):

    def __init__(self, paths, labels, clip_length=16, min_step=24,
                 max_segs=8, stride=1, transform=None, n_threads=1):
        assert len(paths), 'No file paths provided'
        assert len(labels), 'No labels provided'

        self.paths, self.labels = paths, labels
        self.clip_length = clip_length
        self.transform = transform
        self.decord_device = decord.cpu(0)
        self.n_threads = n_threads
        seg_length = (clip_length - 1) * stride + 1

        frame_count = get_frame_count(paths)
        segments, n_short = split_segments_pad(
            frame_count=frame_count,
            clip_length=clip_length,
            min_step=min_step,
            max_segs=max_segs,
            stride=stride,
        )

        n_videos = len(paths)
        assert n_short < n_videos, 'All files are too short'
        print(
            f'Val dataset: '
            f'{n_short}/{n_videos} ({(n_short / n_videos):.3f}) videos need padding'
        )

        self.idxs_frame, self.n_seg = [], []
        for idx, seg in enumerate(segments):

            if seg is None:
                n_seg_ = 1
                idxs_frame_ = np.arange(0, frame_count[idx], stride)
                n_pad = clip_length - len(idxs_frame_)
                idxs_frame_ = np.pad(
                    idxs_frame_, (0, n_pad), constant_values=frame_count[idx] - 1)

            else:
                n_seg_ = len(seg)
                idxs_frame_ = []

                for start in seg:
                    arange = np.arange(start, start + seg_length, stride)
                    idxs_frame_.append(arange)

                idxs_frame_ = np.concatenate(idxs_frame_, axis=0)

            self.idxs_frame.append(idxs_frame_)
            self.n_seg.append(n_seg_)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        reader = decord.VideoReader(
            uri=self.paths[idx],
            ctx=self.decord_device,
            num_threads=self.n_threads,
        )

        idxs_frame, n_seg = self.idxs_frame[idx], self.n_seg[idx]

        video = reader.get_batch(
            idxs_frame).permute(3, 0, 1, 2)
        # video -> [C, S * T, H, W]

        if self.transform is not None:
            video = self.transform(video)
        # video -> [C, S * T, H, W]
        C, _, H, W = video.shape

        video = video.view(
            C, n_seg, self.clip_length, H, W).permute(1, 0, 2, 3, 4)
        # video -> [S, C, T, H, W]
        label = torch.full(size=(n_seg,), fill_value=self.labels[idx])
        # label -> [S]

        return video, label


##########################################################################
##########################################################################
# CONTRASTIVE/UNLABELED DATASETS
##########################################################################
##########################################################################


class Dataset_Contrastive(torch.utils.data.Dataset):

    def __init__(self, paths, labels, clip_length=16, stride=1, transform=None, n_threads=1):
        assert len(paths), 'No file paths provided'
        assert len(labels), 'No labels provided'

        self.paths, self.labels = paths, labels
        self.clip_length, self.stride = clip_length, stride
        self.full_length = (clip_length - 1) * stride + 1
        self.n_threads = n_threads
        self.decord_device = decord.cpu(0)

        self.transform = transform
        self.use_transform = transform is not None

        n_videos = len(paths)
        frame_count = get_frame_count(paths)
        n_short = (frame_count < self.full_length).sum()

        print(
            f'Contrastive dataset: '
            f'{n_short}/{n_videos} ({(n_short / n_videos):.3f}) videos need padding'
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        reader = decord.VideoReader(
            uri=self.paths[idx],
            ctx=self.decord_device,
            num_threads=self.n_threads,
        )

        n_frames = len(reader)
        if n_frames < self.full_length:
            idxs_frame = list(range(0, n_frames, self.stride))
            idxs_frame += [n_frames - 1] * (self.clip_length - len(idxs_frame))

            video1 = reader.get_batch(idxs_frame).permute(3, 0, 1, 2)
            video2 = video1.clone()

        else:
            idx_start1 = random.randint(0, n_frames - self.full_length)
            idx_start2 = random.randint(0, n_frames - self.full_length)

            idxs_frame1 = list(range(
                idx_start1, idx_start1 + self.full_length, self.stride))
            idxs_frame2 = list(range(
                idx_start2, idx_start2 + self.full_length, self.stride))
            idxs_frame = idxs_frame1 + idxs_frame2

            video1, video2 = torch.chunk(
                reader.get_batch(idxs_frame).permute(3, 0, 1, 2), chunks=2, dim=1)

        # video1, video2 -> [C, T, H, W]
        if self.use_transform:
            video1, video2 = self.transform(video1), self.transform(video2)

        video = torch.stack([video1, video2], dim=0)
        # video1, video2 -> [2, C, T, H, W]

        label = self.labels[idx]

        return video, label


##########################################################################
##########################################################################
# DATALOADERS
##########################################################################
##########################################################################


def collate_labeled_val(batch):
    inputs = torch.cat([sample[0] for sample in batch], dim=0)
    labels = torch.cat([sample[1] for sample in batch], dim=0)

    return inputs, labels
