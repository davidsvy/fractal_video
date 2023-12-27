import collections
import numpy as np
import os
import pathlib


##########################################################################
##########################################################################
# GENERAL
##########################################################################
##########################################################################

def find_files(dir, ext):
    if isinstance(ext, str):
        paths = pathlib.Path(dir).rglob(f'**/*.{ext}')

    elif isinstance(ext, (list, tuple)):
        paths = []
        for ext_ in ext:
            paths_ = pathlib.Path(dir).rglob(f'**/*.{ext_}')
            paths += [str(path_) for path_ in paths_]

    else:
        raise ValueError(f'Unknown ext "{ext}" with type "{type(ext)}"')

    paths = sorted([str(path) for path in paths])

    return paths


def find_classes(paths, label_dict=None):
    labels = [path.split('/')[-2] for path in paths]

    if label_dict is None:
        label_set = sorted(list(set(labels)))
        label_dict = {str_: idx for idx, str_ in enumerate(label_set)}

    labels = np.array([label_dict[label] for label in labels])

    return labels, label_dict


def get_frame_count(paths):
    import decord

    decord.bridge.set_bridge('torch')

    frame_count = []

    for path in paths:
        n_frames = len(decord.VideoReader(path))
        frame_count.append(n_frames)

    frame_count = np.array(frame_count)

    return frame_count


def filter_length(paths, labels, frame_count, min_length):
    paths, frame_count = np.array(paths), np.array(frame_count)
    mask = frame_count > min_length
    paths, frame_count = paths[mask], frame_count[mask]

    if labels is not None:
        labels = np.array(labels)[mask]

    return paths, labels, frame_count


def split_segments(frame_count, clip_length=16, min_step=24, max_segs=8, stride=1):
    # frame_count -> [n_videos]
    assert clip_length > 0 and min_step > 0 and stride > 0
    assert max_segs > 0

    n_videos = len(frame_count)
    segments = []
    valid_mask = np.full(shape=n_videos, fill_value=False)
    # valid_mask -> [n_videos]
    seg_length = (clip_length - 1) * stride + 1

    for idx, n_frames in enumerate(frame_count):
        if n_frames < seg_length:
            continue

        valid_mask[idx] = True

        if max_segs == 1:
            seg_ = [0]

        else:
            n_seg = (n_frames + min_step - seg_length) // min_step
            if n_seg <= max_segs:
                step = min_step

            else:
                step = (n_frames - seg_length) // (max_segs - 1)
                n_seg = max_segs

            seg_ = [step * idx for idx in range(n_seg)]

        segments.append(seg_)

    return segments, valid_mask


def split_segments_pad(frame_count, clip_length=16, min_step=24, max_segs=8, stride=1):
    # frame_count -> [n_videos]
    assert clip_length > 0 and min_step > 0 and stride > 0
    assert max_segs > 0

    n_short = 0
    segments = [None] * len(frame_count)
    seg_length = (clip_length - 1) * stride + 1

    for idx, n_frames in enumerate(frame_count):
        if n_frames < seg_length:
            n_short += 1
            continue

        if max_segs == 1:
            seg_ = [0]

        else:
            n_seg = (n_frames + min_step - seg_length) // min_step
            if n_seg <= max_segs:
                step = min_step
                n_seg = n_seg
            else:
                step = (n_frames - seg_length) // (max_segs - 1)
                n_seg = max_segs

            seg_ = [step * idx for idx in range(n_seg)]

        segments[idx] = seg_

    return segments, n_short


def split_stats(dir, ext):
    paths = find_files(dir=dir, ext=ext)
    labels, _ = find_classes(paths)
    count = np.array(list(collections.Counter(labels).values()))

    out = {
        'n_classes': len(count),
        'n_files_total': len(paths),
        'n_files_avg': count.mean(),
        'n_files_min': count.min(),
        'n_files_max': count.max(),
    }

    return out


def dataset_stats(root, ext):
    n_train = len(find_files(dir=os.path.join(root, 'train'), ext=ext))
    n_val = len(find_files(dir=os.path.join(root, 'val'), ext=ext))
    n_test = len(find_files(dir=os.path.join(root, 'test'), ext=ext))
    print(f'train -> {n_train} files')
    print(f'val -> {n_val} files')
    print(f'test -> {n_test} files')
    