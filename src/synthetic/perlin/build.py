import itertools
import numpy as np
import numba
import os
import random
import time

from ..args import print_status
from ..io import save_video
from ...utils.other import Average_Meter, update_size_counter

# By construction, perlin noise is rendered with a fixed resolution.
VIDEO_SHAPE = (240, 240, 30)

VALID_SPACE_FREQS = [
    1, 2, 3, 5, 6, 8, 10, 12, 16,
    20, 30, 40, 60, 80, 120, 240
]

VALID_TIME_FREQS = [1]


@numba.njit(parallel=True, fastmath=True)
def interpolant(t):
    return t ** 3 * (t * (t * 6 - 15) + 10)


@numba.njit(parallel=True, fastmath=True)
def perlin3d(shape, res):
    """Based on:
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin3d.py
    and
    https://github.com/pvigier/perlin-numpy/issues/9#issue-968667149
    """
    dtype = np.float32
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    range1 = np.arange(0, res[0], delta[0]).astype(dtype) % 1
    range2 = np.arange(0, res[1], delta[1]).astype(dtype) % 1
    range3 = np.arange(0, res[2], delta[2]).astype(dtype) % 1

    grid = np.empty(shape=(shape[0], shape[1], shape[2], 3), dtype=dtype)
    # grid -> [shape[0], shape[1], shape[2], 3]

    for idx in numba.prange(shape[0]):
        grid[idx, :, :, 0] = range1[idx]

    for idx in numba.prange(shape[1]):
        grid[:, idx, :, 1] = range2[idx]

    for idx in numba.prange(shape[2]):
        grid[:, :, idx, 2] = range3[idx]

    # Gradients
    theta = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)
    phi = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)

    gradients = np.stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)),
        axis=-1
    )
    # gradients -> [res[0] + 1, res[1] + 1, res[2] + 1, 3]

    grad_shape = (
        d[0] * gradients.shape[0], d[1] * gradients.shape[1],
        d[2] * gradients.shape[2], 3)
    grad_matrix = np.empty(shape=grad_shape, dtype=dtype)

    for idx1 in numba.prange(gradients.shape[0]):
        for idx2 in numba.prange(gradients.shape[1]):
            for idx3 in numba.prange(gradients.shape[2]):
                grad_matrix[
                    d[0] * idx1: d[0] * (idx1 + 1),
                    d[1] * idx2: d[1] * (idx2 + 1),
                    d[2] * idx3: d[2] * (idx3 + 1),
                ] = gradients[idx1, idx2, idx3]

    gradients = grad_matrix
    # gradients -> [shape[0] + d[0], shape[1] + d[1], shape[2] + d[2], 3]

    g000 = gradients[:-d[0], :-d[1], :-d[2]]
    g100 = gradients[d[0]:, :-d[1], :-d[2]]
    g010 = gradients[:-d[0], d[1]:, :-d[2]]
    g110 = gradients[d[0]:, d[1]:, :-d[2]]
    g001 = gradients[:-d[0], :-d[1], d[2]:]
    g101 = gradients[d[0]:, :-d[1], d[2]:]
    g011 = gradients[:-d[0], d[1]:, d[2]:]
    g111 = gradients[d[0]:, d[1]:, d[2]:]
    # gxy -> [shape[0], shape[1], shape[2], 3]

    # Ramps

    n_bits = 3
    len_ = 2 ** n_bits
    code = ((np.arange(len_).reshape(len_, 1) & (1 << np.arange(n_bits)))) > 0
    code = code.astype(np.int32)
    # gradients -> [8, 3]

    n000 = np.sum((grid - code[0]) * g000, 3)
    n100 = np.sum((grid - code[1]) * g100, 3)
    n010 = np.sum((grid - code[2]) * g010, 3)
    n110 = np.sum((grid - code[3]) * g110, 3)
    n001 = np.sum((grid - code[4]) * g001, 3)
    n101 = np.sum((grid - code[5]) * g101, 3)
    n011 = np.sum((grid - code[6]) * g011, 3)
    n111 = np.sum((grid - code[7]) * g111, 3)
    # nxyz -> [shape[0], shape[1], shape[2]]

    t = interpolant(grid)
    t1 = 1 - t[:, :, :, 0]

    n00 = t1 * n000 + t[:, :, :, 0] * n100
    n10 = t1 * n010 + t[:, :, :, 0] * n110
    n01 = t1 * n001 + t[:, :, :, 0] * n101
    n11 = t1 * n011 + t[:, :, :, 0] * n111

    t2 = 1 - t[:, :, :, 1]
    n0 = t2 * n00 + t[:, :, :, 1] * n10
    n1 = t2 * n01 + t[:, :, :, 1] * n11

    output = (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1

    output = (output.transpose(2, 0, 1) + 1) / 2

    return output


@numba.njit(parallel=False, fastmath=True)
def augmented_perlin3d(shape, freqs, weights):
    video = np.zeros(shape=shape, dtype=np.float32)

    for idx in range(len(freqs)):
        video += weights[idx] * perlin3d(
            shape=shape, res=(freqs[idx, 0], freqs[idx, 1], freqs[idx, 2]))

    video = (video.transpose(2, 0, 1) + 1) / 2

    return video


########################################################
########################################################


def build_dataset(args):

    args.res = VIDEO_SHAPE[0]

    all_labels = list(
        itertools.product(VALID_SPACE_FREQS, VALID_SPACE_FREQS, VALID_TIME_FREQS))
    random.shuffle(all_labels)
    n_labels = len(all_labels)

    step, idx_label = 1, 0
    last_iter = False
    counter_size, time_start = Average_Meter(), time.time()

    while True:
        label = all_labels[idx_label]

        label_str = '-'.join([str(x) for x in label])
        dir_out = os.path.join(args.dir_out, label_str)
        path_out = os.path.join(dir_out, f'{step}.mp4')

        n_frames = random.randint(18, 20)
        video_shape = (240, 240, n_frames)
        
        video = perlin3d(shape=video_shape, res=label)
        
        save_video(frames=video, path=path_out, fps=args.fps, lib=args.lib_save)        

        update_size_counter(counter=counter_size, path=path_out)

        time_run = time.time() - time_start

        if args.samples and step >= args.samples:
            last_iter = True

        if args.time and time_run >= args.time:
            last_iter = True

        if step % args.print_every == 0 or last_iter:
            print_status(args=args, n_created=step, time_run=time_run, counter_size=counter_size)

        if last_iter:
            break

        step += 1
        idx_label = (idx_label + 1) % n_labels


def compile_numba():
    _ = perlin3d(shape=(20, 16, 6), res=(5, 8, 2))


def build(args):
    assert args.gen_param is None
    assert args.samples or args.time

    compile_numba()
    build_dataset(args)
