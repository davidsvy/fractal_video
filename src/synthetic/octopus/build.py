import cv2
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw
import random
import scipy.interpolate
import time

from ..args import print_status
from ..io import save_video
from ..params import interp_array, interp_linear, random_curve, sample_interp
from ...utils.other import Average_Meter, update_size_counter


PROB_EMBED_POLY = 0.8
PROB_COLOR = 0.8
PROB_VAR_THICK = 0.3
PROB_INTERIOR = 0.4


def sample_curve_1d():
    n_points_init = random.randint(3, 5)
    sample_x = interp_linear(n_points_init)
    sample_y = np.random.uniform(0.85, 1.15, size=n_points_init)

    curve_fn = scipy.interpolate.interp1d(
        sample_x, sample_y, kind='quadratic')

    return curve_fn


def mutate_curve_1d(fn, n_points):
    start = np.random.uniform(0, 0.1)
    end = np.random.uniform(0.4, 0.6)
    linear = interp_linear(size=n_points, start=start, end=end)
    curve = fn(linear)
    # interpolant -> [n_points]

    return curve


def mutate_curve_2d(fn_h, fn_w, n_points):
    curve_h = mutate_curve_1d(fn=fn_h, n_points=n_points)
    curve_w = mutate_curve_1d(fn=fn_w, n_points=n_points)
    curve = np.stack((curve_h, curve_w), axis=1)
    # curve -> [n_points, 2]

    return curve


def join_limbs(coords):
    # coords -> list of [n_frames, n_points, 2]
    n_limbs = len(coords)
    n_points = coords[0].shape[1]

    idx_center = np.random.randint(0, n_limbs)
    idxs_joint = np.random.uniform(0, 0.15, size=n_limbs) * n_points
    idxs_joint = np.clip(idxs_joint, 0, n_points - 1).astype(np.int32)

    for idx in range(n_limbs):
        if idx != idx_center:
            offset = (
                coords[idx_center][:, idxs_joint[idx_center]] - coords[idx][:, idxs_joint[idx]])
            offset = offset[:, None]
            coords[idx] += offset

    coords = np.concatenate(coords, axis=1)

    return coords


def quantize_coords(coords, res=256):
    # coords -> [..., 2]
    axis = tuple(range(coords.ndim - 1))
    min_ = np.amin(coords, axis=axis, keepdims=True)
    max_ = np.amax(coords, axis=axis, keepdims=True)

    min_ -= 0.01 * np.abs(min_)
    max_ += 0.01 * np.abs(max_)

    borders_add = min_
    border_scale = (res / (max_ - min_))
    # borders_add, border_scale -> [..., 2]

    pixels = (
        (coords - borders_add) * border_scale).astype(np.uint16)
    # pixels -> [..., 2]

    return pixels


def fill_canvas(coords, res=256):
    # coords -> [n_frames, n_points, 2]
    n_frames = coords.shape[0]
    out = np.zeros(shape=(n_frames, res, res), dtype=np.uint8)

    for frame in range(n_frames):
        out[frame, coords[frame, :, 0], coords[frame, :, 1]] = 255

    return out


def thicken(video):
    # video -> [n_frames, res, res]
    n_frames = video.shape[0]
    color = np.random.randint(0, 256, (1, 1, 1, 3), dtype=np.uint8)
    sigma = np.random.randint(3, 9, size=(2,))

    if np.random.rand() < PROB_VAR_THICK:
        threshold = np.random.randint(3, 10, size=2)
        threshold = np.linspace(threshold[0], threshold[1], num=n_frames)
        threshold = np.around(threshold).astype(np.uint8)
        threshold = threshold[:, None, None]

    else:
        threshold = np.random.randint(3, 10)

    video = video.astype(np.float32)
    for idx in range(n_frames):
        video[idx] = cv2.GaussianBlur(
            video[idx], (0, 0), sigmaX=sigma[0], sigmaY=sigma[1], borderType=cv2.BORDER_DEFAULT)

    video = (video > threshold).astype(np.uint8) * 255
    video = np.repeat(video[..., None], 3, axis=-1) * color

    kernel_closing = np.ones((2, 2), dtype=np.uint8)
    for idx in range(n_frames):
        video[idx] = cv2.morphologyEx(
            video[idx], cv2.MORPH_CLOSE, kernel_closing)

    return video


def embed_polygons(video, coords):
    # video -> [n_frames, res, res]
    # coords -> [n_frames, n_coords, 2]
    if random.random() > PROB_EMBED_POLY:
        return video

    n_poly = np.random.randint(1, 4)
    n_corner = np.random.randint(2, 8, size=n_poly)
    angle_poly = np.random.randint(0, 360, size=n_poly)
    idx_coord = np.random.randint(0, coords.shape[1], size=n_poly)
    radius = np.random.randint(18, 28, size=n_poly)
    radius = np.maximum(
        np.round_(radius * video.shape[1] / 256).astype(np.int32), 1)

    color = video[0, coords[0, 0, 0], coords[0, 0, 1]]
    if isinstance(color, np.ndarray):
        color = tuple(color)
    else:
        color = int(color)

    for idx_frame in range(video.shape[0]):
        frame = Image.fromarray(video[idx_frame])
        draw = ImageDraw.Draw(frame)

        for idx_poly in range(n_poly):
            center = coords[idx_frame, idx_coord[idx_poly]][::-1]

            if n_corner[idx_poly] < 3:
                bbox = np.concatenate(
                    (center - radius[idx_poly], center + radius[idx_poly]), axis=0).tolist()
                draw.ellipse(xy=bbox, fill=color)

            else:
                bcircle = (center[0], center[1], radius[idx_poly])
                bcircle = [int(val) for val in bcircle]
                draw.regular_polygon(
                    bounding_circle=bcircle,
                    n_sides=int(n_corner[idx_poly]),
                    rotation=int(angle_poly[idx_poly]),
                    fill=color,
                )

        video[idx_frame] = np.array(frame)

    return video


def colorize(video, coords):
    # video -> [n_frames, res, res]
    # coords -> [n_frames, n_coords, 2]
    if random.random() > PROB_COLOR:
        return video

    n_poly = np.random.randint(2, 6)
    n_corner = np.random.randint(2, 8, size=n_poly)
    angle_poly = np.random.randint(0, 360, size=n_poly)
    idx_coord = np.random.randint(0, coords.shape[1], size=n_poly)
    radius = np.random.randint(18, 32, size=n_poly)
    radius = np.maximum(
        np.round_(radius * video.shape[1] / 256).astype(np.int32), 1)
    color = np.random.randint(0, 256, size=(n_poly, 3))
    color = [tuple(c) for c in color]

    res = video.shape[1]
    color_mask = []
    for idx_frame in range(video.shape[0]):
        frame = Image.new(mode='RGB', size=(res, res), color=(0, 0, 0))
        draw = ImageDraw.Draw(frame)

        for idx_poly in range(n_poly):
            center = coords[idx_frame, idx_coord[idx_poly]][::-1]

            if n_corner[idx_poly] < 3:
                bbox = np.concatenate(
                    (center - radius[idx_poly], center + radius[idx_poly]), axis=0).tolist()
                draw.ellipse(xy=bbox, fill=color[idx_poly])

            else:
                bcircle = (center[0], center[1], radius[idx_poly])
                bcircle = [int(val) for val in bcircle]
                draw.regular_polygon(
                    bounding_circle=bcircle,
                    n_sides=int(n_corner[idx_poly]),
                    rotation=int(angle_poly[idx_poly]),
                    fill=color[idx_poly],
                )

        color_mask.append(np.array(frame))

    color_mask = np.stack(color_mask, axis=0)
    mask = (color_mask[..., :1] > 0) & (video[..., :1] > 0)
    video = np.where(mask, color_mask, video)

    return video


def remove_interior(video):
    # video -> [n_frames, res, res]
    if random.random() > PROB_INTERIOR:
        return video

    n_frames = video.shape[0]
    if np.random.rand() < PROB_VAR_THICK:
        kernel_size = np.empty((n_frames, 2), dtype=np.uint8)

        for idx in range(2):
            kernel_size_ = np.random.randint(2, 7, size=2)
            kernel_size_ = np.linspace(
                kernel_size_[0], kernel_size_[1], num=n_frames)
            kernel_size_ = np.around(kernel_size_).astype(np.uint8)
            kernel_size[:, idx] = kernel_size_
    else:
        kernel_size = np.random.randint(2, 7, size=(1, 2))
        kernel_size = kernel_size.repeat(n_frames, axis=0)

    kernel = [np.ones(k, dtype=np.uint8) for k in kernel_size]
    #kernel = np.ones(np.random.randint(3, 6, size=2), dtype=np.uint8)
    for idx in range(video.shape[0]):
        video[idx] = cv2.morphologyEx(
            video[idx], cv2.MORPH_GRADIENT, kernel[idx])

    return video


def animate_curve(coords, res=256):
    # coords -> list of [n_frames, n_points, 2]
    coords = join_limbs(coords)
    # coords -> [n_frames, n_limbs * n_points, 2]
    coords = quantize_coords(coords=coords, res=res)
    # coords -> [n_frames, n_limbs * n_points, 2]
    video = fill_canvas(coords=coords, res=res)
    # video -> [n_frames, res, res]
    video = thicken(video)
    # video -> [n_frames, res, res, 3]
    video = embed_polygons(video=video, coords=coords)
    video = colorize(video=video, coords=coords)
    video = remove_interior(video)
    # video -> [n_frames, res, res, 3]

    return video


def sample_params():
    params = {}
    params['n_limbs'] = random.randint(3, 5)

    params['curve_fn'] = [
        sample_curve_1d() for _ in range(4 * params['n_limbs'])]
    params['interp'] = sample_interp(params['n_limbs'])

    return params


def compose_params(params, mutate=True):
    n_points = 512

    curves = [
        mutate_curve_2d(
            fn_h=params['curve_fn'][idx],
            fn_w=params['curve_fn'][idx + 1],
            n_points=n_points,
        )
        for idx in range(0, 4 * params['n_limbs'], 2)
    ]
    # curves -> [2 * n_limbs, n_points, 2]

    coords = []
    for idx in range(params['n_limbs']):
        curve = interp_array(
            start=curves[2 * idx],
            end=curves[2 * idx + 1],
            interp=params['interp'][:, idx],
        )
        # curve -> [n_frames, n_points, 2]
        if mutate:
            mutation = [
                random_curve(curve.shape[0], eps=0.03) for _ in range(2)]
            mutation = np.stack(mutation, axis=1)[:, None]
            # mutation -> [n_frames, 1, 2]
            curve *= mutation
        coords.append(curve)

    return coords


def generate_classes(args):
    print('\n' * 3 + '#' * 80)
    print(f'Sampling parameters for {args.gen_param} classes...')
    print('#' * 80 + '\n' * 3)

    for step in range(1, 1 + args.gen_param):
        params = sample_params()
        path = os.path.join(args.dir_out, f'{step}.npz')
        np.savez(path, **params)


def build_dataset(args):
    is_labeled = hasattr(args, 'paths_param') and len(args.paths_param) > 0

    step, idx_class = 1, 0
    counter_size = Average_Meter()
    time_start = time.time()
    last_iter = False
    n_classes = None

    if is_labeled:
        random.shuffle(args.paths_param)
        n_classes = len(args.paths_param)

    while True:
        if is_labeled:
            name_class = pathlib.Path(args.paths_param[idx_class]).stem
            dir_out = os.path.join(args.dir_out, name_class)
            path_out = os.path.join(dir_out, f'{step}.mp4')

            params = np.load(args.paths_param[idx_class], allow_pickle=True)

        else:
            path_out = os.path.join(args.dir_out, f'{step}.mp4')
            params = sample_params()

        coords = compose_params(params=params, mutate=is_labeled)
        # coords -> [n_limbs, n_frames, n_points, 2]
        video = animate_curve(coords=coords, res=args.res)

        save_video(frames=video, path=path_out, fps=args.fps, scale=False, lib=args.lib_save)
        
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
        if is_labeled:
            idx_class = (idx_class + 1) % n_classes


def build(args):
    assert args.dir_in is None or args.gen_param is None
    os.makedirs(args.dir_out, exist_ok=True)

    if args.gen_param is not None:
        generate_classes(args)

    else:
        assert args.samples or args.time, 'Provide "samples" or "time"'
        build_dataset(args)
