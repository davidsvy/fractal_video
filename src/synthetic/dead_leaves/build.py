import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw
import random
import time

from ..args import print_status
from ..io import save_video
from ..params import random_curve, sample_interp
from ...utils.other import Average_Meter, update_size_counter


def sample_curve(n_frames):
    interp = np.concatenate(
        [sample_interp(n_fn=1, n_frames=n_frames)
         for _ in range(2)],
        axis=-1
    )
    # interp -> [n_frames, 2]

    min_ = np.amin(interp, axis=0, keepdims=True)
    max_ = np.amax(interp, axis=0, keepdims=True)
    interp = (interp - min_) / (max_ - min_)

    scale = np.random.uniform(
        1.0, 2.0, size=(1, 2)) * np.random.choice([-1, 1], size=(1, 2))
    offset = np.random.uniform(-2.0, 2.0, size=(1, 2))
    interp = interp * scale + offset

    for idx in range(2):
        if random.random() < 0.5:
            interp[:, idx] = interp[:, idx][::-1]
    # interp -> [n_frames, 2]

    return interp


def mutate_coords(coords):
    # coords -> [n_frames, n_poly, 2]
    mask = np.random.rand(coords.shape[1]) > 0.2
    coords = coords[:, mask]

    n_frames, n_poly = coords.shape[:2]
    perm = np.random.permutation(n_poly)
    coords = coords[:, perm]

    mut = [
        random_curve(n_frames=n_frames, eps=0.2) for _ in range(2 * n_poly)]
    mut = np.stack(mut, axis=1)
    # mut -> [n_frames, 2 * n_poly]
    mut = mut.reshape((n_frames, n_poly, 2))
    coords *= mut

    min_ = np.amin(coords, axis=(0, 1), keepdims=True)
    max_ = np.amax(coords, axis=(0, 1), keepdims=True)

    offset = np.random.uniform(-1, 1, size=(n_poly, 2))
    offset = offset * (max_ - min_) * 0.08
    coords += offset

    return coords


def normalize_coords(coords, res):
    min_ = np.amin(coords, axis=(0, 1), keepdims=True)
    max_ = np.amax(coords, axis=(0, 1), keepdims=True)
    coords = (coords - min_) / (max_ - min_)
    coords = np.clip(np.around(coords * res), 0, res - 1).astype(np.int32)

    return coords


def sample_appearance(n_poly):
    n_sides = np.random.randint(1, 9, size=n_poly)
    rotation = np.random.randint(0, 360, size=n_poly)
    radius = np.random.randint(12, 33, size=n_poly)
    color = np.random.randint(0, 256, size=(n_poly, 3))

    return n_sides, rotation, radius, color


def draw_frame(res, coords, radius, n_sides, rotation, color):
    # coords -> [n_poly, 2]
    # radius, n_sides, rotation -> [n_poly]
    frame = Image.new(mode='RGB', size=(res, res), color=(0, 0, 0))
    draw = ImageDraw.Draw(frame)

    for idx in range(coords.shape[0]):
        center = coords[idx][::-1]

        if n_sides[idx] < 3:
            bbox = np.concatenate(
                (center - radius[idx], center + radius[idx]), axis=0).tolist()
            draw.ellipse(
                xy=bbox,
                fill=tuple(color[idx]),
            )

        else:
            bcircle = (center[0], center[1], radius[idx])
            bcircle = tuple(int(c) for c in bcircle)

            draw.regular_polygon(
                bounding_circle=bcircle,
                n_sides=int(n_sides[idx]),
                rotation=int(rotation[idx]),
                fill=tuple(color[idx]),
            )

    frame = np.array(frame)

    return frame


def build_sample(coords, res, mutate=True):
    # coords -> [n_frames, n_poly, 2]
    n_poly = coords.shape[1]
    if mutate:
        coords = mutate_coords(coords)

    coords = normalize_coords(coords=coords, res=res)

    n_sides, rotation, radius, color = sample_appearance(n_poly)

    video = []
    for coords_ in coords:
        frame = draw_frame(
            res=res,
            coords=coords_,
            radius=radius,
            n_sides=n_sides,
            rotation=rotation,
            color=color,
        )

        video.append(frame)

    video = np.stack(video, axis=0)

    return video


def sample_params():
    params = {}
    n_frames = random.randint(18, 20)
    n_poly = random.randint(15, 30)
    coords = [sample_curve(n_frames) for _ in range(n_poly)]
    coords = np.stack(coords, axis=1)
    # coords -> [n_frames, n_poly, 2]
    params['coords'] = coords

    return params


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

        video = build_sample(
            coords=params['coords'], res=args.res, mutate=is_labeled)

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
