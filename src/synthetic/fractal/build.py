import copy
import numpy as np
import os
import pathlib
import random
import time

from .bounding_box import estimate_bounding_box
from .chaos_game import chaos_game, chaos_game_binary
from .params import (
    compose_params,
    mutate_weights,
    sample_params_video,
    sample_params_img,
)

from ..args import print_status
from ..io import save_imgs, save_video
from ...utils.other import Average_Meter, update_size_counter


def postprocess(frames, img=False):
    # frames -> [n_frames, height, width]
    frames = np.log(frames + 1)
    if img:
        frames = frames / np.maximum(1, frames.max(axis=(1, 2), keepdims=True))
    else:
        frames = frames / np.maximum(1, frames.max())

    return frames


def detect_borders(weights, var, args):
    # weights -> [n_frames, n_fn, 6]
    const_b = 6
    eps = 0.1
    n_frames = weights.shape[0]
    n_kept = n_frames if args.img else 2
    
    borders = np.array([
        -const_b, const_b, -const_b, const_b], dtype=np.float32)
    borders = np.ascontiguousarray(np.repeat(borders[None, :], n_kept, axis=0))
    # borders -> [n_frames, 4] or [2, 4]
    if not args.img:
        weights = weights[[0, -1]]
    # weights -> [n_frames, n_fn, 6] or [2, n_fn, 6]

    frames = chaos_game_binary(
        weights=weights,
        borders=borders,
        batch_size=args.bs_point,
        res=args.res_b,
        iter=args.iter_b,
        iter_skip=args.iter_skip,
        var=var,
    )
    # frames -> [n_frames, res, res] or [2, res, res]

    borders = estimate_bounding_box(img=frames, kept=args.thres_b)
    # borders -> [n_frames, 4] or [2, 4]

    if args.img:
        h1, h2, w1, w2 = borders.T 
        # h1, h2, w1, w2 -> [n_frames]
        h_eps = np.round(eps * (h2 - h1)).astype(int)
        w_eps = np.round(eps * (w2 - w1)).astype(int)

        h1, h2 = np.clip(h1 - h_eps, 0, args.res_b - 1), np.clip(h2 + h_eps, 0, args.res_b - 1)
        w1, w2 = np.clip(w1 - w_eps, 0, args.res_b - 1), np.clip(w2 + w_eps, 0, args.res_b - 1)
        
        borders = np.stack([h1, h2, w1, w2], axis=1)
        
    else:
        h1, h2, w1, w2 = borders.T
        h1, w1 = h1.min(), w1.min()
        h2, w2 = h2.max(), w2.max()
        
        h_eps = int(round(eps * (h2 - h1)))
        w_eps = int(round(eps * (w2 - w1)))

        h1, w1 = max(0, h1 - h_eps), max(0, w1 - w_eps)
        h2, w2 = min(args.res_b - 1, h2 + h_eps), min(args.res_b - 1, w2 + w_eps)

        borders = np.array([h1, h2, w1, w2])
        borders = np.repeat(borders[None, :], n_frames, axis=0)

    borders = np.ascontiguousarray(borders.astype(np.float32))
    borders = const_b * (2 * borders / (args.res_b - 1) - 1)
    # borders -> [n_frames, 4]

    return borders


def build_single(args, weights, var):
    weights = np.ascontiguousarray(weights).astype(np.float32)

    # Detect bounding box (real numbers) for fractal, so that it covers the majority of the canvas.
    borders = detect_borders(weights=weights, var=var, args=args)

    frames = chaos_game(
        weights=weights,
        borders=borders,
        batch_size=args.bs_point,
        res=args.res,
        iter=args.iter,
        iter_skip=args.iter_skip,
        var=var,
    )
    # frames -> [n_frames, height, width]
    frames = postprocess(frames=frames, img=args.img)
    
    return frames


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
            params = np.load(args.paths_param[idx_class])
            name_class = pathlib.Path(args.paths_param[idx_class]).stem
            
            dir_out = os.path.join(args.dir_out, name_class)
            filename = str(step)
            
        else:
            if args.img:
                params = sample_params_img(var=args.var)
            else:
                params = sample_params_video(var=args.var)
                
            dir_out = args.dir_out
            filename = f'{step}-{params["var"]}'                

        if args.img:
            weights = params['weights']
            if is_labeled:
                weights = mutate_weights(weights)

            frames = build_single(args=args, weights=weights, var=params['var'])
            
            path_out = [
                os.path.join(dir_out, f'{filename}-{idx}.jpg')
                for idx in range(weights.shape[0])]
            
            save_imgs(paths=path_out, frames=frames)

        else:
            weights = compose_params(params, linear_motion=args.linear_motion)
            if is_labeled:
                weights = mutate_weights(weights)

            frames = build_single(args=args, weights=weights, var=params['var'])
            
            path_out = os.path.join(dir_out, f'{filename}.mp4')
            save_video(frames=frames, path=path_out, fps=args.fps, lib=args.lib_save)            

        update_size_counter(counter=counter_size, path=path_out)

        time_run = time.time() - time_start

        n_created = step * len(frames) if args.img else step
        if args.samples and n_created >= args.samples:
            last_iter = True

        if args.time and time_run >= args.time:
            last_iter = True

        if step % args.print_every == 0 or last_iter:
            print_status(args=args, n_created=n_created, time_run=time_run, counter_size=counter_size)

        if last_iter:
            break

        step += 1
        if is_labeled:
            idx_class = (idx_class + 1) % n_classes


def compile_numba(args):
    args = copy.deepcopy(args)

    args.n_images = 4
    args.res, args.res_b = 128, 128
    args.iter, args.iter_b = 3000, 3000
    args.iter_skip = 100
    args.var = [20]

    if args.img:
        params = sample_params_img(var=args.var)
        weights = params['weights']
    else:
        params = sample_params_video(var=args.var)
        weights = compose_params(params)
    
    _ = build_single(args=args, weights=weights, var=params['var'])


def generate_classes(args):
    modality = 'image' if args.img else 'video'
    print('\n' * 3 + '#' * 80)
    print(f'Sampling parameters for {args.gen_param} {modality} classes...')
    print('#' * 80 + '\n' * 3)

    for step in range(1, 1 + args.gen_param):
        if args.img:
            params = sample_params_img(var=args.var)
        else:
            params = sample_params_video(var=args.var)

        path = os.path.join(args.dir_out, f'{step}-{params["var"]}.npz')
        np.savez(path, **params)


def build(args):
    assert args.dir_in is None or args.gen_param is None
    os.makedirs(args.dir_out, exist_ok=True)

    if args.gen_param is not None:
        generate_classes(args)

    else:
        assert args.samples or args.time, 'Provide "samples" or "time"'
        compile_numba(args) 
        build_dataset(args)
        