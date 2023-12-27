import numba
import numpy as np

from .sampling import calculate_probs, weighted_sampling
from .variations import *


@numba.njit(parallel=True, fastmath=True, cache=True)
def chaos_game_step(coords, weights, probs_J, probs_q, var=0):
    # coords -> [n_images, batch_size, 2]
    # weights -> [n_images, n_fn, 6]
    # probs_J, probs_q -> [n_images, n_fn]
    n_images, batch_size = coords.shape[:2]

    idxs_fn = weighted_sampling(
        size=batch_size, probs_J=probs_J, probs_q=probs_q)
    # idxs_fn -> [n_images, batch_size]

    for img in numba.prange(n_images):
        for batch in numba.prange(batch_size):
            idx_fn = idxs_fn[img, batch]
            coords[img, batch] = (
                (
                    coords[img, batch, 0] * weights[img, idx_fn, 0]
                    + coords[img, batch, 1] * weights[img, idx_fn, 1]
                    + weights[img, idx_fn, 2]
                ),
                (
                    coords[img, batch, 0] * weights[img, idx_fn, 3]
                    + coords[img, batch, 1] * weights[img, idx_fn, 4]
                    + weights[img, idx_fn, 5]
                ),
            )

    if var == 0:
        return

    # Very ugly, but numba does not work otherwise
    if var in [1, 14, 18, 20, 29, 31, 42, 44, 48]:
        if var == 1:
            coords = var_1(coords)
        elif var == 14:
            coords = var_14(coords)
        elif var == 18:
            coords = var_18(coords)
        elif var == 20:
            coords = var_20(coords)
        elif var == 29:
            coords = var_29(coords)
        elif var == 31:
            coords = var_31(coords)
        elif var == 42:
            coords = var_42(coords)
        elif var == 44:
            coords = var_44(coords)
        else:
            coords = var_48(coords)

    elif var in [2, 3, 4, 16, 27, 28]:
        if var == 2:
            coords = var_2(coords)
        elif var == 3:
            coords = var_3(coords)
        elif var == 4:
            coords = var_4(coords)
        elif var == 16:
            coords = var_16(coords)
        elif var == 27:
            coords = var_27(coords)
        else:
            coords = var_28(coords)

    elif var in [5, 6, 7, 8, 9, 10, 11, 12, 13, 19]:
        if var == 5:
            coords = var_5(coords)
        elif var == 6:
            coords = var_6(coords)
        elif var == 7:
            coords = var_7(coords)
        elif var == 8:
            coords = var_8(coords)
        elif var == 9:
            coords = var_9(coords)
        elif var == 10:
            coords = var_10(coords)
        elif var == 11:
            coords = var_11(coords)
        elif var == 12:
            coords = var_12(coords)
        elif var == 13:
            coords = var_13(coords)
        else:
            coords = var_19(coords)

    elif var in [15, 17]:
        if var == 15:
            coords = var_15(coords, weights, idxs_fn)
        else:
            coords = var_17(coords, weights, idxs_fn)

    elif var in [21, 22]:
        if var == 21:
            coords = var_21(coords, weights, idxs_fn)
        else:
            coords = var_22(coords, weights, idxs_fn)

    elif var == 45:
        coords = var_45(coords)


@numba.njit(parallel=True, fastmath=True, cache=True)
def update_canvas(canvas, pixels, res, n_frames, batch_size):
    for img in numba.prange(n_frames):
        for batch in numba.prange(batch_size):
            if (0 <= pixels[img, batch, 0] < res) and (0 <= pixels[img, batch, 1] < res):
                """
                canvas[
                    img, pixels[img, batch, 0], pixels[img, batch, 1]] += 1
                """
                for h in range(max(0, pixels[img, batch, 0] - 1), min(res, pixels[img, batch, 0] + 2)):
                    for w in range(max(0, pixels[img, batch, 1] - 1), min(res, pixels[img, batch, 1] + 2)):
                        canvas[img, h, w] += 1


@numba.njit(parallel=True, fastmath=True, cache=True)
def update_canvas_binary(canvas, pixels, res, n_frames, batch_size):
    for img in numba.prange(n_frames):
        for batch in numba.prange(batch_size):
            if (0 <= pixels[img, batch, 0] < res) and (0 <= pixels[img, batch, 1] < res):
                for h in range(max(0, pixels[img, batch, 0] - 1), min(res, pixels[img, batch, 0] + 2)):
                    for w in range(max(0, pixels[img, batch, 1] - 1), min(res, pixels[img, batch, 1] + 2)):
                        canvas[img, h, w] = True


@numba.njit(parallel=True, fastmath=True, cache=True)
def chaos_game(weights, borders, batch_size=128, res=512, iter=10000, iter_skip=100, var=0):
    # weights -> [n_frames, n_fn, 6]
    # borders -> [n_frames, 4]
    n_frames = weights.shape[0]

    coef_norm, offset_norm = np.empty(shape=(2, n_frames, 2), dtype=np.float32)
    # coef_norm, offset_norm -> [n_frames, 2]
    for f in numba.prange(n_frames):
        coef_norm[f] = (
            (res - 1) / (borders[f, 1] - borders[f, 0]),
            (res - 1) / (borders[f, 3] - borders[f, 2]),
        )

        offset_norm[f] = (
            -borders[f, 0] * coef_norm[f, 0],
            -borders[f, 2] * coef_norm[f, 1],
        )

    canvas = np.zeros(shape=(n_frames, res, res), dtype=np.uint16)
    # canvas -> [n_frames, height, width]
    coords = np.empty(shape=(n_frames, batch_size, 2), dtype=np.float32)
    pixels = np.empty(shape=(n_frames, batch_size, 2), dtype=np.int16)
    # coords, pixels -> [n_frames, batch_size, 2]

    coords_init = np.random.uniform(
        a=-1, b=1, size=(batch_size, 2)).astype(np.float32)
    for idx in numba.prange(n_frames):
        coords[idx] = coords_init

    probs_J, probs_q = calculate_probs(weights)
    # probs_J, probs_q -> [n_frames, n_fn]

    for step in range(iter):
        chaos_game_step(
            coords=coords,
            weights=weights,
            probs_J=probs_J,
            probs_q=probs_q,
            var=var,
        )
        # coords -> [n_frames, batch_size, 2]

        # The initial iterations are ignored as the coordinates have not converged yet.
        if step < iter_skip:
            continue

        for f in numba.prange(n_frames):
            for b in numba.prange(batch_size):
                pixels[f, b] = (
                    coords[f, b, 0] * coef_norm[f, 0] + offset_norm[f, 0],
                    coords[f, b, 1] * coef_norm[f, 1] + offset_norm[f, 1],
                )

        # pixels -> [n_frames, batch_size, 2]

        update_canvas(
            canvas=canvas,
            pixels=pixels,
            res=res,
            n_frames=n_frames,
            batch_size=batch_size,
        )

    return canvas


@numba.njit(parallel=True, fastmath=True, cache=True)
def chaos_game_binary(weights, borders, batch_size=128, res=512, iter=10000, iter_skip=100, var=0):
    # weights -> [n_frames, n_fn, 4]
    # borders -> [n_frames, 4]
    n_frames = weights.shape[0]

    coef_norm, offset_norm = np.empty(shape=(2, n_frames, 2), dtype=np.float32)
    # coef_norm, offset_norm -> [n_frames, 2]
    for f in numba.prange(n_frames):
        coef_norm[f] = (
            (res - 1) / (borders[f, 1] - borders[f, 0]),
            (res - 1) / (borders[f, 3] - borders[f, 2]),
        )

        offset_norm[f] = (
            -borders[f, 0] * coef_norm[f, 0],
            -borders[f, 2] * coef_norm[f, 1],
        )

    canvas = np.zeros(shape=(n_frames, res, res), dtype=np.bool_)
    # canvas -> [n_frames, height, width]
    coords = np.empty(shape=(n_frames, batch_size, 2), dtype=np.float32)
    pixels = np.empty(shape=(n_frames, batch_size, 2), dtype=np.int16)
    # coords, pixels -> [n_frames, batch_size, 2]

    coords_init = np.random.uniform(
        a=-1, b=1, size=(batch_size, 2)).astype(np.float32)
    for idx in numba.prange(n_frames):
        coords[idx] = coords_init

    probs_J, probs_q = calculate_probs(weights)
    # probs_J, probs_q -> [n_frames, n_fn]

    for step in range(iter):
        chaos_game_step(
            coords=coords,
            weights=weights,
            probs_J=probs_J,
            probs_q=probs_q,
            var=var,
        )
        # coords -> [n_frames, batch_size, 2]

        # The initial iterations are ignored as the coordinates have not converged yet.
        if step < iter_skip:
            continue

        for f in numba.prange(n_frames):
            for b in numba.prange(batch_size):
                pixels[f, b] = (
                    coords[f, b, 0] * coef_norm[f, 0] + offset_norm[f, 0],
                    coords[f, b, 1] * coef_norm[f, 1] + offset_norm[f, 1],
                )
        # pixels -> [n_frames, batch_size, 2]

        update_canvas_binary(
            canvas=canvas,
            pixels=pixels,
            res=res,
            n_frames=n_frames,
            batch_size=batch_size,
        )

    return canvas


@numba.njit(parallel=True, fastmath=True, cache=True)
def compute_features(frames):
    # frames -> [T, H, W]
    T, H, W = frames.shape
    frames_float = frames.astype(np.float32)

    pixel_diff = np.logical_xor(frames[1:], frames[:-1]).sum()

    pixel_std = 0.0
    for h in numba.prange(H):
        for w in numba.prange(W):
            pixel_std += np.std(frames_float[:, h, w])

    _area = np.empty(shape=(T,), dtype=np.int32)
    for t in numba.prange(T):
        _area[t] = frames[t].sum()

    area = _area.sum()

    area_diff = np.abs(_area[1:] - _area[:-1]).sum()
    area_std = np.std(_area.astype(np.float32))

    return pixel_diff, pixel_std, area_diff, area_std, area
