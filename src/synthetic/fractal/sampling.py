import numpy as np
import numba


@numba.njit(parallel=False, fastmath=True)
def calculate_jq(probs):
    """Adapted from:
    https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
    """
    # probs -> [n_items]
    n_items = probs.shape[0]
    J = np.empty((n_items,), dtype=np.uint8)
    q = probs * n_items
    # J, q -> [n_items]

    smaller, larger = [], []
    for idx in range(n_items):
        if q[idx] < 1.0:
            smaller.append(idx)
        else:
            larger.append(idx)

    while len(smaller) > 0 and len(larger) > 0:
        small, large = smaller.pop(), larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


@numba.njit(parallel=True, fastmath=True, cache=True)
def calculate_probs(weights):
    # weights -> [n_frames, n_fn, 6]
    n_frames, n_fn = weights.shape[:2]

    probs = np.abs(
        weights[..., 0] * weights[..., 4] - weights[..., 1] * weights[..., 3])
    # probs -> [n_frames, n_fn]

    prob_norm = probs.sum(axis=1)
    # prob_norm -> [n_frames]

    for img in numba.prange(n_frames):
        probs[img] /= prob_norm[img]

    J = np.empty((n_frames, n_fn), dtype=np.uint8)
    q = np.empty((n_frames, n_fn), dtype=np.float32)
    # J, q -> [n_frames, n_fn]

    for img in numba.prange(n_frames):
        J[img], q[img] = calculate_jq(probs[img])

    return J, q


@numba.njit(parallel=True, fastmath=True)
def weighted_sampling(size, probs_J, probs_q):
    # probs_J, probs_q -> [n_frames, n_fn]
    n_frames, n_fn = probs_J.shape

    r1, r2 = np.random.rand(2, n_frames, size).astype(np.float32)
    output = np.empty((n_frames, size), dtype=np.uint8)
    # r1, r2, output -> [n_frames, size]

    for img in numba.prange(n_frames):
        for batch in numba.prange(size):
            kk = int(np.floor(r1[img, batch] * n_fn))

            if r2[img, batch] < probs_q[img, kk]:
                output[img, batch] = kk
            else:
                output[img, batch] = probs_J[img, kk]

    return output
