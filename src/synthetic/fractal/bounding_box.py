import numba
import numpy as np


@numba.njit(parallel=True, fastmath=True)
def remove_empty(img, threshold, borders):
    # img -> [batch_size, height, width]
    # threshold -> [batch_size]
    # borders -> [batch_size, 4]
    batch_size, height, width = img.shape

    for b in numba.prange(batch_size):
        if threshold[b] == 0:
            continue

        h1, h2 = 0, height - 1
        while h1 < h2 and not np.any(img[b, h1]):
            h1 += 1
        while h2 > h1 and not np.any(img[b, h1]):
            h2 -= 1

        w1, w2 = 0, width - 1
        while w1 < w2 and not np.any(img[b, h1: h2 + 1, w1]):
            w1 += 1
        while w2 > w1 and not np.any(img[b, h1: h2 + 1, w2]):
            w2 -= 1

        borders[b] = (h1, h2, w1, w2)


@numba.njit(parallel=True, fastmath=True)
def remove_nonempty(img, count, threshold, borders):
    # img -> [batch_size, height, width]
    # count, threshold -> [batch_size]
    # borders -> [batch_size, 4]
    batch_size = img.shape[0]
    count_slice = np.empty(shape=(batch_size, 4), dtype=np.uint16)

    for b in numba.prange(batch_size):
        if count[b] == 0 or threshold[b] == 0:
            continue

        h1, h2, w1, w2 = borders[b]
        count_slice[b, 0] = np.count_nonzero(img[b, h1, w1: w2 + 1])
        count_slice[b, 1] = np.count_nonzero(img[b, h2, w1: w2 + 1])
        count_slice[b, 2] = np.count_nonzero(img[b, h1: h2 + 1, w1])
        count_slice[b, 3] = np.count_nonzero(img[b, h1: h2 + 1, w2])

        while count[b] > threshold[b]:
            idx_min = np.argmin(count_slice[b])
            count[b] -= count_slice[b, idx_min]

            if idx_min == 0:
                count_slice[b, 2] -= img[b, h1, w1]
                count_slice[b, 3] -= img[b, h1, w2]
                h1 += 1
                count_slice[b, 0] = np.count_nonzero(img[b, h1, w1: w2 + 1])

            elif idx_min == 1:
                count_slice[b, 2] -= img[b, h2, w1]
                count_slice[b, 3] -= img[b, h2, w2]
                h2 -= 1
                count_slice[b, 1] = np.count_nonzero(img[b, h2, w1: w2 + 1])

            elif idx_min == 2:
                count_slice[b, 0] -= img[b, h1, w1]
                count_slice[b, 1] -= img[b, h2, w1]
                w1 += 1
                count_slice[b, 2] = np.count_nonzero(img[b, h1: h2 + 1, w1])

            else:
                count_slice[b, 0] -= img[b, h1, w2]
                count_slice[b, 1] -= img[b, h2, w2]
                w2 -= 1
                count_slice[b, 3] = np.count_nonzero(img[b, h1: h2 + 1, w2])

        borders[b] = (h1, h2, w1, w2)


@numba.njit(parallel=True, fastmath=True)
def estimate_bounding_box(img, kept=0.9):
    # img -> [batch_size, height, width]
    batch_size, height, width = img.shape

    count = np.empty(shape=(batch_size,), dtype=np.uint32)
    threshold = np.empty(shape=(batch_size,), dtype=np.uint32)
    borders = np.empty(shape=(batch_size, 4), dtype=np.uint32)

    for b in numba.prange(batch_size):
        borders[b] = (0, height - 1, 0, width - 1)
        count[b] = img[b].sum()
        threshold[b] = kept * count[b]

    if kept == 1.:
        return borders

    remove_empty(img=img, threshold=threshold, borders=borders)

    remove_nonempty(
        img=img, count=count, threshold=threshold, borders=borders)
    # borders -> [batch_size, 4]

    return borders