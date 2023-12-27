import numba
import numpy as np

__all__ = [f'var_{idx}' for idx in [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    27, 28, 29, 31, 42, 44, 45, 48,
]
] + ['var_names', 'var_idxs']


"""All functions below are defined at:
The Fractal Flame Algorithm (pages 16 - 41)
    https://flam3.com/flame_draves.pdf
"""


EPS = np.array(1e-7, dtype=np.float32)
PI = np.array(np.pi, dtype=np.float32)
PI_HALF = np.array(np.pi / 2, dtype=np.float32)
ONE = np.array(1, dtype=np.float32)
TWO = np.array(2, dtype=np.float32)


@numba.njit(parallel=True, fastmath=True, cache=True)
def coord_r(x, y):
    return np.sqrt(np.square(x) + np.square(y))


@numba.njit(parallel=True, fastmath=True, cache=True)
def coord_theta(x, y):
    return np.arctan2(y, x)


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_0(coords):
    "Linear"
    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_1(coords):
    "Sinusoidal"
    # coords -> [n_frames, batch_size, 2]
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            coords[f, b] = (
                np.sin(coords[f, b, 0]),
                np.sin(coords[f, b, 1]),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_2(coords):
    "Spherical"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = np.square(coords[f, b, 0]) + np.square(coords[f, b, 1]) + EPS
            coords[f, b] /= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_3(coords):
    "Swirl"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = np.square(coords[f, b, 0]) + np.square(coords[f, b, 1])
            cos, sin = np.cos(r), np.sin(r)

            coords[f, b] = (
                coords[f, b, 0] * sin - coords[f, b, 1] * cos,
                coords[f, b, 0] * cos + coords[f, b, 1] * sin,
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_4(coords):
    "Horseshoe"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1]) + EPS
            coords[f, b] = (
                (coords[f, b, 0] - coords[f, b, 1]) *
                (coords[f, b, 0] + coords[f, b, 1]),
                TWO * coords[f, b, 0] * coords[f, b, 1],
            )

            coords[f, b] /= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_5(coords):
    "Polar"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            coords[f, b] = (
                coord_theta(coords[f, b, 0], coords[f, b, 1]) / PI,
                coord_r(coords[f, b, 0], coords[f, b, 1]) - ONE,
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_6(coords):
    "Handkerchief"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])

            coords[f, b] = (
                np.sin(theta + r),
                np.cos(theta - r),
            )

            coords[f, b] *= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_7(coords):
    "Heart"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])
            theta_r = theta * r

            coords[f, b] = (
                np.sin(theta_r),
                -np.cos(theta_r),
            )

            coords[f, b] *= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_8(coords):
    "Disc"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1]) * PI
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1]) / PI

            coords[f, b] = (
                np.sin(r),
                np.cos(r),
            )

            coords[f, b] *= theta

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_9(coords):
    "Spiral"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])

            coords[f, b] = (
                np.cos(theta) + np.sin(r),
                np.sin(theta) - np.cos(r),
            )

            coords[f, b] /= r + EPS

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_10(coords):
    "Hyperbolic"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])

            coords[f, b] = (
                np.sin(theta) / (r + EPS),
                np.cos(theta) * r,
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_11(coords):
    "Diamond"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])

            coords[f, b] = (
                np.sin(theta) * np.cos(r),
                np.cos(theta) * np.sin(r),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_12(coords):
    "Ex"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])
            p0, p1 = np.sin(theta + r) ** 3, np.cos(theta - r) ** 3

            coords[f, b] = (
                p0 + p1,
                p0 - p1,
            )

            coords[f, b] *= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_13(coords):
    "Julia"
    omega = PI * np.random.choice(2, size=()).astype(np.int32)

    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = np.sqrt(coord_r(coords[f, b, 0], coords[f, b, 1]))
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1]) / TWO + omega

            coords[f, b] = (
                np.cos(theta),
                np.sin(theta),
            )
            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_14(coords):
    "Bent"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            if coords[f, b, 0] < 0:
                coords[f, b, 0] *= 2

            if coords[f, b, 1] < 0:
                coords[f, b, 1] /= 2

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_15(coords, weights, idxs_fn):
    "Waves"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            i = idxs_fn[f, b]

            coords[f, b] = (
                (coords[f, b, 0]
                 + weights[f, i, 1] *
                 np.sin(coords[f, b, 1] / np.square(weights[f, i, 2]))
                 ),
                (coords[f, b, 1]
                 + weights[f, i, 4]
                 * np.sin(coords[f, b, 0] / np.square(weights[f, i, 5]))
                 ),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_16(coords):
    "Fisheye"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = 2 / (coord_r(coords[f, b, 0], coords[f, b, 1]) + 1)

            coords[f, b] = (
                coords[f, b, 1],
                coords[f, b, 0],
            )

            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_17(coords, weights, idxs_fn):
    "Popcorn"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            i = idxs_fn[f, b]

            coords[f, b] = (
                (coords[f, b, 0]
                 + weights[f, i, 2] * np.sin(np.tan(3 * coords[f, b, 1]))
                 ),
                (coords[f, b, 1]
                 + weights[f, i, 5] * np.sin(np.tan(3 * coords[f, b, 0]))
                 ),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_18(coords):
    "Exponential"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = np.exp(coords[f, b, 0] - 1)

            coords[f, b] = (
                np.cos(PI * coords[f, b, 1]),
                np.sin(PI * coords[f, b, 1]),
            )

            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_19(coords):
    "Power"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])
            sin = np.sin(theta)

            coords[f, b] = (np.cos(theta), sin)
            coords[f, b] *= np.power(r, sin)

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_20(coords):
    "Cosine"
    # cosh & sinh produce very large numbers and eventually nans.
    # Therefore, they are replaced with tanh.
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = np.tanh(coords[f, b, 1])
            coords[f, b] = (
                np.cos(PI * coords[f, b, 0]),
                -np.sin(PI * coords[f, b, 0]),
            )

            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_20_1(coords):
    "Cosine"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            coords[f, b] = (
                np.cos(PI * coords[f, b, 0]) * np.cosh(coords[f, b, 1]),
                -np.sin(PI * coords[f, b, 0]) * np.sinh(coords[f, b, 1]),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_21(coords, weights, idxs_fn):
    "Rings"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])

            c2 = np.square(weights[f, idxs_fn[f, b], 2])
            mult = np.mod(r + c2, 2 * c2) - c2 + r * (1 - c2)

            coords[f, b] = (
                np.cos(theta),
                np.sin(theta),
            )

            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_22(coords, weights, idxs_fn):
    "Fan"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            i = idxs_fn[f, b]
            r = coord_r(coords[f, b, 0], coords[f, b, 1])
            theta = coord_theta(coords[f, b, 0], coords[f, b, 1])
            t2 = 0.5 * PI * np.square(weights[f, i, 2])

            if np.mod(theta + weights[f, i, 5], 2 * t2) > t2:
                theta -= t2
            else:
                theta += t2

            coords[f, b] = (
                np.cos(theta),
                np.sin(theta),
            )

            coords[f, b] *= r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_27(coords):
    "Eyefish"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = 2 / (coord_r(coords[f, b, 0], coords[f, b, 1]) + 1)
            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_28(coords):
    "Bubble"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = 4 / (
                np.square(coords[f, b, 0]) + np.square(coords[f, b, 1]) + 4)
            coords[f, b] *= mult

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_29(coords):
    "Cylinder"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            coords[f, b, 0] = np.sin(coords[f, b, 0])

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_31(coords):
    "Noise"
    psi1, psi2 = np.random.rand(2).astype(np.float32)
    phase = 2 * PI * psi2
    sin, cos = np.sin(phase), np.sin(phase)

    coords *= psi1
    coords[..., 0] *= cos
    coords[..., 1] *= sin

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_42(coords):
    "Tangent"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            coords[f, b] = (
                np.sin(coords[f, b, 0]) / np.cos(coords[f, b, 1]),
                np.tan(coords[f, b, 1]),
            )

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_44(coords):
    "Rays"
    tan = np.tan(np.random.rand(1).astype(np.float32)[0] * PI)

    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = np.square(coords[f, b, 0]) + np.square(coords[f, b, 1]) + EPS

            coords[f, b] = (
                np.cos(coords[f, b, 0]),
                np.sin(coords[f, b, 1]),
            )

            coords[f, b] *= tan / r

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_45(coords):
    "Blade"
    psi = np.random.rand(1).astype(np.float32)[0]

    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            r = coord_r(coords[f, b, 0], coords[f, b, 1]) * psi
            sin, cos = np.sin(r), np.cos(r)
            x = coords[f, b, 0]

            coords[f, b] = (
                cos + sin,
                cos - sin,
            )

            coords[f, b] *= x

    return coords


@numba.njit(parallel=True, fastmath=True, cache=True)
def var_48(coords):
    "Cross"
    for f in numba.prange(coords.shape[0]):
        for b in numba.prange(coords.shape[1]):
            mult = (
                np.abs(np.square(coords[f, b, 0]) - np.square(coords[f, b, 1])) + EPS)
            coords[f, b] /= mult

    return coords


def get_all_vars():
    var_str, var_names, var_idxs = [], [], []

    for idx in range(49):
        try:
            str_ = f'var_{idx}'
            fn = eval(str_)
            name = fn.__doc__.split('\n')[0]

            var_str.append(str_)
            var_names.append(name)
            var_idxs.append(idx)

        except:
            pass

    return var_str, var_names, var_idxs


var_str, var_names, var_idxs = get_all_vars()
