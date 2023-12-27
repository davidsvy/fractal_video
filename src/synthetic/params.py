import math
import numpy as np
import random
import scipy.interpolate



def interp_linear(size, start=0, end=1, *args, **kwargs):
    return np.linspace(start, end, size)


def interp_sharp(size):
    n_sharp = random.randint(4, 8)
    n_min = 4
    n_start = random.randint(n_min, size - n_sharp - n_min)
    n_end = size - n_sharp - n_start
    interp = interp_linear(n_sharp)
    interp = np.pad(interp, pad_width=(n_start, n_end), constant_values=(0, 1))
    interp = interp[:size]

    return interp


def interp_curve(size):
    n_points = random.randint(5, 9)
    sample_x = interp_linear(n_points)

    for _ in range(6):
        sample_y = np.random.uniform(0., 1., size=n_points)

        if np.abs(sample_y[1:] - sample_y[:-1]).mean() > 0.45:
            break

    interp_fn = scipy.interpolate.interp1d(
        sample_x, sample_y, kind='quadratic')
    _interp_linear = interp_linear(size)
    interp = interp_fn(_interp_linear)

    return interp


def random_sin(period, n_periods):
    """Based on:
    https://stackoverflow.com/a/64971796
    https://stackoverflow.com/a/19777462
    """
    step = 0.01

    freq_x = np.arange(0, (n_periods + 1) * 2 * np.pi, np.pi)
    freq_y = np.random.uniform(0.15, 1.7, size=freq_x.shape)
    appl_y = np.random.uniform(0.6, 1., size=freq_x.shape)

    x = np.arange(0, n_periods * 2 * np.pi, step)
    dx = np.full_like(x, step)

    freq_interp = scipy.interpolate.interp1d(freq_x, freq_y, kind='quadratic')
    freq = freq_interp(x)

    ampl_interp = scipy.interpolate.interp1d(freq_x, appl_y, kind='quadratic')
    ampl = ampl_interp(x)

    distorted_x = (freq * dx).cumsum()
    sin = (np.sin(distorted_x - np.pi / 2) * ampl + 1) / 2
    quantized_sin = sin[::len(sin) // (period * n_periods)]

    return quantized_sin


def interp_periodic(size):
    period = random.randint(5, 15)
    n_periods = math.ceil(size / period)
    amplitude = random.uniform(1.5, 4.0) * period / 100

    interp = random_sin(period=period, n_periods=n_periods)
    interp = (amplitude * interp)[:size]

    return interp


def sample_interp(n_fn, n_frames=None, linear_only=False):
    PROB_SINGLE_INTERP = 0.4
    PROB_LINEAR_ONLY = 0.2

    all_fn = [
        (interp_sharp, 0.2),
        (interp_curve, 0.55),
        (interp_periodic, 0.25),
    ]

    list_fn, list_prob = zip(*all_fn)
    assert min(list_prob) > 0
    list_prob = np.array(list_prob)
    list_prob = list_prob / list_prob.sum()

    if n_frames is None:
        n_frames = random.randint(18, 20)

    if random.random() < PROB_LINEAR_ONLY:
        fns_chosen = interp_linear(n_frames)
        fns_chosen = np.repeat(fns_chosen[:, None], repeats=n_fn, axis=1)

    elif random.random() < PROB_SINGLE_INTERP:
        fns_chosen = random.choices(
            list_fn, weights=list_prob, k=1)[0](n_frames)
        fns_chosen = np.repeat(fns_chosen[:, None], repeats=n_fn, axis=1)

    else:
        fns_chosen = random.choices(
            list_fn, weights=list_prob, k=min(n_fn, random.randint(1, 2)))
        fns_chosen.append(interp_linear)
        fns_chosen = [fn(n_frames) for fn in fns_chosen]
        fns_chosen = random.choices(fns_chosen, k=n_fn)
        fns_chosen = np.stack(fns_chosen, axis=-1)

    # fns_chosen -> [n_frames, n_fn]
    return fns_chosen


def interp_array(start, end, interp):
    # start, end -> [...]
    # interp -> [n_frames, [start prefix]]
    n_dim = start.ndim - interp.ndim + 1

    interp = interp[(...,) + (None,) * n_dim]
    # interp -> [n_frames, ...]
    start, end = start[None, ...], end[None, ...]
    # start, end -> [1, ...]
    output = start * (1 - interp) + end * interp
    # output -> [n_frames, ...]

    return output


def random_curve(n_frames, eps=0.35):
    n_points = random.randint(4, 7)
    sample_x = interp_linear(n_points)

    sample_y = np.random.uniform(1 - eps, 1 + eps, size=n_points)

    interp_fn = scipy.interpolate.interp1d(
        sample_x, sample_y, kind='quadratic')
    _interp_linear = interp_linear(n_frames)
    interp = interp_fn(_interp_linear)
    # interp -> [n_frames]

    return interp