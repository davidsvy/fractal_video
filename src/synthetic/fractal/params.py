import numpy as np
import random

from ..params import interp_array, interp_linear, sample_interp, random_curve


def sample_angle_img(n_frames, n_fn):
    angle = np.random.uniform(low=0, high=2 * np.pi, size=(2, n_frames, n_fn))
    # angle -> [2, n_frames, n_fn]
    cos_theta, cos_phi = np.cos(angle)
    sin_theta, sin_phi = np.sin(angle)
    # cos_theta, cos_phi, sin_theta, sin_phi -> [n_frames, n_fn]
    r_theta = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]]).transpose([2, 3, 0, 1])
    r_phi = np.array(
        [[cos_phi, -sin_phi], [sin_phi, cos_phi]]).transpose([2, 3, 0, 1])
    # r_theta, r_phi -> [n_frames, n_fn, 2, 2]

    return r_theta, r_phi


def sample_angle(n_fn):
    angle_start = np.random.uniform(
        low=0, high=2 * np.pi, size=(n_fn, 2))
    angle_end = angle_start + np.random.uniform(
        low=-np.pi, high=np.pi, size=(n_fn, 2))
    # angle_start, angle_end -> [n_fn, 2]
    angle = np.stack((angle_start, angle_end), axis=0)
    # angle -> [2, n_fn, 2]

    return angle


def compose_angle(angle, interp):
    # angle -> [2, n_fn, 2]
    # interp -> [n_frames, n_fn]

    angle = interp_array(start=angle[0], end=angle[1], interp=interp)
    # angle -> [n_frames, n_fn, 2]

    angle = angle.transpose(2, 0, 1)
    # angle -> [2, n_frames, n_fn]

    cos_theta, cos_phi = np.cos(angle)
    sin_theta, sin_phi = np.sin(angle)
    # cos_theta, cos_phi, sin_theta, sin_phi -> [n_frames, n_fn]

    r_theta = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]]).transpose([2, 3, 0, 1])
    r_phi = np.array(
        [[cos_phi, -sin_phi], [sin_phi, cos_phi]]).transpose([2, 3, 0, 1])
    # r_theta, r_phi -> [n_frames, n_fn, 2, 2]

    return r_theta, r_phi


def sample_sigma_aux(n_fn):
    sigma = np.zeros((n_fn, 2), dtype=np.float32)
    alpha = np.random.uniform(low=0.5 * (5 + n_fn), high=0.5 * (6 + n_fn))
    bound_low = alpha - 3 * n_fn + 3
    bound_high = alpha

    for idx in range(n_fn - 1):
        _bound_low = max(0, bound_low / 3)
        _bound_high = min(1, bound_high)
        sigma[idx, 0] = np.random.uniform(_bound_low, _bound_high)
        bound_low -= sigma[idx, 0]
        bound_high -= sigma[idx, 0]

        _bound_low = max(0, bound_low / 2)
        _bound_high = min(sigma[idx, 0], bound_high / 2)
        sigma[idx, 1] = np.random.uniform(_bound_low, _bound_high)
        bound_low -= 2 * sigma[idx, 1] - 3
        bound_high -= 2 * sigma[idx, 1]

    _bound_low = max(0, (bound_high - 1) / 2)
    _bound_high = bound_high / 3
    sigma[-1, 1] = np.random.uniform(_bound_low, _bound_high)
    sigma[-1, 0] = bound_high - 2 * sigma[-1, 1]
    # sigma -> [n_fn, 2]

    return sigma


def sample_sigma(n_frames, n_fn):
    sigma = np.zeros(shape=(n_frames, n_fn, 2))
    # sigma -> [n_frames, n_fn, 2]
    for idx in range(n_frames):
        sigma[idx] = sample_sigma_aux(n_fn)

    det_sort = np.argsort(-sigma[..., 0] * sigma[..., 1], axis=-1)
    # det_sort -> [n_frames, n_fn]
    for idx in range(n_frames):
        sigma[idx] = sigma[idx, det_sort[idx]]

    sigma = sigma.reshape(n_frames * n_fn, 2)
    # sigma -> [n_frames * n_fn, 2]
    sigma = np.stack([np.diag(x) for x in sigma])
    # sigma -> [n_frames * n_fn, 2, 2]
    sigma = sigma.reshape(n_frames, n_fn, 2, 2)
    # sigma -> [n_frames, n_fn, 2, 2]

    return sigma


def compose_sigma(sigma, interp):
    # sigma -> [2, n_fn, 2, 2]
    # interp -> [n_frames, n_fn]
    sigma = interp_array(start=sigma[0], end=sigma[1], interp=interp)
    # sigma -> [n_frames, n_fn, 2, 2]

    return sigma


def sample_delta(n_frames, n_fn):
    delta = np.random.choice([-1., 1.], size=[n_frames * n_fn, 2])
    delta = np.stack([np.diag(x) for x in delta])
    # delta -> [n_frames * n_fn, 2, 2]
    delta = delta.reshape(n_frames, n_fn, 2, 2)
    # delta -> [n_frames, n_fn, 2, 2]

    return delta


def sample_bias(n_frames, n_fn, is_constant=None):
    if is_constant is None:
        is_constant = random.random() < 0.2

    if is_constant:
        bias = np.random.uniform(low=-1, high=1, size=[1, n_fn, 2])
        # bias -> [1, n_fn, 2]
        bias = np.repeat(bias, repeats=n_frames, axis=0)

    else:
        bias = np.random.uniform(low=-1, high=1, size=[n_frames, n_fn, 2])

    # bias -> [n_frames, n_fn, 2]

    return bias, is_constant


def compose_bias(bias, interp, is_constant):
    # bias -> [2, n_fn, 2]
    # interp -> [n_frames, n_fn]
    n_frames = interp.shape[0]

    if is_constant:
        bias = np.repeat(bias[:1], repeats=n_frames, axis=0)
    else:
        bias = interp_array(start=bias[0], end=bias[1], interp=interp)
    # bias -> [n_frames, n_fn, 2]

    return bias


def sample_var(vars_arg):
    vars_nonlinear = [4, 14, 16, 17, 20, 27, 29]
    
    if vars_arg == [-1]:
        if random.random() < 0.5:
            var = 0
        else:
            var = random.choice(vars_nonlinear)
            
    else:
        var = random.choice(vars_arg)    
        
    return var


def sample_params_img(n_frames=16, var=[0]):
    params = {}
    
    params['var'] = sample_var(var)

    n_fn = random.randint(3, 8)

    r_theta, r_phi = sample_angle_img(n_frames=n_frames, n_fn=n_fn)
    # r_theta, r_phi -> [2, n_frames, n_fn, 2, 2]
    sigma = sample_sigma(n_frames=n_frames, n_fn=n_fn)
    # sigma -> [n_frames, n_fn, 2, 2]
    delta = sample_delta(n_frames=2 * n_frames, n_fn=n_fn)
    delta1, delta2 = delta.reshape(2, n_frames, n_fn, 2, 2)
    # delta1, delta2 -> [n_frames, n_fn, 2, 2]
    bias, _ = sample_bias(n_frames=n_frames, n_fn=n_fn, is_constant=False)
    # bias -> [n_frames, n_fn, 2]

    weights = np.einsum(
        'abcd, abde, abef, abfg, abgh -> abch', r_theta, delta1, sigma, r_phi, delta2)
    # weights -> [n_frames, n_fn, 2, 2]
    weights = np.concatenate((weights, bias[..., None]), axis=-1)
    # weights -> [n_frames, n_fn, 2, 3]
    weights = weights.reshape(n_frames, n_fn, -1)
    # weights -> [n_frames, n_fn, 6]

    params['weights'] = np.ascontiguousarray(weights).astype(np.float32)

    return params


def sample_params_video(n_fn=None, var=[0]):
    params = {}

    params['var'] = sample_var(var)
    
    if n_fn is None:
        n_fn = random.randint(3, 8)

    params['interp'] = sample_interp(n_fn)
    # interp -> [n_frames, n_fn]

    params['angle'] = sample_angle(n_fn)
    # angle -> [2, n_fn, 2]
    #params['sigma'] = sample_sigma1(n_frames=2, n_fn=n_fn)
    params['sigma'] = sample_sigma(n_frames=2, n_fn=n_fn)
    # sigma -> [2, n_fn, 2, 2]
    params['delta'] = sample_delta(n_frames=2, n_fn=n_fn)
    # delta -> [2, n_fn, 2, 2]
    params['bias'], params['is_constant'] = sample_bias(n_frames=2, n_fn=n_fn)
    # bias -> [2, n_fn, 2]

    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params[key] = np.ascontiguousarray(value).astype(np.float32)

    return params


def compose_params(params, linear_motion=False):
    """Based on:
    http://graphicsinterface.org/wp-content/uploads/gi1997-18.pdf
    """
    interp = params['interp']
    # interp -> [n_frames, n_fn]
    n_frames, n_fn = interp.shape
    #interp = sample_interp_singlesegm(interp.shape[1])

    if linear_motion:
        interp = interp_linear(n_frames)
        interp = np.repeat(interp[:, None], repeats=n_fn, axis=1)

    r_theta, r_phi = compose_angle(angle=params['angle'], interp=interp)
    # r_theta, r_phi -> [n_frames, n_fn, 2, 2]
    sigma = compose_sigma(sigma=params['sigma'], interp=interp)
    # sigma -> [n_frames, n_fn, 2, 2]
    delta = params['delta']
    # delta -> [2, n_fn, 2, 2]
    bias = compose_bias(
        bias=params['bias'], interp=interp, is_constant=params['is_constant'])
    # bias -> [n_frames, n_fn, 2]

    weights = np.einsum(
        'abcd, bde, abef, abfg, bgh -> abch', r_theta, delta[0], sigma, r_phi, delta[1])
    # weights -> [n_frames, n_fn, 2, 2]

    weights = np.concatenate((weights, bias[..., None]), axis=-1)
    # weights -> [n_frames, n_fn, 2, 3]
    weights = weights.reshape(n_frames, n_fn, -1)
    # weights -> [n_frames, n_fn, 6]
    weights = np.ascontiguousarray(weights).astype(np.float32)

    return weights


def mutate_weights(weights, ac=0.4, dc=0.2):
    # weights -> [n_frames, n_fn, d_weight]
    n_frames, n_fn, d_weight = weights.shape

    if isinstance(ac, float) and ac > 0:
        curve = [
            random_curve(n_frames=n_frames, eps=ac) for _ in range(d_weight)]
        curve = np.stack(curve, axis=1)[:, None]
        # curve -> [n_frames, 1, d_last]
        weights *= curve

    if isinstance(dc, float) and dc > 0:
        offset = np.random.uniform(-dc, dc, size=(1, n_fn, d_weight))
        # curve -> [1, n_fn, d_last]
        weights += offset

    return weights
