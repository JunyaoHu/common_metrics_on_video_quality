"""Pure functional JAX implementation of Inception I3D for FVD feature extraction.

All functions are stateless: model parameters are passed explicitly as dicts.
Input/output tensors use NDHWC layout (batch, depth/time, height, width, channels).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


def unit3d(params, x, stride=(1, 1, 1), use_bn=True, use_relu=True):
    """Conv3D (SAME padding) + optional BatchNorm (inference mode) + optional ReLU."""
    x = lax.conv_general_dilated(
        x, params['conv_w'],
        window_strides=stride,
        padding='SAME',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    if 'conv_b' in params:
        x = x + params['conv_b']
    if use_bn:
        x = ((x - params['bn_mean'])
             / jnp.sqrt(params['bn_var'] + 1e-5)
             * params['bn_scale']
             + params['bn_offset'])
    if use_relu:
        x = jax.nn.relu(x)
    return x


def maxpool3d(x, window, stride):
    """Max-pool with SAME padding. window/stride are 3-tuples (D, H, W)."""
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1,) + window + (1,),
        window_strides=(1,) + stride + (1,),
        padding='SAME',
    )


def avgpool3d(x, window, stride):
    """Average-pool with VALID padding. window/stride are 3-tuples (D, H, W)."""
    sum_pool = lax.reduce_window(
        x, 0.0, lax.add,
        window_dimensions=(1,) + window + (1,),
        window_strides=(1,) + stride + (1,),
        padding='VALID',
    )
    return sum_pool / (window[0] * window[1] * window[2])


def inception_module(params, x):
    """One Inception block with 4 branches, concatenated along channel axis."""
    b0 = unit3d(params['b0'], x)
    b1 = unit3d(params['b1b'], unit3d(params['b1a'], x))
    b2 = unit3d(params['b2b'], unit3d(params['b2a'], x))
    b3 = unit3d(params['b3b'], maxpool3d(x, (3, 3, 3), (1, 1, 1)))
    return jnp.concatenate([b0, b1, b2, b3], axis=-1)


def i3d_forward(params, x):
    """Full I3D forward pass.

    Args:
        params: dict of arrays loaded from the converted .npz file.
        x: input tensor of shape [B, T, H, W, C] in [-1, 1].

    Returns:
        Logits of shape [B, 400].
    """
    # Stem
    x = unit3d(params['Conv3d_1a_7x7'], x, stride=(2, 2, 2))
    x = maxpool3d(x, (1, 3, 3), (1, 2, 2))
    x = unit3d(params['Conv3d_2b_1x1'], x)
    x = unit3d(params['Conv3d_2c_3x3'], x)
    x = maxpool3d(x, (1, 3, 3), (1, 2, 2))

    # Inception blocks 3
    x = inception_module(params['Mixed_3b'], x)
    x = inception_module(params['Mixed_3c'], x)
    x = maxpool3d(x, (3, 3, 3), (2, 2, 2))

    # Inception blocks 4
    x = inception_module(params['Mixed_4b'], x)
    x = inception_module(params['Mixed_4c'], x)
    x = inception_module(params['Mixed_4d'], x)
    x = inception_module(params['Mixed_4e'], x)
    x = inception_module(params['Mixed_4f'], x)
    x = maxpool3d(x, (2, 2, 2), (2, 2, 2))

    # Inception blocks 5
    x = inception_module(params['Mixed_5b'], x)
    x = inception_module(params['Mixed_5c'], x)

    # Head
    x = avgpool3d(x, (2, 7, 7), (1, 1, 1))
    x = unit3d(params['logits'], x, use_bn=False, use_relu=False)

    # Squeeze spatial dims and average over time
    # x shape: [B, T', 1, 1, 400] -> [B, 400]
    x = x[:, :, 0, 0, :]
    x = jnp.mean(x, axis=1)
    return x


def _nest_params(flat):
    """Convert flat '/' separated keys into a nested dict for inception_module access.

    E.g. flat['Mixed_3b/b0/conv_w'] -> nested['Mixed_3b']['b0']['conv_w']
    """
    nested = {}
    for key, val in flat.items():
        parts = key.split('/')
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


def load_params(npz_path):
    """Load .npz and return nested param dict ready for i3d_forward."""
    data = dict(np.load(npz_path))
    # Convert to jax arrays
    data = {k: jnp.array(v) for k, v in data.items()}
    return _nest_params(data)
