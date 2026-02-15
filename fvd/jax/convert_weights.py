"""One-time script to convert PyTorch I3D weights to JAX-compatible .npz format.

Usage:
    python -m fvd.jax.convert_weights
"""
import os
import numpy as np
import torch


def convert(pt_path=None, out_path=None):
    if pt_path is None:
        pt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'videogpt', 'i3d_pretrained_400.pt')
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'i3d_pretrained_400.npz')

    sd = torch.load(pt_path, map_location='cpu')
    params = {}
    for key, val in sd.items():
        arr = val.numpy()
        # Map PyTorch naming to flat JAX naming with '/' separator
        # e.g. "Conv3d_1a_7x7.conv3d.weight" -> "Conv3d_1a_7x7/conv_w"
        #      "Mixed_3b.b0.bn.weight"        -> "Mixed_3b/b0/bn_scale"
        jax_key = key.replace('.', '/')
        jax_key = jax_key.replace('/conv3d/weight', '/conv_w')
        jax_key = jax_key.replace('/conv3d/bias', '/conv_b')
        jax_key = jax_key.replace('/bn/weight', '/bn_scale')
        jax_key = jax_key.replace('/bn/bias', '/bn_offset')
        jax_key = jax_key.replace('/bn/running_mean', '/bn_mean')
        jax_key = jax_key.replace('/bn/running_var', '/bn_var')
        if 'num_batches_tracked' in jax_key:
            continue
        # Transpose conv weights: PyTorch [O, I, D, H, W] -> JAX [D, H, W, I, O]
        if jax_key.endswith('/conv_w'):
            arr = np.transpose(arr, (2, 3, 4, 1, 0))
        params[jax_key] = arr

    np.savez(out_path, **params)
    print(f"Saved {len(params)} arrays to {out_path}")
    return out_path


if __name__ == '__main__':
    convert()
