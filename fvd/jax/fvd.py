"""JAX implementation of FVD: preprocessing, I3D feature extraction, Fréchet distance."""
import os
import jax
import jax.numpy as jnp

from .i3d import i3d_forward, load_params
from .convert_weights import convert


def load_i3d_pretrained():
    """Load I3D params, converting from PyTorch weights on first call."""
    npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'i3d_pretrained_400.npz')
    if not os.path.exists(npz_path):
        pt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'videogpt', 'i3d_pretrained_400.pt')
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"PyTorch weights not found at {pt_path}. "
                "Run the videogpt FVD once first to download them, or place "
                "i3d_pretrained_400.pt in fvd/videogpt/.")
        print("Converting PyTorch I3D weights to JAX format...")
        convert(pt_path, npz_path)
    return load_params(npz_path)


def preprocess(videos):
    """Preprocess videos for I3D.

    Args:
        videos: jnp.ndarray of shape [B, C, T, H, W] in [0, 1].

    Returns:
        jnp.ndarray of shape [B, T, 224, 224, C] in [-1, 1].
    """
    # BCTHW -> BTHWC
    videos = jnp.transpose(videos, (0, 2, 3, 4, 1))
    b, t, h, w, c = videos.shape
    # Resize to 224x224
    # jax.image.resize works on the full array; resize each frame
    videos = jax.image.resize(videos, (b, t, 224, 224, c), method='bilinear')
    # [0, 1] -> [-1, 1]
    videos = videos * 2.0 - 1.0
    return videos


def get_fvd_logits(videos, params, bs=10):
    """Extract I3D logits from a batch of videos.

    Args:
        videos: jnp.ndarray of shape [B, C, T, H, W] in [0, 1].
        params: nested param dict from load_i3d_pretrained().
        bs: batch size for chunked inference.

    Returns:
        jnp.ndarray of shape [B, 400].
    """
    videos = preprocess(videos)
    logits = []
    for i in range(0, videos.shape[0], bs):
        batch = videos[i:i + bs]
        logits.append(i3d_forward(params, batch))
    return jnp.concatenate(logits, axis=0)


def _sqrtm_psd(mat):
    """Symmetric positive semi-definite matrix square root via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = jnp.maximum(eigvals, 0.0)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T


def _trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _sqrtm_psd(sigma)
    sqrt_a_sigmav_a = sqrt_sigma @ sigma_v @ sqrt_sigma
    # Symmetrize to avoid numerical asymmetry before eigh
    sqrt_a_sigmav_a = (sqrt_a_sigmav_a + sqrt_a_sigmav_a.T) / 2.0
    return jnp.trace(_sqrtm_psd(sqrt_a_sigmav_a))


def _cov(m):
    """Estimate covariance matrix. m: [N, D] where rows are observations."""
    m = m - jnp.mean(m, axis=0, keepdims=True)
    return (m.T @ m) / (m.shape[0] - 1)


def frechet_distance(x1, x2):
    """Compute Fréchet distance between two sets of I3D features.

    Args:
        x1, x2: jnp.ndarray of shape [B, 400].

    Returns:
        Scalar FVD value (float).
    """
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    m, m_w = jnp.mean(x1, axis=0), jnp.mean(x2, axis=0)
    sigma, sigma_w = _cov(x1), _cov(x2)
    mean_diff = jnp.sum((m - m_w) ** 2)
    if x1.shape[0] > 1:
        sqrt_trace = _trace_sqrt_product(sigma, sigma_w)
        trace = jnp.trace(sigma + sigma_w) - 2.0 * sqrt_trace
        fd = trace + mean_diff
    else:
        fd = mean_diff
    return float(fd)
