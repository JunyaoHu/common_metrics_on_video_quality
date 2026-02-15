import numpy as np
from tqdm import tqdm

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device=None, method='styleganv', only_final=False):

    if method == 'jax':
        return _calculate_fvd_jax(videos1, videos2, only_final=only_final)

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))

    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):

            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result


def _calculate_fvd_jax(videos1, videos2, only_final=False):
    import jax.numpy as jnp
    from fvd.jax.fvd import load_i3d_pretrained, get_fvd_logits, frechet_distance

    print("calculate_fvd (jax)...")

    # Accept jnp arrays or convert from torch/numpy
    if not isinstance(videos1, jnp.ndarray):
        videos1 = jnp.array(np.array(videos1))
    if not isinstance(videos2, jnp.ndarray):
        videos2 = jnp.array(np.array(videos2))

    assert videos1.shape == videos2.shape

    # Handle grayscale: BTCHW with C=1 -> repeat to C=3
    if videos1.shape[2] == 1:
        videos1 = jnp.repeat(videos1, 3, axis=2)
        videos2 = jnp.repeat(videos2, 3, axis=2)

    # BTCHW -> BCTHW
    videos1 = jnp.transpose(videos1, (0, 2, 1, 3, 4))
    videos2 = jnp.transpose(videos2, (0, 2, 1, 3, 4))

    params = load_i3d_pretrained()
    fvd_results = []

    if only_final:
        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"
        feats1 = get_fvd_logits(videos1, params)
        feats2 = get_fvd_logits(videos2, params)
        fvd_results.append(frechet_distance(feats1, feats2))
    else:
        for clip_timestamp in tqdm(range(10, videos1.shape[2] + 1)):
            videos_clip1 = videos1[:, :, :clip_timestamp]
            videos_clip2 = videos2[:, :, :clip_timestamp]
            feats1 = get_fvd_logits(videos_clip1, params)
            feats2 = get_fvd_logits(videos_clip2, params)
            fvd_results.append(frechet_distance(feats1, feats2))

    return {"value": fvd_results}

# test code / using example

def main():
    import torch
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    result = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=False)
    print("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False)
    print("[fvd-styleganv]", result["value"])

if __name__ == "__main__":
    main()
