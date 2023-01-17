from models.fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained, to_i3d
import numpy as np
import torch
from tqdm import tqdm

def calculate_fvd(batches1, batches2, calculate_per_frame, device):
    print("calculate_fvd...")

    # videos [batch_num, batch_size, timestamps, channel, h, w]
    
    assert batches1.shape == batches2.shape
    
    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    for batch_num in tqdm(range(len(batches1))):
        # get a batch
        # videos [batch_size, timestamps, channel, h, w]
        videos1 = batches1[batch_num]
        videos2 = batches2[batch_num]

        # BTCHW -> BCTHW
        # videos [batch_size, channel, timestamps, h, w]
        videos1 = to_i3d(videos1, videos1.shape[-3], videos1.shape[-2], videos1.shape[-1])
        videos2 = to_i3d(videos2, videos2.shape[-3], videos2.shape[-2], videos2.shape[-1])
        fvd_results_of_a_batch = []

        assert calculate_per_frame >= 10

        for clip_timestamp in range(calculate_per_frame, videos1.shape[-3]+1, calculate_per_frame):
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
            # calculate FVD of a batch
            fvd_results_of_a_batch.append(frechet_distance(feats1, feats2))
        fvd_results.append(fvd_results_of_a_batch)

    result = {
        "fvd_video_setting": batches1.shape,
        "fvd_per_frame": calculate_per_frame,
        "fvd_mean_per_frame": np.mean(fvd_results, axis=0),
        "fvd_std_per_frame": np.std(fvd_results, axis=0)
    }

    return result

# test code / using example

def main():
    NUMBER_OF_BATCHES = 4
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 15
    batches1 = torch.zeros(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    batches2 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    print(calculate_fvd(batches1, batches2, CALCULATE_PER_FRAME, device))

if __name__ == "__main__":
    main()