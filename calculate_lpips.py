import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def calculate_lpips(batches1, batches2, calculate_per_frame, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1] , need (1,3,64,64)
    print("calculate_lpips...")

    # batches [batch_num, batch_size, timestamps, channel, h, w]
    
    assert batches1.shape == batches2.shape

    # videos [batch_num * batch_size, timestamps, channel, h, w]
    # value [0, 1] -> [-1, 1]
    videos1 = torch.flatten(batches1, start_dim=0, end_dim=1) * 2 - 1
    videos2 = torch.flatten(batches2, start_dim=0, end_dim=1) * 2 - 1
    
    lpips_results = []
    
    for video_num in tqdm(range(len(videos1))):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp-1].unsqueeze(0).cuda()
            img2 = video2[clip_timestamp-1].unsqueeze(0).cuda()
            
            loss_fn.to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu())

        lpips_results.append(lpips_results_of_a_video)

    result = {
        "lpips_video_setting": batches1.shape,
        "lpips_per_frame": calculate_per_frame,
        "lpips_mean_per_frame": np.mean(lpips_results, axis=0),
        "lpips_std_per_frame": np.std(lpips_results, axis=0)
    }

    return result

# test code / using example

def main():
    NUMBER_OF_BATCHES = 4
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 1
    batches1 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    batches2 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    print(calculate_lpips(batches1, batches2, CALCULATE_PER_FRAME, device))

if __name__ == "__main__":
    main()