import numpy as np
import torch
from tqdm import tqdm
import math

def psnr(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr

def calculate_psnr(batches1, batches2, calculate_per_frame):
    print("calculate_psnr...")

    # batches [batch_num, batch_size, timestamps, channel, h, w]
    
    assert batches1.shape == batches2.shape

    # videos [batch_num * batch_size, timestamps, channel, h, w]
    videos1 = torch.flatten(batches1, start_dim=0, end_dim=1)
    videos2 = torch.flatten(batches2, start_dim=0, end_dim=1)
    
    psnr_results = []
    
    for video_num in tqdm(range(len(videos1))):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp-1].numpy()
            img2 = video2[clip_timestamp-1].numpy()
            
            # calculate psnr of a video
            psnr_results_of_a_video.append(psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)

    result = {
        "psnr_video_setting": batches1.shape,
        "psnr_per_frame": calculate_per_frame,
        "psnr_mean_per_frame": np.mean(psnr_results, axis=0),
        "psnr_std_per_frame": np.std(psnr_results, axis=0)
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
    print(calculate_psnr(batches1, batches2, CALCULATE_PER_FRAME))

if __name__ == "__main__":
    main()