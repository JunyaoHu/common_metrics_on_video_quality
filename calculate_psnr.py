import numpy as np
import torch
from tqdm import tqdm
import math

def img_psnr(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def trans(x):
    return x

def calculate_psnr(videos1, videos2, calculate_per_frame, calculate_final):
    print("calculate_psnr...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)
    
    psnr_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()
            
            # calculate psnr of a video
            psnr_results_of_a_video.append(img_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)
    
    psnr_results = np.array(psnr_results)
    
    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        psnr[f'avg[:{clip_timestamp}]'] = np.mean(psnr_results[:,:clip_timestamp])
        psnr_std[f'std[:{clip_timestamp}]'] = np.std(psnr_results[:,:clip_timestamp])

    if calculate_final:
        psnr['final'] = np.mean(psnr_results)
        psnr_std['final'] = np.std(psnr_results)
    
    result = {
        "psnr": psnr,
        "psnr_std": psnr_std,
        "psnr_per_frame": calculate_per_frame,
        "psnr_video_setting": video1.shape,
        "psnr_video_setting_name": "time, channel, heigth, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    CALCULATE_FINAL = True
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    import json
    result = calculate_psnr(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()