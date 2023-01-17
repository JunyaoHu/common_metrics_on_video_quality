import numpy as np
import torch
from tqdm import tqdm
import cv2
 
def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
 
 
def calculate_ssim_function(img1, img2):
    # [0,1]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ssim(batches1, batches2, calculate_per_frame):
    print("calculate_ssim...")

    # batches [batch_num, batch_size, timestamps, channel, h, w]
    
    assert batches1.shape == batches2.shape

    # videos [batch_num * batch_size, timestamps, channel, h, w]
    videos1 = torch.flatten(batches1, start_dim=0, end_dim=1)
    videos2 = torch.flatten(batches2, start_dim=0, end_dim=1)
    
    ssim_results = []
    
    for video_num in tqdm(range(len(videos1))):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        ssim_results_of_a_video = []
        for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w]
            # img [h, w, channel] permute(1,2,0)

            img1 = video1[clip_timestamp-1].permute(1,2,0).numpy()
            img2 = video2[clip_timestamp-1].permute(1,2,0).numpy()
            
            # calculate ssim of a video
            ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))

        ssim_results.append(ssim_results_of_a_video)

    result = {
        "ssim_video_setting": batches1.shape,
        "ssim_per_frame": calculate_per_frame,
        "ssim_mean_per_frame": np.mean(ssim_results, axis=0),
        "ssim_std_per_frame": np.std(ssim_results, axis=0)
    }

    return result

# test code / using example

def main():
    NUMBER_OF_BATCHES = 4
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    batches1 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    batches2 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    print(calculate_ssim(batches1, batches2, CALCULATE_PER_FRAME))

if __name__ == "__main__":
    main()