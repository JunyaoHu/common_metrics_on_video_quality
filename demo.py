import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips


# ps: pixel value should be in [0, 1]!

NUMBER_OF_BATCHES = 4
NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
CALCULATE_PER_FRAME = 10
batches1 = torch.zeros(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
batches2 = torch.ones(NUMBER_OF_BATCHES, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
print(calculate_fvd(batches1, batches2, CALCULATE_PER_FRAME, device))
print(calculate_psnr(batches1, batches2, CALCULATE_PER_FRAME))
print(calculate_ssim(batches1, batches2, CALCULATE_PER_FRAME))
print(calculate_lpips(batches1, batches2, CALCULATE_PER_FRAME, device))