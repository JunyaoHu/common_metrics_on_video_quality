import torch
from calculate_fvd import calculate_fvd

# ps: pixel value should be in [0, 1] not [-1, 1]

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
