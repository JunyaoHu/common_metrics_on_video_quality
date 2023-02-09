# common_metrics_on_video_quality

you can calculate Frechét Video Distance (FVD), the peak-signal-to-noise ratio (PSNR), the structural similarity index measure (SSIM) and the learned perceptual image patch similarity (LPIPS) easily.

As for FVD, this code is from [MVCD model project](https://github.com/voletiv/mcvd-pytorch) and other websites and projects, I've just extracted the part of it that's relevant to the calculation. This code can be used to evaluate FVD scores for generative or predictive models. 

ps: pixel value should be in [0, 1], support grayscale and RGB.

# Example

8 videos of a batch, 30 frames, 3 channels, 64x64 size, calculate per 8 frames, and calculate to final.

```
import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
CALCULATE_PER_FRAME = 8
CALCULATE_FINAL = True
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
result['ssim'] = calculate_ssim(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
result['psnr'] = calculate_psnr(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
result['lpips'] = calculate_lpips(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
print(json.dumps(result, indent=4))
```

it means we calculate `FVD-frames[:16]`, `FVD-frames[:24]`, `FVD-frames[:final]` (it means `FVD-frames[:30]`) , `avg-PSNR/SSIM/LPIPS-frame[:8]`, `avg-PSNR/SSIM/LPIPS-frame[:16]`, `avg-PSNR/SSIM/LPIPS-frame[:24]`, `avg-PSNR/SSIM/LPIPS-frame[:final]` (it means `avg-PSNR/SSIM/LPIPS-frame[:30]`) , and their std.

we cannot calculate `FVD-frames[:8]`, and it will pass when calculating, see ps.6.

a all-zero matrix and a all-one matrix, their FVD-30 is 151.17. We also calculate their standard deviation. Other metrics are the same.

```
{
    "fvd": {
        "fvd": {
            "[:16]": 153.11023578170108,
            "[:24]": 66.08097153313875,
            "final": 151.16806952692093
        },
        "fvd_per_frame": 8,
        "fvd_video_setting": [
            8,
            3,
            30,
            64,
            64
        ],
        "fvd_video_setting_name": "batch_size, channel, time, heigth, width"
    },
    "ssim": {
        "ssim": {
            "avg[:8]": 9.999000099990664e-05,
            "avg[:16]": 9.999000099990664e-05,
            "avg[:24]": 9.999000099990664e-05,
            "final": 9.999000099990664e-05
        },
        "ssim_std": {
            "std[:8]": 0.0,
            "std[:16]": 0.0,
            "std[:24]": 0.0,
            "final": 0.0
        },
        "ssim_per_frame": 8,
        "ssim_video_setting": [
            30,
            3,
            64,
            64
        ],
        "ssim_video_setting_name": "time, channel, heigth, width"
    },
    "psnr": {
        "psnr": {
            "avg[:8]": 0.0,
            "avg[:16]": 0.0,
            "avg[:24]": 0.0,
            "final": 0.0
        },
        "psnr_std": {
            "std[:8]": 0.0,
            "std[:16]": 0.0,
            "std[:24]": 0.0,
            "final": 0.0
        },
        "psnr_per_frame": 8,
        "psnr_video_setting": [
            30,
            3,
            64,
            64
        ],
        "psnr_video_setting_name": "time, channel, heigth, width"
    },
    "lpips": {
        "lpips": {
            "avg[:8]": 0.8140146732330322,
            "avg[:16]": 0.8140146732330322,
            "avg[:24]": 0.8140146732330322,
            "final": 0.8140146732330322
        },
        "lpips_std": {
            "std[:8]": 0.0,
            "std[:16]": 0.0,
            "std[:24]": 0.0,
            "final": 0.0
        },
        "lpips_per_frame": 8,
        "lpips_video_setting": [
            30,
            3,
            64,
            64
        ],
        "lpips_video_setting_name": "time, channel, heigth, width"
    }
}
```

# PS before run: 

1. you should download `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) or you can download `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI), you download random one and put it into fvd folder should be fine.
2. you should `pip install lpips`
3. pixel value should be in [0, 1].
4. for gray-scale, we muitiply to 3 channels [as it says](https://github.com/richzhang/PerceptualSimilarity/issues/23#issuecomment-492368812)
5. we average SSIM when images have 3 channels, ssim is the only metric extremely sensitive to gray being compared to b/w.
6. since frames num should > 10 to calculate FVD, so FVD calculation begins from the first multiple of CALCULATE_PER_FRAME greater than 10, like upper example.

If code cannot run correctly, please create an issue here~
