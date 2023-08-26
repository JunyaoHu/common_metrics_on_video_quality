# common_metrics_on_video_quality

You can easily calculate the following video quality metrics:

- FVD: Frechét Video Distance
- PSNR: peak-signal-to-noise ratio
- SSIM: structural similarity index measure
- LPIPS: learned perceptual image patch similarity

As for FVD, the code refers to [MVCD](https://github.com/voletiv/mcvd-pytorch) and other websites and projects, I've just extracted the part of it that's relevant to the calculation. This code can be used to evaluate FVD scores for generative or predictive models. 

- This project supports grayscale and RGB videos.
- This project supports Ubuntu, but maybe something is wrong with Windows. If you can solve it, welcome any PR.
- **If the project cannot run correctly, please give me an issue or PR~**

# Example

8 videos of a batch, 10 frames, 3 channels, 64x64 size.

```
import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
device = torch.device("cpu")

import json
result = {}
# result['fvd'] = calculate_fvd(videos1, videos2, device)
result['ssim'] = calculate_ssim(videos1, videos2)
result['psnr'] = calculate_psnr(videos1, videos2)
result['lpips'] = calculate_lpips(videos1, videos2, device)
print(json.dumps(result, indent=4))
```

It means we calculate:
    
- `FVD-frames[:10]`, `FVD-frames[:11]`, ..., `FVD-frames[:30]` 
- `avg-PSNR/SSIM/LPIPS-frame[0]`, `avg-PSNR/SSIM/LPIPS-frame[1]`, ..., `avg-PSNR/SSIM/LPIPS-frame[:10]`, and their std.

We cannot calculate `FVD-frames[:8]`, and it will pass when calculating, see ps.6.

The result shows: a all-zero matrix and a all-one matrix, their FVD-30 is 151.17. We also calculate their standard deviation. Other metrics are the same.

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

# Notice

1. You should `pip install lpips` first.
3. Make sure the pixel value of videos should be in [0, 1].
2. If you have something wrong with downloading FVD pre-trained model, you should manually download any of the following and put it into FVD folder. 
    - `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) 
    - `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)
4. For grayscale videos, we multiply to 3 channels [as it says](https://github.com/richzhang/PerceptualSimilarity/issues/23#issuecomment-492368812).
5. We average SSIM when images have 3 channels, ssim is the only metric extremely sensitive to gray being compared to b/w.
6. Since `frames_num` should > 10 when calculating FVD, so FVD calculation begins from 10-th frame, like upper example.