# common_metrics_on_video_quality

you can calculate Frechét Video Distance (FVD), the peak-signal-to-noise ratio (PSNR), the structural similarity index measure (SSIM) and the learned perceptual image patch similarity (LPIPS) easily.

As for FVD, this code is from [MVCD model project](https://github.com/voletiv/mcvd-pytorch) and other websites and projects, I've just extracted the part of it that's relevant to the calculation. This code can be used to evaluate FVD scores for generative or predictive models. 

ps: pixel value should be in [0, 1], support grayscale and RGB.

Example: 8 videos of a batch, 30 frames, 3 channels, 64x64 size, calculate per 10 frames (>=10 necessary, FVD calculation needed) 

it means we calculate FVD-frames[:10], FVD-frames[:20], FVD-frames[:30], avg-PSNR/SSIM/LPIPS-frame[:10], avg-PSNR/SSIM/LPIPS-frame[:20], avg-PSNR/SSIM/LPIPS-frame[:30], and their std.

a all-zero matrix and a all-one matrix, their FVD10 is 570.07, FVD20 is 338.18 and FVD30 is 151.17. We also calculate their standard deviation. Other metrics are the same.

```
{
    "fvd": {
        "fvd": {
            "[:10]": 570.07320378183,
            "[:20]": 338.18221898178774,
            "[:30]": 151.16806952692093
        },
        "fvd_per_frame": 10,
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
            "avg[:10]": 9.999000099990664e-05,
            "avg[:20]": 9.999000099990664e-05,
            "avg[:30]": 9.999000099990664e-05
        },
        "ssim_std": {
            "std[:10]": 0.0,
            "std[:20]": 0.0,
            "std[:30]": 0.0
        },
        "ssim_per_frame": 10,
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
            "avg[:10]": 0.0,
            "avg[:20]": 0.0,
            "avg[:30]": 0.0
        },
        "psnr_std": {
            "std[:10]": 0.0,
            "std[:20]": 0.0,
            "std[:30]": 0.0
        },
        "psnr_per_frame": 10,
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
            "avg[:10]": 0.8140146732330322,
            "avg[:20]": 0.8140146732330322,
            "avg[:30]": 0.8140146732330322
        },
        "lpips_std": {
            "std[:10]": 0.0,
            "std[:20]": 0.0,
            "std[:30]": 0.0
        },
        "lpips_per_frame": 10,
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

before run: 

1. you should download `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) or you can download `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI), you download random one and put it into fvd folder should be fine.
2. you should `pip install lpips`
3. pixel value should be in [0, 1].
4. for gray-scale, we muitiply to 3 channels [as it says](https://github.com/richzhang/PerceptualSimilarity/issues/23#issuecomment-492368812)
5. we average SSIM when images have 3 channels, ssim is the only metric extremely sensitive to gray being compared to b/w.

If code cannot run correctly, please create an issue here~
