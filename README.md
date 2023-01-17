# common_metrics_on_video_quality

you can calculate Frechét Video Distance (FVD), the peak-signal-to-noise ratio (PSNR), the structural similarity index measure (SSIM) and the learned perceptual image patch similarity (LPIPS) easily.

As for FVD, this code is from [MVCD model project](https://github.com/voletiv/mcvd-pytorch) and other websites and projects, I've just extracted the part of it that's relevant to the calculation. This code can be used to evaluate FVD scores for generative or predictive models. 

ps: pixel value should be in [0, 1].

Example: 4 batches, 8 videos of a batch, 30 frames, 3 channels (=3 necessary), 64x64 size, calculate per 10 frames (>=10 necessary, FVD calculation needed) 

it means we calculate FVD-frames[:10], FVD-frames[:20], FVD-frames[:30], PSNR/SSIM/LPIPS-frame[10], PSNR/SSIM/LPIPS-frame[20], PSNR/SSIM/LPIPS-frame[30], and their std.

a all-zero matrix and a all-one matrix, their FVD10 is 570.07, FVD20 is 338.18 and FVD30 is 151.17, as shown in the picture below. we also calculate their standard deviation. other metrics are the same.

![image](https://user-images.githubusercontent.com/67564714/212931946-7b6924aa-88db-4ade-8787-4bcb18460fc4.png)

before run: 

1. you should download `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) or you can download `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI), you download random one and put it into fvd folder should be fine.
2. you should `pip install lpips`
3. pixel value should be in [0, 1].
4. if gray-scale you should muitiply to 3 channels, [as it says](https://github.com/richzhang/PerceptualSimilarity/issues/23#issuecomment-492368812)

If code cannot run correctly, please create an issue here~
