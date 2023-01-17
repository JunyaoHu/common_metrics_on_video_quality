# frechet_video_distance_calculation
calculate FVD (frechet video distance)

This code is from [MVCD model project](https://github.com/voletiv/mcvd-pytorch), I've just extracted the part of it that's relevant to the FVD calculation. This code can be used to evaluate FVD scores for generative models. 

4 batches, 8 videos of a batch, 30 frames, 3 channels (=3 necessary), 64x64 size, calculate per 10 frames (>=10 necessary) 

it means we calculate FVD-frames[:15] and FVD-frames[:30]

pixel value should be 0-1.

a all-zero matrix and a all-one matrix, their FVD15 is 133.89, and FVD30 is 151.17, as shown in the picture below. we also calculate their standard deviation.

![image](https://user-images.githubusercontent.com/67564714/212869110-d6fcacc9-3c77-4749-b417-8e587ef9a985.png)

before run: you should download `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) or you can download `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI), you download random one and put it into fvd folder should be fine.

If code cannot run correctly, please create an issue here~
