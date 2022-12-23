# frechet_video_distance_calculation
calculate FVD (frechet video distance)

This code is from [MVCD model project](https://github.com/voletiv/mcvd-pytorch), I've just extracted the part of it that's relevant to the FVD calculation. This code can be used to evaluate FVD scores for new generative models. 

16 batches, 3 channels, 15 frames, 64x64 size

a all-zero matrix and a all-one matrix, their FVD is 134.24, as shown in the picture below.

![image](https://user-images.githubusercontent.com/67564714/209288914-30519bce-caea-4069-85f7-4fb82fb10324.png)

before run: you should download 'i3d_torchscript.pt' from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) or you can download 'i3d_pretrained_400.pt' from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI), you download random one and put it into fvd folder should be fine.

If code cannot run correctly, please create an issue here~
