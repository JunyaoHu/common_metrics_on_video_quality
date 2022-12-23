from models.fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import numpy as np
import scipy.stats as st
import torch

def to_i3d(x, c, h, w):
    x = x.reshape(x.shape[0], -1, c, h, w)
    if c == 1:
        x = x.repeat(1, 1, 3, 1, 1) # hack for greyscale images
    x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
    return x

def fvd_stuff(fake_embeddings, real_embeddings, skip_frame=5):
        avg_fvd = frechet_distance(fake_embeddings, real_embeddings)
        if skip_frame > 1:
            fvds_list = []
            # Calc FVD for [skip_frame] random trajs (each), and average that FVD
            trajs = np.random.choice(np.arange(skip_frame), (skip_frame,), replace=False)
            for traj in trajs:
                fvds_list.append(frechet_distance(fake_embeddings[traj::skip_frame], real_embeddings))
            fvd_traj_mean, fvd_traj_std  = float(np.mean(fvds_list)), float(np.std(fvds_list))
            fvd_traj_conf95 = fvd_traj_mean - float(st.norm.interval(confidence=0.95, loc=fvd_traj_mean, scale=st.sem(fvds_list))[0])
        else:
            fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = -1, -1, -1
        return avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95

NUMBER_OF_VIDEOS = 16
CHANNEL = 3
VIDEO_LENGTH = 15
SIZE = 64

def main():
    first_set_of_videos = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    second_set_of_videos = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    i3d = load_i3d_pretrained(device=device)

    first_set_of_videos = to_i3d(first_set_of_videos, CHANNEL, SIZE, SIZE)
    second_set_of_videos = to_i3d(second_set_of_videos, CHANNEL, SIZE, SIZE)
    y1 = get_fvd_feats(first_set_of_videos, i3d=i3d, device=device)
    y2 = get_fvd_feats(second_set_of_videos, i3d=i3d, device=device)

    fvd_result = frechet_distance(y1, y2)
    print("FVD:", fvd_result)

    print(fvd_stuff(y1,y2,skip_frame=5))

if __name__ == "__main__":
    main()