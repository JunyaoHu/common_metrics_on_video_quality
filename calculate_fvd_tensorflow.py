import numpy as np
import tensorflow as tf
from tqdm import tqdm

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = np.repeat(x, 3, axis=-3)

    # permute BTCHW -> BTHWC
    x = x.transpose(0, 1, 3, 4, 2)

    # https://github.com/google-research/google-research/tree/master/frechet_video_distance
    # value range [0, 1] -> [0, 255] int
    x = (x * 255).astype(np.uint8)

    return x

def calculate_fvd(videos1, videos2, only_final=False):

    from fvd.tensorflow.fvd import create_id3_embedding, preprocess
    from fvd.tensorflow.fvd import calculate_fvd as frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BTHWC
    # videos -> [batch_size, timestamps, h, w, channel]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[1] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, timestamps, h, w, channel]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = create_id3_embedding(preprocess(videos_clip1, (224, 224)))
        feats2 = create_id3_embedding(preprocess(videos_clip2, (224, 224)))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

        # calculate FVD
        result = frechet_distance(feats1, feats2)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            fvd_results.append(sess.run(result))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[1]+1)):
        
            # get a video clip
            # videos_clip [batch_size, timestamps[:clip], h, w, channel]
            videos_clip1 = videos1[:, : clip_timestamp]
            videos_clip2 = videos2[:, : clip_timestamp]

            # get FVD features
            feats1 = create_id3_embedding(preprocess(videos_clip1, (224, 224)))
            feats2 = create_id3_embedding(preprocess(videos_clip2, (224, 224)))
        
            # calculate FVD when timestamps[:clip]
            result = frechet_distance(feats1, feats2)
            
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.tables_initializer())
                fvd_results.append(sess.run(result))

    result = {
        "value": fvd_results,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = np.zeros((NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE)).astype(np.float32)
    videos2 = np.ones((NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE)).astype(np.float32)

    result = calculate_fvd(videos1, videos2, only_final=True)
    print("[fvd-tensorflow]", result["value"])

if __name__ == "__main__":
    main()
