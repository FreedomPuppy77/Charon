import os
import numpy as np
from tqdm import tqdm
import cv2
from scipy.interpolate import interp2d
import torch.nn.functional as F
import torch
import pandas as pd
import pickle


video_frames_dict = {}
with open(
    "/data/lyh/8th_result/pre_test/CVPR_8th_ABAW_VA_test_set_sample.txt",
    "r",
) as f:
    lines = f.readlines()
    for line in lines[1:]:
        t = line.split("/")[0]
        if t not in video_frames_dict:
            video_frames_dict[t] = 1
        else:
            video_frames_dict[t] += 1


test_label = "/data/lyh/8th_result/pre_test/Valence_Arousal_Estimation_Challenge_test_set_release.txt"

video_dir = "/data/lyh/8th_data/raw_videos"
features_dir = "/data/lyh/8th_result/pre_test"


sample_root_dir = "/data/lyh/8th_result/pre_test/npz_test"

mae_dir = os.path.join(features_dir, "npy_test")


os.makedirs(sample_root_dir, exist_ok=True)

video_files = []
with open(test_label, "r") as f:
    for line in f:
        video_files.append(line.strip())


for video_file in tqdm(video_files):

    len_label_frames = video_frames_dict[video_file]


    img_mae_feature_dir = os.path.join(mae_dir, video_file.split(".")[0])
    img_mae_features = np.zeros((len_label_frames, 1024), dtype=np.float32)
    for i in tqdm(range(len_label_frames)):

        img_mae_feature_file = os.path.join(
            img_mae_feature_dir, str(i + 1).zfill(5) + ".npy"
        )
        if os.path.exists(img_mae_feature_file):
            img_mae_feature = np.load(img_mae_feature_file)
            img_mae_features[i] = img_mae_feature
        else:
            print(f"{i} not exist")

    select_mae_features = img_mae_features
    print("select_mae_features: ", select_mae_features.shape)

    save_path = os.path.join(sample_root_dir, video_file.split(".")[0] + ".npz")

    np.savez(save_path, select_mae_features=select_mae_features)

    print("-" * 20)
