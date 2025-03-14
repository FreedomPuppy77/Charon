# %%
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import interp2d
from tqdm import tqdm

label_root_dir = "/data/lyh/8th_data/8th_Annotations/VA_Estimation_Challenge/val"
video_dir = "/data/lyh/8th_data/raw_videos"
features_dir = "/data/lyh/8th_result/pre_data/npy_data"
crop_dir = "/data/lyh/8th_data/cropped_aligned/val"
sample_root_dir = "/data/lyh/8th_result/pre_data/npz_data/val"

mae_dir = os.path.join(features_dir, "val")
mods = os.listdir(label_root_dir)

for mod in mods:

    label_dir = os.path.join(label_root_dir, mod)
    label_files = os.listdir(label_dir)

    sample_dir = os.path.join(sample_root_dir, mod)
    os.makedirs(sample_dir, exist_ok=True)

    for label_file in tqdm(label_files):

        save_path = os.path.join(sample_dir, label_file.split(".")[0] + ".npz")

        if os.path.exists(save_path):
            print(f"输出文件已存在，跳过: {save_path}")
            continue
        video_path = os.path.join(
            video_dir,
            label_file.replace("_left", "")
            .replace("_right", "")
            .replace(".mp4", "")
            .replace(".avi", "")
            .replace(".txt", ".mp4"),

        )
        if not os.path.exists(video_path):

            video_path = os.path.join(
                video_dir,
                label_file.replace("_left", "")
                .replace("_right", "")
                .replace(".txt", ".avi"),
            )
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Warning: 无法打开视频文件 - {video_path}")
            continue
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        print("num_frames", num_frames)

        img_dir = None
        for root, dirs, files in os.walk(crop_dir):
            if os.path.basename(root) == label_file.split(".")[0]:
                img_dir = root
                break

        if img_dir is None:
            print(f"Warning: Directory for {label_file} not found in {crop_dir}")
            continue

        img_names = os.listdir(img_dir)

        if len(img_names) == 0:
            print(f"Warning: No images found in {img_dir}")
            continue
        img_names.sort()
        print(f"当前处理的文件夹: {img_dir}, 标签文件: {label_file}")
        begin = int(img_names[0].split(".")[0]) - 1
        try:
            end = int(img_names[-1].split(".")[0]) - 1
            end = min(end, num_frames)
        except:
            print(img_dir, img_names[-1])


        valences = []
        arousals = []
        with open(os.path.join(label_dir, label_file)) as f:

            lines = f.readlines()
            for line in lines[1:]:
                valence, arousal = line.strip().split(",")
                valence = float(valence)
                arousal = float(arousal)
                valences.append(valence)
                arousals.append(arousal)


        img_mae_feature_dir = os.path.join(mae_dir, label_file.split(".")[0])
        prev_mae_feature = None
        img_mae_features = []
        for i in tqdm(range(begin, end)):

            img_mae_feature_file = os.path.join(
                img_mae_feature_dir, str(i + 1).zfill(5) + ".npy"
            )
            if os.path.exists(img_mae_feature_file):
                img_mae_feature = np.load(img_mae_feature_file)
                prev_mae_feature = img_mae_feature
            else:
                img_mae_feature = prev_mae_feature
            img_mae_features.append(img_mae_feature)
        img_mae_features = np.stack(img_mae_features)
        print("img_mae_features", img_mae_features.shape)

        print("begin,end", begin, end)

        select_mae_features = img_mae_features
        print("select_mae_features", select_mae_features.shape)

        select_valences = np.array(valences[begin:end])
        select_arousals = np.array(arousals[begin:end])

        print("select_valences", len(select_valences))
        print("select_arousals", len(select_arousals))

        save_path = os.path.join(sample_dir, label_file.split(".")[0] + ".npz")

        np.savez(
            save_path,
            select_mae_features=select_mae_features,
            select_valences=select_valences,
            select_arousals=select_arousals,
        )

        print("-" * 20)
