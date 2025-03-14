import os
import sys
import random
import numpy as np
import torch
import yaml
from munch import DefaultMunch
from tqdm import tqdm
import pickle
from model import Model
from utils.loss import *


if __name__ == "__main__":
    device = torch.device("cuda")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config_path = "config/config.yml"
    yaml_dict = yaml.load(
        open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    cfg = DefaultMunch.fromDict(yaml_dict)

    model = Model(cfg)

    pretrained_path = sys.argv[1]

    save_path = sys.argv[2]

    print("loading from:", pretrained_path)
    pretrain_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace("module.", ""): v for k, v in pretrain_dict.items()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    window = 500
    stride = 500
    video_frames_dict = {}

    # 获取每个视频帧数
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
    print("videos:", len(video_frames_dict))

    # 获取视频列表
    test_label = "/data/lyh/8th_result/pre_test/Valence_Arousal_Estimation_Challenge_test_set_release.txt"
    test_videos = []
    with open(test_label, "r") as f:
        for line in f:
            test_videos.append(line.strip())

    src_dir = "/data/lyh/8th_result/pre_test/npz_test"

    with open(save_path, "w") as f:
        f.write("image_location,valence,arousal\n")
        for test_video in tqdm(test_videos):
            pred_a = []
            pred_v = []
            data = np.load(os.path.join(src_dir, test_video + ".npz"))

            select_mae_features = data["select_mae_features"].astype(np.float32)

            length = len(select_mae_features)

            for i in range(length // stride + 1):
                begin = stride * i
                if begin >= length:
                    break
                end = min(begin + window, length)

                mae_features = select_mae_features[begin:end]

                x = torch.tensor(mae_features).to(device).unsqueeze(0)

                with torch.no_grad():
                    output = model(x)
                    v = output['valence']
                    a = output['arousal']

                v = v.squeeze()
                a = a.squeeze()

                a = a.detach().cpu().numpy()
                v = v.detach().cpu().numpy()

                pred_a = pred_a + a.tolist()
                pred_v = pred_v + v.tolist()

            pred_a = np.array(pred_a)
            pred_a = np.clip(pred_a, -1, 1).tolist()

            pred_v = np.array(pred_v)
            pred_v = np.clip(pred_v, -1, 1).tolist()

            frames_length = video_frames_dict[test_video]

            common_length = min(length, frames_length)

            # 填充共同部分
            for i in range(frames_length):
                f.write(
                    test_video
                    + "/"
                    + str(i + 1).zfill(5)
                    + ".jpg,"
                    + str(pred_v[i])
                    + ","
                    + str(pred_a[i])
                    + "\n"
                )
