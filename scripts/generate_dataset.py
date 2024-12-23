import sys

sys.path.append("./")
import os
import torch
import cv2
import numpy as np
from utils.pre_process import color_normalization


def load_gif(path):
    video_cap = cv2.VideoCapture(path)
    ret, frame = video_cap.read()
    video_cap.release()
    frame = frame[..., ::-1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


if __name__ == "__main__":
    datas = []
    targets = []

    if os.path.exists("dataset/DRIVE/training/datas.pt"):
        os.remove("dataset/DRIVE/training/datas.pt")
    if os.path.exists("dataset/DRIVE/training/targets.pt"):
        os.remove("dataset/DRIVE/training/targets.pt")

    # 3 datasets
    imgs = {}
    imgs["DRIVE"] = "dataset/DRIVE/training/DRIVE"
    imgs["AdamHoover"] = "dataset/DRIVE/training/AdamHoover"
    imgs["HRF"] = "dataset/DRIVE/training/HRF"
    suffix_mapping = {
        "DRIVE": ".gif",
        "AdamHoover": ".ah.ppm",
        "HRF": ".tif",
    }

    for k, v in imgs.items():
        training_data_dir = os.path.join(v, "images")
        label_data_dir = os.path.join(v, "targets")
        if not os.path.exists(training_data_dir):
            continue
        for img_path in os.listdir(training_data_dir):
            tarin_img_path = os.path.join(training_data_dir, img_path)
            img = cv2.imread(tarin_img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (512, 512))
            img = color_normalization(img)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2, 0, 1)
            datas.append(img)
            target_img_path = os.path.join(
                label_data_dir, img_path.split(".")[0] + suffix_mapping[k]
            )
            if suffix_mapping[k] == ".gif":
                target = load_gif(target_img_path)
            else:
                target = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
            target = cv2.resize(target, (512, 512))
            target = np.where(target >= 128, 1, 0)
            target = torch.tensor(target, dtype=torch.float32)
            target.unsqueeze_(0)
            targets.append(target)
            print(f"{tarin_img_path} loaded.")
        
    print(f" ** Total {len(datas)} samples.")
    torch.save(datas, "dataset/DRIVE/training/datas.pt")
    torch.save(targets, "dataset/DRIVE/training/targets.pt")
    print(f" ** Dataset saved.")
