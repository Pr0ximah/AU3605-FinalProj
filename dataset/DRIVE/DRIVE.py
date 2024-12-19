import sys

sys.path.append("./")
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from utils.pre_process import color_normalization


def load_gif(path):
    video_cap = cv2.VideoCapture(path)
    ret, frame = video_cap.read()
    video_cap.release()
    frame = frame[..., ::-1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


class DRIVE_Dataset(Dataset):
    def __init__(self):
        print(" ** Loading dataset...")
        self.data = []
        self.target = []

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
            for img_path in os.listdir(training_data_dir):
                tarin_img_path = os.path.join(training_data_dir, img_path)
                img = cv2.imread(tarin_img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (572, 572))
                img = color_normalization(img)
                img = torch.tensor(img, dtype=torch.float32)
                img = img.permute(2, 0, 1)
                self.data.append(img)
                target_img_path = os.path.join(
                    label_data_dir, img_path.split(".")[0] + suffix_mapping[k]
                )
                target = load_gif(target_img_path)
                target = cv2.resize(target, (388, 388))
                target = torch.tensor(target, dtype=torch.float32)
                target.unsqueeze_(0)
                self.target.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


if __name__ == "__main__":
    dataset = DRIVE_Dataset()
    print(len(dataset))
    img = dataset[5][0].permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imshow("img", img)
    img_2 = img.copy()
    img_2 = color_normalization(img_2)
    cv2.imshow("img_2", img_2)
    cv2.waitKey(0)
