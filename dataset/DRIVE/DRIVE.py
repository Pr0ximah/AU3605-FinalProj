import torch
from torch.utils.data import Dataset
import os
import cv2


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
        imgs = os.listdir(r"dataset\DRIVE\training\images")
        for img in imgs:
            img_path = os.path.join(r"dataset\DRIVE\training\images", img)
            id = img.split("_")[0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (572, 572))
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2, 0, 1)
            self.data.append(img)
            target_path = os.path.join(
                r"dataset\DRIVE\training\1st_manual", id + "_manual1.gif"
            )
            target = load_gif(target_path)
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
    print(dataset[0][0].shape)
