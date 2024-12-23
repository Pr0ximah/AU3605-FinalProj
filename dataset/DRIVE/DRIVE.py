import sys

sys.path.append("./")
import torch
from torch.utils.data import Dataset
import numpy as np


class DRIVE_Dataset(Dataset):
    def __init__(self):
        print(" ** Loading dataset...")
        self.data = torch.load("dataset/DRIVE/training/datas.pt", weights_only=True)
        self.target = torch.load("dataset/DRIVE/training/targets.pt", weights_only=True)
        print(f" ** Dataset loaded, {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


if __name__ == "__main__":
    dataset = DRIVE_Dataset()
    img = dataset[50][0].permute(1, 2, 0).numpy().astype(np.uint8)
    target = dataset[50][1].squeeze(0).numpy().astype(np.uint8)
    print(target.shape)
