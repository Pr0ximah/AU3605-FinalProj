import sys
sys.path.append('./')

import utils.pre_process as pre_process
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
import torch

class DISK_Dataset(Dataset):
    def __init__(self, images_dir, target_csv_path, img_size=(256, 256)):
        """
        初始化数据集。
        :param images_dir: 本地图片目录路径。
        :param target_csv_path: 包含目标坐标的CSV文件路径。
        :param img_size: 调整图片大小，默认为 (256, 256)。
        """
        print(" ** Loading dataset...")
        self.data = []
        self.target = []
        self.img_size = img_size

        # 读取目标坐标的 CSV 文件
        targets_df = pd.read_csv(target_csv_path)
        data_cleaned = targets_df.iloc[:, :3]  # 选择前三列
        data_cleaned.columns = ['Identifier', 'X', 'Y']  # 重命名列

        # 清理数据，去掉含有 NaN 的行
        data_cleaned = data_cleaned.dropna(subset=['Identifier', 'X', 'Y'])

        # 转换为字典
        targets_dict = data_cleaned.set_index('Identifier').apply(
            lambda row: (row['X'], row['Y']), axis=1).to_dict()
        # targets_dict = {row['filename']: (row['x_disk'], row['y_disk'], row['x_macula'], row['y_macula'])
        #                 for _, row in targets_df.iterrows()}

        # 读取图片并关联坐标
        # imgs = os.listdir(images_dir)
        for img_filename in targets_dict.keys():
            img_path = os.path.join(images_dir, img_filename)
            img_path = img_path+'.jpg'
            print(img_path)
            if img_filename in targets_dict:
                # 读取并调整图片
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_size_ori = img.shape
                print(img_size_ori)
                y, x = img.shape[:2]
                diff = 0
                if x > y:
                    diff = (x - y) // 2
                    img = cv2.copyMakeBorder(
                        img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                img = cv2.resize(img, self.img_size)
                img_show = img
                img = pre_process.color_normalization(img)
                img = torch.tensor(img, dtype=torch.float32)
                img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
                self.data.append(img)

                # 添加对应的目标坐标
                target_temp = torch.tensor((targets_dict[img_filename][0]*256/img_size_ori[1], (targets_dict[img_filename][1] + diff)*256/(img_size_ori[0] + 2 * diff)), dtype=torch.float32)
                self.target.append(target_temp)
                # print(img_size_ori[1])
                # print(diff)
                # print(img_size_ori[0])
                # plt.imshow(img_show)
                # plt.scatter(target_temp[0].item(), target_temp[1].item(), c='r')
                # plt.show()

            else:
                print(f"Warning: No target found for image {img_filename}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


if __name__ == "__main__":
    # 示例：本地目录和CSV文件路径
    images_dir = r"dataset/DISK/shipanbiaozhu/ab/ab2"
    target_csv_path = r"dataset/DISK/shipanbiaozhu/FCT.csv"

    # 创建数据集实例
    dataset = DISK_Dataset(images_dir, target_csv_path)
    print(len(dataset.data))
    for i in range(-10,0):
        img = dataset.data[i].permute(1, 2, 0).numpy().astype(np.uint8)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.scatter(dataset.target[i][0].item(),
                    dataset.target[i][1].item(), c='r')
        plt.show()
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Sample image shape: {dataset[0][0].shape}")
    # print(f"Sample target: {dataset[0][1]}")
