import sys
sys.path.append('./')
import torch.optim as optim
from models.DISK_net import DiskMaculaNet
from torch import nn
from dataset.DISK.disk_OD import DISK_Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.pre_process as pre_process

# 加载模型结构
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DiskMaculaNet()
model.load_state_dict(torch.load('models/logs/model_OD_final.pth'))
model.to(device)  # 将模型移动到设备
model.eval()
# for i in range(10,55,1):
#     print("i = ",i)
#     img_path = f"dataset/DISK/shipanbiaozhu/c/Training c/IDRiD_{i}.jpg"
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#     # 图片归一化
#     y, x = img.shape[:2]
#     if x > y:
#         diff = (x - y) // 2
#         img = cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

#     img = cv2.resize(img, (256, 256))
#     img = pre_process.color_normalization(img)
#     img_show = img
#     img = torch.tensor(img, dtype=torch.float32)
#     img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
#     img = img.unsqueeze(0)  # 添加 batch 维度

#     with torch.no_grad():
#         output = model(img.to(device))

#     plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
#     plt.scatter(output[0][0].item(), output[0][1].item(), c='r')
#     plt.show()
#     print([output[0][0].item(), output[0][1].item()])



img_path = f"dataset/DISK/shipanbiaozhu/c/Training c/16_test.tif"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 图片归一化
y, x = img.shape[:2]
if x > y:
    diff = (x - y) // 2
    img = cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

img = cv2.resize(img, (256, 256))
img = pre_process.color_normalization(img)
img_show = img
img = torch.tensor(img, dtype=torch.float32)
img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
img = img.unsqueeze(0)  # 添加 batch 维度

with torch.no_grad():
    output = model(img.to(device))

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.scatter(output[0][0].item(), output[0][1].item(), c='r')
plt.show()
print([output[0][0].item(), output[0][1].item()])