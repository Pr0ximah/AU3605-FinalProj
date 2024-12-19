import torch.optim as optim
import sys
sys.path.append('./')
from models.p1net import DiskMaculaNet
from torch import nn
from dataset.DISK.disk import DISK_Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import utils.pre_process as pre_process

# 加载模型结构
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DiskMaculaNet()
model.load_state_dict(torch.load('models/logs/model_FCT_2.pth'))
model.to(device)  # 将模型移动到设备
model.eval()

img_path = "dataset/DISK/shipanbiaozhu/c/Training c/IDRiD_38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (256, 256))
img = pre_process.color_normalization(img)
img = torch.tensor(img, dtype=torch.float32)
img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
img = img.unsqueeze(0)  # 添加 batch 维度

with torch.no_grad():
    output = model(img.to(device))

plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.scatter(output[0][0].item(), output[0][1].item(), c='r')
plt.show()
