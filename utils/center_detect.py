import torch.optim as optim
import sys
sys.path.append('./')
from models.DISK_net import DiskMaculaNet
from torch import nn
from dataset.DISK.disk_OD import DISK_Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import utils.pre_process as pre_process
class CenterDetector:
    def __init__(self):
        pass

    def detect(self, img_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_1 = DiskMaculaNet()
        model_1.load_state_dict(torch.load('models/logs/model_OD_final.pth'))
        model_1.to(device)  # 将模型移动到设备
        model_1.eval()
        model_2 = DiskMaculaNet()
        model_2.load_state_dict(torch.load('models/logs/model_FCT_final.pth'))
        model_2.to(device)  # 将模型移动到设备
        model_2.eval()


        # img_path = "dataset/DISK/shipanbiaozhu/c/Training c/IDRiD_05.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # 图片归一化
        y, x = img.shape[:2]
        if x > y:
            diff = (x - y) // 2
            img = cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(img, (256, 256))
        img_show = img
        img = pre_process.color_normalization(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
        img = img.unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            output_1 = model_1(img.to(device))
            output_2 = model_2(img.to(device))
        plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        plt.scatter(output_1[0][0].item(), output_1[0][1].item(), c='r')
        plt.scatter(output_2[0][0].item(), output_2[0][1].item(), c='b')
        plt.show()
                # some detection code
        return [[output_1[0][0].item(), output_1[0][1].item()], [output_2[0][0].item(), output_2[0][1].item()]]
    
if __name__ == '__main__':
    center_detector = CenterDetector()
    print(center_detector.detect("dataset/DISK/shipanbiaozhu/c/Training c/07_test.tif"))
