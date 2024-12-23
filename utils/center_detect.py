import sys

sys.path.append("./")
from models.DISK_net import DiskMaculaNet
import torch
import cv2
import numpy as np


def color_normalization(test_image):
    ref_image = cv2.imread("utils/ref_img.jpg")
    test_image = test_image.astype(np.float32)
    test_image = test_image.copy()
    ref_image = ref_image.astype(np.float32)

    test_mean, test_std = cv2.meanStdDev(test_image)
    ref_mean, ref_std = cv2.meanStdDev(ref_image)

    normalized_image = np.zeros_like(test_image)
    for i in range(3):
        normalized_image[:, :, i] = (
            (test_image[:, :, i] - test_mean[i]) / test_std[i]
        ) * ref_std[i] + ref_mean[i]

    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

    return normalized_image


class CenterDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_OD = DiskMaculaNet()
        self.model_OD.load_state_dict(torch.load("models/logs/model_OD_final.pth", weights_only=False))
        self.model_OD.to(self.device)  # 将模型移动到设备
        self.model_OD.eval()
        self.model_FCT = DiskMaculaNet()
        self.model_FCT.load_state_dict(torch.load("models/logs/model_FCT_final.pth", weights_only=False))
        self.model_FCT.to(self.device)  # 将模型移动到设备
        self.model_FCT.eval()

    def detect(self, img):
        # 图片归一化
        y, x = img.shape[:2]
        if x > y:
            diff = (x - y) // 2
            img = cv2.copyMakeBorder(
                img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        img = cv2.resize(img, (256, 256))
        img = color_normalization(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)  # 转换为 (C, H, W)
        img = img.unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            output_1 = self.model_OD(img.to(self.device))
            output_2 = self.model_FCT(img.to(self.device))

        output_1 *= 2
        output_2 *= 2

        return [
            [output_1[0][0].item(), output_1[0][1].item()],
            [output_2[0][0].item(), output_2[0][1].item()],
        ]


if __name__ == "__main__":
    center_detector = CenterDetector()
    print(center_detector.detect("dataset/DISK/shipanbiaozhu/c/Training c/07_test.tif"))
