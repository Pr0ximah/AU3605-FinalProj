import torch
import cv2
from models.unet import UNet
import numpy as np


class BV_Split:
    def __init__(self, model_path="models/logs/unet_model_final.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(3)
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()
        self.model = self.model.to(self.device)

    def split(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512))
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
        output = output.to("cpu")
        output = output.squeeze(0).squeeze(0).numpy()

        # method 1
        # output = np.where(output >= 0.1, 255, 0)

        # # method 2
        # output = np.where(output >= 0.5, 255, 0)

        return output
