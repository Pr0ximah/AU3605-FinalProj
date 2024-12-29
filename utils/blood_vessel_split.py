import torch
import cv2
import numpy as np
from models.unet import UNet


class BV_Split:
    def __init__(self, model_path="models/logs/unet_model_final.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps")
        self.model = UNet(3)
        self.model.load_state_dict(torch.load(model_path, weights_only=False, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

    def split(self, img_):
        img = cv2.resize(img_, (512, 512))
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
        output = output.to("cpu")
        output = output.squeeze(0).squeeze(0).numpy()
        output *= 255
        output = np.array(output).astype(np.uint8)

        return output
