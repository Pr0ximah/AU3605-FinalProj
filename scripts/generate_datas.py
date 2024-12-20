import sys

sys.path.append("./")
from dataset.DRIVE.DRIVE import DRIVE_Dataset
import numpy as np
import cv2
import os
import shutil
from tqdm import trange

if __name__ == "__main__":
    # saved directory
    saved_dir = "dataset/DRIVE/training/Generate"

    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)

    os.makedirs(f"{saved_dir}/images", exist_ok=True)
    os.makedirs(f"{saved_dir}/targets", exist_ok=True)

    dataset = DRIVE_Dataset()

    idx = 0
    
    for i in trange(len(dataset)):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        label = label.squeeze(0).numpy()
        print(label.shape)
        label = np.where(label >= 0.5, 255, 0)
        label = label.astype(np.uint8)

        # rotate
        for rotate in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            img2 = cv2.rotate(img, rotate)
            label2 = cv2.rotate(label, rotate)
            img_name = f"{saved_dir}/images/{idx}.png"
            label_name = f"{saved_dir}/targets/{idx}.png"
            cv2.imwrite(img_name, img2)
            cv2.imwrite(label_name, label2)
            idx += 1
        # flip
        for flip in [0, 1, -1]:
            img1 = cv2.flip(img, flip)
            label1 = cv2.flip(label, flip)
            img_name = f"{saved_dir}/images/{idx}.png"
            label_name = f"{saved_dir}/targets/{idx}.png"
            cv2.imwrite(img_name, img1)
            cv2.imwrite(label_name, label1)
            idx += 1
            # flip & rotate
            for rotate in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                img2 = cv2.rotate(img1, rotate)
                label2 = cv2.rotate(label1, rotate)
                img_name = f"{saved_dir}/images/{idx}.png"
                label_name = f"{saved_dir}/targets/{idx}.png"
                cv2.imwrite(img_name, img2)
                cv2.imwrite(label_name, label2)
                idx += 1
        