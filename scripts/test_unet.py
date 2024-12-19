import sys

sys.path.append("./")
from utils.blood_vessel_split import BV_Split
import matplotlib.pyplot as plt
import cv2
from utils.pre_process import color_normalization


def load_gif(path):
    video_cap = cv2.VideoCapture(path)
    ret, frame = video_cap.read()
    video_cap.release()
    frame = frame[..., ::-1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


if __name__ == "__main__":
    model_dir = "models/logs/oold/unet_model_700.pth"
    # use_training_data = True
    use_training_data = False

    if use_training_data:
        img_idx = 25
        test_img_dir = f"dataset/DRIVE/training/images/{img_idx}_training.tif"
        test_img = cv2.imread(test_img_dir, cv2.IMREAD_COLOR)
        test_img = color_normalization(test_img)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        label_img_dir = f"dataset/DRIVE/training/1st_manual/{img_idx}_manual1.gif"
        label_img = load_gif(label_img_dir)
        assert test_img is not None
        bv_split = BV_Split(model_dir)
        output = bv_split.split(test_img_dir)
        plt.subplots(1, 3, figsize=(18, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(test_img)
        plt.subplot(1, 3, 2)
        plt.imshow(label_img, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(output, cmap="gray")
        plt.show()
    else:
        test_img_dir = "dataset/DRIVE/test/images/05_test.tif"
        test_img = cv2.imread(test_img_dir, cv2.IMREAD_COLOR)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        assert test_img is not None
        bv_split = BV_Split(model_dir)
        output = bv_split.split(test_img_dir)
        plt.subplots(1, 2, figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.subplot(1, 2, 2)
        plt.imshow(output, cmap="gray")
        plt.show()
