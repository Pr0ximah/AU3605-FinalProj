import sys

sys.path.append("./")
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import load
import argparse
from utils.center_detect import CenterDetector, color_normalization
from utils.blood_vessel_split import BV_Split
from utils.pre_process import Normalizer, fill


def parse_args():
    parser = argparse.ArgumentParser(description="PCA Detection")
    parser.add_argument("-p", "--image_path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取pca结果
    np.load("PCA/principal_components.npy")
    pca = load("PCA/pca.joblib")

    # 读取目标灰度图像
    image_path = args.image_path
    assert image_path, "Please specify the image path using -p or --image_path"
    image = cv2.imread(image_path)

    # 预处理
    img = fill(image)
    img = color_normalization(img)
    OD_center, FCT_center = CenterDetector().detect(img)
    aligned_img = Normalizer().alignment(img, OD_center, FCT_center)
    mask = BV_Split().split(aligned_img)
    image = cv2.inpaint(aligned_img, mask, 5, cv2.INPAINT_TELEA)

    green_channel = image[:, :, 1]
    image_original = green_channel.copy()
    height, width = green_channel.shape
    img_data = [green_channel.reshape(-1)]

    # 进行PCA处理
    transformed = pca.transform(img_data)
    # x = np.linalg.lstsq(np.array(principal_components).T, np.array(transformed).T, rcond=None)[0]
    # transformed = (np.array(principal_components).T @ x).T
    reconstructed = pca.inverse_transform(transformed)

    # 重塑为二维图像
    reconstructed_2d = reconstructed.reshape(height, width)

    # 计算原始图像与重建图像的差值图像
    diff_image = np.abs(
        image_original.astype(np.float32) - reconstructed_2d.astype(np.float32)
    )
    binary_image = np.where(diff_image > 25, 255, 0).astype(np.uint8)

    # 显示原始图片、重建图片以及差值图片
    plt.subplots(1, 3, figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_2d, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap="gray")
    plt.title("Difference Image")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
