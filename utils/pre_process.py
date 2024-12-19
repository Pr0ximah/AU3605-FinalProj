import cv2
import numpy as np


def color_normalization(test_image):
    # 将图像转换为浮点型
    ref_image=cv2.imread('utils/ref_img.jpg')
    test_image = test_image.astype(np.float32)
    ref_image = ref_image.astype(np.float32)

    # 分别计算参考图像和测试图像的均值和标准差
    test_mean, test_std = cv2.meanStdDev(test_image)
    ref_mean, ref_std = cv2.meanStdDev(ref_image)

    # 对每个通道进行归一化处理
    normalized_image = np.zeros_like(test_image)
    for i in range(3):  # 遍历 B, G, R 三个通道
        normalized_image[:, :, i] = (
            (test_image[:, :, i] - test_mean[i]) / test_std[i]) * ref_std[i] + ref_mean[i]

    # 将结果截断到 [0, 255] 范围，并转换为 uint8 类型
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

    return normalized_image
