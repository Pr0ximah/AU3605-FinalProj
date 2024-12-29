import sys

sys.path.append("./")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.center_detect import CenterDetector, color_normalization


def fill(image):
    x, y = image.shape[:2]
    diff = 0
    if x > y:
        diff = (x - y) // 2
        if diff % 2 == 0:
            image = cv2.copyMakeBorder(
                image, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            image = cv2.copyMakeBorder(
                image, 0, 0, diff, diff + 1, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
    else:
        diff = (y - x) // 2
        if diff % 2 == 0:
            image = cv2.copyMakeBorder(
                image, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            image = cv2.copyMakeBorder(
                image, diff, diff + 1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


class Normalizer:
    def __init__(self):
        self.ref_image = cv2.imread(
            "dataset/DISK/shipanbiaozhu/c/Training c/11_test.tif", cv2.IMREAD_COLOR
        )
        self.ref_image = fill(self.ref_image)
        self.ref_disc_center, self.ref_fovea_center = [
            99.36289978027344,
            236.4633331298828,
        ], [235.9381561279297, 260.9501953125]
        self.ref_image = cv2.resize(
            self.ref_image, (512, 512), interpolation=cv2.INTER_LINEAR
        )
        self.test_image = None
        self.test_disc_center, self.test_fovea_center = None, None

    def if_orientation(self):
        # 判断是否为右眼（中央凹的 x 坐标是否大于视盘中心的 x 坐标）
        if (
            self.ref_fovea_center[0] > self.ref_disc_center[0]
            and self.test_fovea_center[0] > self.test_disc_center[0]
        ):
            flip = False
        elif (
            self.ref_fovea_center[0] < self.ref_disc_center[0]
            and self.test_fovea_center[0] < self.test_disc_center[0]
        ):
            flip = False
        else:
            flip = True
        return flip

    def normalize_orientation(self, image, disc_center, fovea_center, flag):
        # 判断是否为右眼（中央凹的 x 坐标是否大于视盘中心的 x 坐标）
        if flag:
            # 水平镜像翻转
            normalized_image = cv2.flip(image, 1)

            # 更新坐标
            img_width = image.shape[1]
            new_disc_center = (img_width - disc_center[0], disc_center[1])
            new_fovea_center = (img_width - fovea_center[0], fovea_center[1])
        else:
            normalized_image = image
            new_disc_center = disc_center
            new_fovea_center = fovea_center

        return normalized_image, new_disc_center, new_fovea_center

    def detect_circle(self, image):
        """
        使用 Hough 圆变换检测图像中的圆并返回圆心和半径。
        """
        # 转为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

        # 使用 HoughCircles 检测圆
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=gray_image.shape[0] // 4,  # 限制圆之间的最小距离
            param1=100,  # 边缘检测高阈值
            param2=50,  # 累加器阈值
            minRadius=100,  # 最小半径
            maxRadius=300,  # 最大半径
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))  # 转换为整数坐标
            return [
                (circle[0], circle[1], circle[2]) for circle in circles[0, :]
            ]  # 返回 (x, y, r)
        else:
            return []

    def move_circle_to_target(self, image, source_center, target_center):
        # 计算平移向量
        translation = (
            (int(target_center[0]) - int(source_center[0])),
            (int(target_center[1]) - int(source_center[1])),
        )

        # 创建平移矩阵
        translation_matrix = np.float32(
            [[1, 0, translation[0]], [0, 1, translation[1]]]
        )

        # 应用平移变换
        moved_image = cv2.warpAffine(
            image, translation_matrix, (image.shape[1], image.shape[0])
        )

        return moved_image

    def calculate_angle(self, center, disc_center):
        dx = disc_center[0] - center[0]
        dy = disc_center[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx))  # atan2 返回的是弧度值，转换为度
        return angle

    def rotate_image(self, image, center, angle):
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转矩阵
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image

    def alignment(self, image, disc_center, fovea_center):
        self.test_image = image
        self.test_disc_center, self.test_fovea_center = disc_center, fovea_center

        # 图像归一化
        self.test_image = color_normalization(self.test_image)

        # 调整图像方向
        flip = self.if_orientation()
        self.test_image, self.test_disc_center, self.test_fovea_center = (
            self.normalize_orientation(
                self.test_image, self.test_disc_center, self.test_fovea_center, flip
            )
        )

        # 检测圆心
        ref_circles = self.detect_circle(self.ref_image)[0]
        ref_x, ref_y, ref_r = ref_circles
        ref_center = (ref_x, ref_y)

        test_circles = self.detect_circle(self.test_image)[0]
        test_x, test_y, test_r = test_circles
        test_center = (test_x, test_y)

        # 圆心对齐
        aligned_image = self.move_circle_to_target(
            self.test_image, test_center, ref_center
        )

        # 旋转
        angle_ref = self.calculate_angle(ref_center, self.ref_disc_center)
        angle_test = self.calculate_angle(test_center, self.test_disc_center)
        angle = -angle_ref + angle_test
        aligned_image = self.rotate_image(aligned_image, ref_center, angle)

        # 颜色归一化
        aligned_image = color_normalization(aligned_image)

        return aligned_image


if __name__ == "__main__":
    img_path = "input/33.tif"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x, y = image.shape[:2]
    diff = 0
    if x > y:
        diff = (x - y) // 2
        image = cv2.copyMakeBorder(
            image, 0, 0, diff, diff + 1, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    print("test_size: ", image.shape)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()
    center_detector = CenterDetector()
    disc_center, fovea_center = center_detector.detect(image)
    print(f"Test Disc Center: {disc_center}")
    print(f"Test Fovea Center: {fovea_center}")

    normalizer = Normalizer()
    test_image = normalizer.alignment(image, disc_center, fovea_center)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)
    plt.show()
