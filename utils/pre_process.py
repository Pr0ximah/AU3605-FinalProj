import sys

sys.path.append("./")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.center_detect import CenterDetector, color_normalization


class Normalizer:
    def __init__(self):
        self.ref_image = cv2.imread(
            "dataset/DISK/shipanbiaozhu/c/Training c/11_test.tif", cv2.IMREAD_COLOR
        )
        self.ref_disc_center, self.ref_fovea_center = [
            48.0453987121582,
            117.51459503173828,
        ], [119.68902587890625, 129.96429443359375]
        self.test_image = None
        self.test_disc_center, self.test_fovea_center = None, None

    def normalize_size(self, image, disc_center, fovea_center, target_size=(512, 512)):
        # 获取原始图像大小
        original_h, original_w = image.shape[:2]

        # 计算视盘和中央凹的新坐标
        scale_x = target_size[0] / original_w
        scale_y = target_size[1] / original_h
        new_disc_center = (int(disc_center[0] * scale_x), int(disc_center[1] * scale_y))
        new_fovea_center = (
            int(fovea_center[0] * scale_x),
            int(fovea_center[1] * scale_y),
        )

        # 调整图像大小
        normalized_image = cv2.resize(
            image, target_size, interpolation=cv2.INTER_LINEAR
        )

        return normalized_image, new_disc_center, new_fovea_center

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

    def align_image_with_rotation(self):
        # Step 1: 缩放测试图像的视盘-中央凹距离
        d_test = np.sqrt(
            (self.test_fovea_center[0] - self.test_disc_center[0]) ** 2
            + (self.test_fovea_center[1] - self.test_disc_center[1]) ** 2
        )
        d_ref = np.sqrt(
            (self.ref_fovea_center[0] - self.ref_disc_center[0]) ** 2
            + (self.ref_fovea_center[1] - self.ref_disc_center[1]) ** 2
        )
        scale = d_ref / d_test

        # 为保持图像大小不变，我们需要对内容进行缩放但保持画布大小
        transform_matrix = np.array(
            [
                [scale, 0, (1 - scale) * self.test_disc_center[0]],
                [0, scale, (1 - scale) * self.test_disc_center[1]],
            ]
        )
        scaled_image = cv2.warpAffine(
            self.test_image,
            transform_matrix,
            (self.test_image.shape[1], self.test_image.shape[0]),
        )

        # Step 2: 平移图像
        translation = (
            int(self.ref_disc_center[0] - self.test_disc_center[0]),
            int(self.ref_disc_center[1] - self.test_disc_center[1]),
        )

        translation_matrix = np.float32(
            [[1, 0, translation[0]], [0, 1, translation[1]]]
        )
        translated_image = cv2.warpAffine(
            scaled_image,
            translation_matrix,
            (scaled_image.shape[1], scaled_image.shape[0]),
        )

        # Step 3: 计算旋转角度
        dx_test = self.test_fovea_center[0] - self.test_disc_center[0]
        dy_test = self.test_fovea_center[1] - self.test_disc_center[1]
        test_angle = np.degrees(np.arctan2(dy_test, dx_test))

        dx_ref = self.ref_fovea_center[0] - self.ref_disc_center[0]
        dy_ref = self.ref_fovea_center[1] - self.ref_disc_center[1]
        ref_angle = np.degrees(np.arctan2(dy_ref, dx_ref))

        rotation_angle = ref_angle - test_angle

        # Step 4: 旋转图像
        center = (
            int(self.ref_disc_center[0]),
            int(self.ref_disc_center[1]),
        )  # 以基准视盘中心为旋转中心
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        aligned_image = cv2.warpAffine(
            self.test_image,
            rotation_matrix,
            (self.test_image.shape[1], self.test_image.shape[0]),
        )

        return aligned_image

    def get_center(self, img_path):
        center_detector = CenterDetector()
        return center_detector.detect(img_path)

    def alignment(self, image, disc_center, fovea_center):
        self.test_image = image
        self.test_disc_center, self.test_fovea_center = disc_center, fovea_center

        # 图像归一化
        self.test_image = color_normalization(self.test_image)

        # 调整图像大小
        self.test_image, disc_center_, fovea_center_ = self.normalize_size(
            self.test_image, self.test_disc_center, self.test_fovea_center
        )
        self.ref_image, ref_disc_center_, ref_fovea_center_ = self.normalize_size(
            self.ref_image, self.ref_disc_center, self.ref_fovea_center
        )

        # 调整图像方向
        flip = self.if_orientation()
        self.test_image, self.test_disc_center, self.test_fovea_center = (
            self.normalize_orientation(
                self.test_image, self.test_disc_center, self.test_fovea_center, flip
            )
        )

        # 对齐图像
        aligned_image = self.align_image_with_rotation()

        # 颜色归一化
        aligned_image = color_normalization(aligned_image)

        return aligned_image


if __name__ == "__main__":
    img_path = "dataset/DISK/shipanbiaozhu/c/Training c/15_test.tif"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    center_detector = CenterDetector()
    disc_center, fovea_center = center_detector.detect(image)

    normalizer = Normalizer()
    test_image = normalizer.alignment(image, disc_center, fovea_center)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)
    plt.show()
