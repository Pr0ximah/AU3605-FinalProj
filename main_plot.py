import os
import cv2
import tqdm
from utils.center_detect import color_normalization
from utils.blood_vessel_split import BV_Split
from utils.center_detect import CenterDetector
from utils.pre_process import Normalizer, fill
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    # Image directory settings
    input_img_dir = "input"
    output_img_dir = "output"

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    # Initialize classes
    center_detector = CenterDetector()
    bv_split = BV_Split()
    normalizer = Normalizer()
    
    plt.subplots(2, 4, figsize=(15, 20))
    plt.rcParams['font.sans-serif']=['STHeiti'] #用来正常显示中文标签

    for img_filename in tqdm.tqdm(os.listdir(input_img_dir)):
        # print(f" ** Processing image: {img_path}")
        img_path = os.path.join(input_img_dir, img_filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        
        img = fill(img)
        
        plt.subplot(2, 4, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("填充黑边+尺寸统一")

        # Color normalization
        img = color_normalization(img)
        
        plt.subplot(2, 4, 3)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("颜色归一化")

        # Detect center
        OD_center, FCT_center = center_detector.detect(img)
        
        plt.subplot(2, 4, 4)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.scatter(OD_center[0], OD_center[1], c='r', s=10)
        plt.scatter(FCT_center[0], FCT_center[1], c='b', s=10)
        plt.title("OD/FCT检测")

        # Transform image
        aligned_img = normalizer.alignment(img, OD_center, FCT_center)
        
        plt.subplot(2, 4, 5)
        plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        plt.title("空域对齐")

        # Split blood vessels
        mask = bv_split.split(aligned_img)
        
        plt.subplot(2, 4, 6)
        plt.imshow(mask, cmap='gray')
        plt.title("血管分割")

        # Fill blood vessels
        output_img = cv2.inpaint(aligned_img, mask, 5, cv2.INPAINT_TELEA)
        
        plt.subplot(2, 4, 7)
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title("血管填充")

        # output_path = os.path.join("output", img_filename.split(".")[0] + "_output.png")
        # cv2.imwrite(output_path, output_img)
        # cv2.imwrite(os.path.join("output", img_filename.split(".")[0] + "_aligned.png"), aligned_img)
        # cv2.imwrite(os.path.join("output", img_filename.split(".")[0] + "_mask.png"), mask)

        # print(f" ** Output saved to: {output_path}")


        plt.subplot(2, 4, 8)
        plt.axis('off')
        plt.show()
        break


if __name__ == "__main__":
    main()
