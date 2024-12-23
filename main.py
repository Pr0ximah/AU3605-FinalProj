import os
import cv2
from utils.center_detect import color_normalization
from utils.blood_vessel_split import BV_Split
from utils.center_detect import CenterDetector
from utils.pre_process import Normalizer


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

    for img_filename in os.listdir(input_img_dir):
        img_path = os.path.join(input_img_dir, img_filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print(f" ** Processing image: {img_path}")

        # Color normalization
        img = color_normalization(img)

        # Detect center
        OD_center, FCT_center = center_detector.detect(img)

        # Transform image
        aligned_img = normalizer.alignment(img, OD_center, FCT_center)

        # Split blood vessels
        mask = bv_split.split(aligned_img)

        # Fill blood vessels
        output_img = cv2.inpaint(aligned_img, mask, 5, cv2.INPAINT_TELEA)

        output_path = os.path.join("output", img_filename.split(".")[0] + "_output.png")
        cv2.imwrite(output_path, output_img)
        # cv2.imwrite(os.path.join("output", img_filename.split(".")[0] + "_aligned.png"), aligned_img)
        # cv2.imwrite(os.path.join("output", img_filename.split(".")[0] + "_mask.png"), mask)

        print(f" ** Output saved to: {output_path}")


if __name__ == "__main__":
    main()
