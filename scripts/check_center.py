import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    img = cv2.imread("dataset/DISK/shipanbiaozhu/ab/ab2/IDRiD_001.jpg")
    coord = (2858, 1805)
    plt.imshow(img)
    plt.scatter(coord[0], coord[1], c='r')
    plt.show()