import os
import cv2
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_images(folder_path):
    """
    读取指定文件夹下的所有绿色通道图像信息，将数据存储到一个数组中。
    folder_path: 图像存储文件夹。
    return: 绿色通道图像信息二维数组。
    """
    img_data = []
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            image = cv2.imread(file_path)
            if image is None:
                print(f"无法读取图像文件: {file_path}，跳过该文件。")
                continue
            green_channel = image[:, :, 1]
            img_data.append(green_channel.reshape(-1))
    return np.array(img_data)


def main():
    img_folder_path = "output"
    img_data = read_images(img_folder_path)

    # 创建PCA对象
    pca = PCA(n_components=0.85)  # 用于实际PCA
    # pca = PCA(n_components=20) #用于数据展示
    # 对数据进行PCA
    pca.fit(img_data)
    principal_components = pca.transform(img_data)
    n_components = pca.n_components_

    # 处理主成分数据进行可视化
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_explained_variance_ratios = np.cumsum(explained_variance_ratios)
    sorted_indices = np.argsort(explained_variance_ratios)[::-1]
    sorted_explained_variance_ratios = explained_variance_ratios[sorted_indices]
    sorted_cumulative_explained_variance_ratios = cumulative_explained_variance_ratios[
        sorted_indices
    ]
    index = np.arange(len(sorted_explained_variance_ratios))

    # 可视化贡献度和累计贡献度
    bar_width = 0.9
    plt.bar(
        index + 1 + bar_width / 2,
        sorted_explained_variance_ratios,
        width=bar_width,
        label="Contribution Ratio",
        color="#2980b9",
    )
    plt.step(
        index + 1,
        sorted_cumulative_explained_variance_ratios,
        where="post",
        label="Cumulative Contribution Ratio",
        color="#3498db",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Ratio")
    plt.title("Contribution Degree of Principal Components")
    plt.title("Contribution Degree of Principal Components")
    plt.axis(
        [
            0,
            len(index),
            0,
            max(
                max(sorted_explained_variance_ratios),
                max(sorted_cumulative_explained_variance_ratios),
            )
            + 0.1,
        ]
    )
    plt.axhline(y=0.85, linestyle="--", color="#bdc3c7")
    plt.grid(visible="major", axis="both", color="black", alpha=0.1)

    # 找到0.85这条线和累计贡献度曲线交点的横坐标
    closest_difference = float("inf")
    intersection_x = None
    for i, y_value in enumerate(sorted_cumulative_explained_variance_ratios):
        if y_value > 0.85:
            intersection_x = index[i]
            break

    # 在图中标示交点横坐标
    plt.plot(
        [intersection_x + 1, intersection_x + 1],
        [0, 0.85],
        linestyle="--",
        color="#bdc3c7",
    )
    plt.text(
        intersection_x + 1,
        0.85,
        str(intersection_x + 1),
        verticalalignment="bottom",
        color="#bdc3c7",
        fontweight="bold",
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.show()

    print(principal_components)
    print(f"提取的特征数目: {n_components}")

    # 可视化各主成分分量
    fig, axes = plt.subplots(4, 5, figsize=(10, 6))
    axes = axes.flatten()
    for i in range(n_components):
        component_image = pca.components_[i].reshape((512, 512))
        row_index = i // 5
        col_index = i % 5
        ax = axes[i]
        ax.imshow(component_image, cmap="gray")
        ax.set_title(f"Principal Component{i + 1}")
        ax.axis("off")
    for j in range(i + 1, 4 * 5):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

    # 保存PCA结果
    np.save("principal_components.npy", principal_components)
    dump(pca, "pca.joblib")


if __name__ == "__main__":
    main()
