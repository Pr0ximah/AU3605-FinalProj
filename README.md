# SJTU AU3605大作业 - 基于背景重构的眼底视网膜图像的异常区域检测方法复现

**上海交通大学 数字图像处理与模式分析 课程大作业**

![流程图](<./readme/readmeimg1.png>)

通过归一化处理和基于机器学习的检测分割方法，此方法实现了不同光照、不同尺寸的眼底视网膜图像的血管分割填充和OD/FCT位置对齐，并通过PCA分析重构图像，实现视网膜病灶区域的自动检测与分离。

## 使用

*测试环境:*

- `python=3.10`
- `Windows/MacOS`

*以下所有操作均在仓库根目录`AU3605-FinalProj`下进行*

### 安装依赖

`pip install -r requirements.txt`

### 模型训练

1. 生成数据集打包文件

    `python scripts/generate_dataset.py`

2. 训练中心检测模型

    `python train_FCT.py`
    
    `python train_OD.py`

3. 训练血管分割模型

    `python train_unet.py`

### 图像预处理

1. 准备图像

    将要处理的图像放置于 `input/` 下

2. 开始处理

    运行 `python main.py`，在 `output/` 路径下获得处理结果
    

### PCA（主成分分析）

*假设**所有正常图像已完成预处理**，放置于 `output/` 路径下*

1. 使用正常图像进行主分量提取

    `python PCA/pca_learning.py`

2. 使用潜在病灶图像进行重构和病灶检测

    `python PCA/pca_detect.py -p <待检测图像路径>`

## 备注

- 颜色归一化参考图像路径为 `utils/ref_img.jpg`，可根据需要修改
- `dataset` 中 `DISK` 为OD/FCT标注，`DRIVE` 为血管分割标注
