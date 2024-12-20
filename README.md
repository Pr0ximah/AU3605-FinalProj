# AU3605-FinalProj

## DRIVE数据集数据增强步骤

1. 运行`scripts/generate_dataset.py`，生成`dataset/DRIVE/training/datas.pt`和`dataset/DRIVE/training/targets.pt`文件，命令行提示`Total 85 samples`
2. 运行`scripts/generate_datas.py`，等待运行结束，生成`dataset/DRIVE/training/Generate`下的若干文件
3. 运行`scripts/generate_dataset.py`，刷新数据集，等待`dataset/DRIVE/training/datas.pt`和`dataset/DRIVE/training/targets.pt`文件生成即可，命令行提示`Total 1360 samples`