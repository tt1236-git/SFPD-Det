# Spatial Focusing and Progressive Decoupling Detector for High-Aspect-Ratio Rotated Objects

项目基于MMRotate框架，实现了针对DIOR-R数据集的旋转目标检测。DIOR-R数据集包含20个类别的旋转目标，是遥感图像旋转目标检测的重要基准数据集。



## 环境安装

依赖于[PyTorch](https://pytorch.org/)、[MMCV](https://github.com/open-mmlab/mmcv)和[mmrotate](https://github.com/open-mmlab/mmrotate)。
以下是快速安装步骤：

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/tt1236-git/SFPD-Det.git
cd diorr-detection
pip install -r requirements/build.txt
pip install -v -e .
```

## 数据准备

### DIOR-R数据集

DIOR-R数据集是一个用于旋转目标检测的遥感图像数据集，包含20个类别：

```
'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
'chimney', 'expressway-service-area', 'expressway-toll-station', 
'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 
'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
'vehicle', 'windmill'
```

### 数据集下载

您可以从以下链接下载DIOR-R数据集：
[DIOR-R数据集下载链接](https://pan.baidu.com/s/104I7pegeWrdeKCguCpntCQ?pwd=1234)

### 数据集准备

1. 下载DIOR-R数据集并解压到`data/`目录下
2. 执行数据预处理脚本：

```shell
python tools/data/diorr/split_diorr.py
```

处理后的数据集结构应如下：

```
data/
├── split_ms_diorr
│   ├── trainval
│   │   ├── annfiles
│   │   └── images
│   └── test
│       ├── annfiles
│       └── images
└── split_ss_diorr
    └── val
        ├── annfiles
        └── images

```

## 模型训练

### 训练命令

```shell
python ./tools/train.py ./configs/rotated_sfpd_det/dior-r/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_xyawh321v.py
```

### 测试命令

```shell
python ./tools/test.py ./configs/rotated_sfpd_det/dior-r/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_xyawh321v.py ./work_dirs/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_xyawh321v/xxx.pth --eval mAP
```

## 预训练模型

您可以从以下链接下载预训练模型：
[预训练模型下载链接](https://pan.baidu.com/s/1pisK3HUCCZkVOFJLYlVYGg?pwd=1234)

## 引用

如果您在研究中使用了本项目，请引用以下论文：

```bibtex
@inproceedings{your-paper,
  title   = {Your Paper Title},
  author  = {Your Name},
  booktitle={Proceedings of the Conference},
  year={2023}
}

@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

## 许可证

本项目采用[Apache 2.0 license](LICENSE)许可证。
