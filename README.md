![segmentation_sample_58](https://github.com/user-attachments/assets/7ef835f9-8f13-4338-8117-26a6637406b6)


# 轻量级多模态3D脑肿瘤分割网络

本项目实现了一个轻量级的多模态3D脑肿瘤分割网络，基于深度可分离卷积、模态-通道双重注意力机制和残差连接，适用于BraTS等多模态医学图像分割任务。网络参数量仅为1.8M，适合在资源受限的设备上部署。

## 项目特点

- **轻量化设计**：使用深度可分离卷积和轻量级注意力机制，显著减少参数量和计算量。
- **多模态融合**：支持T1、T1ce、T2、FLAIR四种模态的输入，通过模态-通道双重注意力机制实现多模态特征融合。
- **高效训练**：支持梯度累积、动态学习率调整和早停机制，提升训练效率。
- **可视化支持**：提供训练过程中的损失曲线和分割结果可视化，便于模型调试和分析。

## 代码结构

```
E:/Brain_Tumor_Segmentation/
├── data/
│   ├── train/
│   │   ├── BraTS_00000/
│   │   │   ├── BraTS_00000_t1.nii.gz
│   │   │   ├── BraTS_00000_t1ce.nii.gz
│   │   │   ├── BraTS_00000_t2.nii.gz
│   │   │   ├── BraTS_00000_flair.nii.gz
│   │   │   └── BraTS_00000_seg.nii.gz
│   │   └── ...
│   ├── val/
│   └── test/
├── src/
├──├──data_loader.py # 数据加载与预处理
├──├──model.py # 网络模型定义
├──├──train.py # 训练与验证流程
├──├──inference.py # 推理与可视化
├──└──checkpoints/ # 保存训练好的模型(训练好模型后自动创建，想放主目录下的，懒得改了）
└── results/    # 自动创建，保存推理结果
└── README.md               # 项目说明文档
```

## 环境依赖

- Python 3.8+
- PyTorch 1.10+
- nibabel
- numpy
- scipy
- matplotlib
- tqdm

可以通过以下命令安装依赖：
```bash
pip install torch nibabel numpy scipy matplotlib tqdm
```

## 数据集

本项目使用BraTS 2021数据集，包含多模态脑部MRI扫描（T1、T1ce、T2、FLAIR）及专家标注的分割标签。数据需要按照以下结构组织：
```
 data/
├── train/
│   ├── BraTS_00000/
│   │   ├── BraTS_00000_t1.nii.gz
│   │   ├── BraTS_00000_t1ce.nii.gz
│   │   ├── BraTS_00000_t2.nii.gz
│   │   ├── BraTS_00000_flair.nii.gz
│   │   └── BraTS_00000_seg.nii.gz
│   └── ...
├── val/
└── test/
```

## 使用方法

### 1. 训练模型
运行以下命令开始训练：
```bash
python train.py
```
训练过程中会保存最佳模型到 `checkpoints/` 目录，并输出训练损失和验证集Dice系数。

### 2. 推理与可视化
运行以下命令对测试集进行推理并可视化分割结果：
```bash
python inference.py
```
结果将保存到 `results/` 目录。

### 3. 自定义配置
可以通过修改 `train.py` 和 `inference.py` 中的配置参数来调整训练和推理过程，例如：
- 学习率 (`lr`)
- 批量大小 (`batch_size`)
- 训练轮数 (`epochs`)
- 模型路径 (`model_path`)




## 贡献

如果你在使用过程中遇到问题或有改进建议，请在 [Issues](https://github.com/ACB015/BraTS/issues) 中提出。



### 说明
 **数据集路径**：你需要根据实际的数据集路径修改 `data_root` 配置。

