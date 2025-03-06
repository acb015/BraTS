import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复OpenMP冲突警告

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
from typing import Tuple
import random
from scipy.ndimage import zoom


class BratsDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 phase: str = "train",
                 patch_size: Tuple[int] = (96, 96, 96),
                 modalities: Tuple[str] = ("t1", "t1ce", "t2", "flair")):
        """
        Args:
            data_root: 数据集根目录路径
            phase: 训练阶段（train/val）
            patch_size: 训练使用的patch尺寸
            modalities: 使用的模态顺序
        """
        self.data_root = Path(data_root)
        self.phase = phase
        self.patch_size = patch_size
        self.modalities = modalities

        # 自动扫描病例目录
        self.case_dirs = sorted([d for d in (self.data_root / phase).iterdir() if d.is_dir()])

        # 缓存归一化参数（加速训练）
        self.norm_params = {}  # {case_id: (mean, std)}

        # 统一重采样参数（BraTS原始尺寸为240×240×155）
        self.target_spacing = (1.0, 1.0, 1.0)  # 目标体素间距

    def __len__(self):
        return len(self.case_dirs)

    def _load_nii(self, path: Path) -> np.ndarray:
        """加载NIFTI文件并返回numpy数组"""
        return nib.load(str(path)).get_fdata().astype(np.float32)

    def _normalize(self, modality: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """基于脑部区域的Z-Score归一化"""
        masked = modality[mask > 0]
        mean = masked.mean()
        std = masked.std()
        return (modality - mean) / (std + 1e-8)

    def _resample(self, data: np.ndarray, original_spacing: Tuple[float]) -> np.ndarray:
        """重采样到统一体素间距"""
        zoom_factor = [
            original_spacing[i] / self.target_spacing[i]
            for i in range(3)
        ]
        return zoom(data, zoom_factor, order=1)

    def _random_crop(self, data: np.ndarray) -> np.ndarray:
        """随机裁剪生成patch"""
        d, h, w = data.shape[-3:]
        pd, ph, pw = self.patch_size

        # 随机选择裁剪起始点
        d_start = random.randint(0, d - pd) if d > pd else 0
        h_start = random.randint(0, h - ph) if h > ph else 0
        w_start = random.randint(0, w - pw) if w > pw else 0

        return data[
            ...,
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]

    def _load_case(self, case_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载单个病例的多模态数据和标签"""
        case_id = case_dir.name
        original_spacing = None  # 显式初始化变量

        # 加载四个模态数据
        modalities = []
        for mod in self.modalities:
            mod_path = case_dir / f"{case_dir.name}_{mod}.nii.gz"
            if not mod_path.exists():
                raise FileNotFoundError(f"模态文件缺失: {mod_path}")

            mod_data = self._load_nii(mod_path)

            # 第一个模态获取spacing参数
            if original_spacing is None:
                original_spacing = nib.load(str(mod_path)).header.get_zooms()

            mod_data = self._resample(mod_data, original_spacing)
            modalities.append(mod_data)

        # 加载标签
        label_path = case_dir / f"{case_dir.name}_seg.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"标签文件缺失: {label_path}")
        label = self._load_nii(label_path)
        label = self._resample(label, original_spacing)

        # 创建脑部区域mask（至少有一个模态有信号）
        brain_mask = np.sum(np.stack(modalities), axis=0) > 0

        # 归一化处理
        normalized_mods = []
        for mod in modalities:
            normalized = self._normalize(mod, brain_mask)
            normalized_mods.append(normalized)

        # 堆叠模态 [C, D, H, W]
        image = np.stack(normalized_mods, axis=0)
        label = np.expand_dims(label, axis=0)

        return image, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_dir = self.case_dirs[idx]
        try:
            image, label = self._load_case(case_dir)
        except Exception as e:
            print(f"加载病例 {case_dir.name} 失败: {str(e)}")
            raise

        # 数据增强（仅训练阶段）
        if self.phase == "train":
            # 随机水平翻转
            if random.random() > 0.5:
                image = np.flip(image, axis=-1)
                label = np.flip(label, axis=-1)

            # 随机旋转（0/90/180/270度）
            rot = random.choice([0, 1, 2, 3])
            image = np.rot90(image, rot, axes=(-2, -1))
            label = np.rot90(label, rot, axes=(-2, -1))

        # 随机裁剪
        if self.phase == "train":
            combined = np.concatenate([image, label], axis=0)
            cropped = self._random_crop(combined)
            image, label = cropped[:4], cropped[4:]
        else:
            # 验证阶段使用中心裁剪
            image = self._center_crop(image)
            label = self._center_crop(label)

        # 转换为torch张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # 标签处理：映射为3类（0: 背景, 1: WT, 2: ET）
        label = self._remap_labels(label)

        return image, label

    @staticmethod
    def _remap_labels(label: torch.Tensor) -> torch.Tensor:
        """将BraTS标签映射为4类肿瘤区域"""
        new_label = torch.zeros_like(label)
        new_label[label == 0] = 0  # 背景
        new_label[label == 1] = 1  # 坏死肿瘤核心 (NCR)
        new_label[label == 2] = 2  # 瘤周围水肿区域 (ED)
        new_label[label == 4] = 3  # 增强肿瘤 (ET)
        return new_label

    def _center_crop(self, data: np.ndarray) -> np.ndarray:
        """验证阶段中心裁剪"""
        _, d, h, w = data.shape
        pd, ph, pw = self.patch_size

        d_start = (d - pd) // 2
        h_start = (h - ph) // 2
        w_start = (w - pw) // 2

        return data[
            ...,
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]


def get_dataloader(data_root: str,
                   batch_size: int = 2,
                   phase: str = "train",
                   num_workers: int = 4) -> DataLoader:
    """获取数据加载器"""
    dataset = BratsDataset(data_root=data_root, phase=phase)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


if __name__ == "__main__":
    train_loader = get_dataloader(r"E:/Brain_Tumor_Segmentation/data", batch_size=2)
    sample = next(iter(train_loader))
    images, labels = sample
    print(f"图像尺寸: {images.shape}")  # [2, 4, 96, 96, 96]
    print(f"标签尺寸: {labels.shape}")  # [2, 1, 96, 96, 96]
    print(f"标签最小值: {labels.min().item()}, 最大值: {labels.max().item()}")  # 应为 0 和 2（3类）或 0 和 4（4类）