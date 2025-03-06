import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import BratsDataset
from model import MiniUNet3D


class TumorSegmentationInference:
    def __init__(self,
                 model_path: str,
                 data_root: str,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = MiniUNet3D(in_channels=4, num_classes=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.dataset = BratsDataset(data_root, phase='test')

        self.color_map = {
            0: [0.5, 0.5, 0.5],  # 背景 - 灰色
            1: [1, 0, 0],  # NCR - 红色
            2: [0, 1, 0],  # ED - 绿色
            3: [0, 0, 1]  # ET - 蓝色
        }

    def _predict_sample(self, image: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            outputs = self.model(image)
            pred = torch.softmax(outputs, dim=1)
            pred = pred.argmax(dim=1).cpu().numpy()

        return pred[0]

    def _has_tumor(self, pred_label: np.ndarray) -> bool:
        """检查预测结果中是否存在肿瘤区域"""
        return np.any(pred_label > 0)

    def visualize_segmentation(self,
                               output_dir: str,
                               num_samples: int = 10,
                               slice_idx: int = None):
        os.makedirs(output_dir, exist_ok=True)

        tumor_samples_count = 0
        for i in range(len(self.dataset)):
            if tumor_samples_count >= num_samples:
                break

            image, true_label = self.dataset[i]
            pred_label = self._predict_sample(image)

            # 仅处理包含肿瘤的样本
            if not self._has_tumor(pred_label):
                continue

            if slice_idx is None:
                slice_idx = image.shape[-1] // 2

            plt.figure(figsize=(12, 5))

            # 原图（T2模态）
            plt.subplot(1, 2, 1)
            plt.imshow(image[1, :, :, slice_idx], cmap='gray')
            plt.title(f'T2 Image - Sample {tumor_samples_count + 1}')
            plt.axis('off')

            # 分割结果
            plt.subplot(1, 2, 2)
            seg_overlay = np.zeros((*pred_label.shape, 3), dtype=float)

            for label, color in self.color_map.items():
                mask = pred_label == label
                seg_overlay[mask] = color

            plt.imshow(image[1, :, :, slice_idx], cmap='gray', alpha=0.5)
            plt.imshow(seg_overlay[:, :, slice_idx], alpha=0.5)
            plt.title(f'Predicted Segmentation - Sample {tumor_samples_count + 1}')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'segmentation_sample_{tumor_samples_count + 1}.png'), dpi=300)
            plt.close()

            tumor_samples_count += 1


def main():
    config = {
        'model_path': 'checkpoints/best_model_20250304_115522.pth',  # 替换为你的模型路径
        'data_root': 'E:/Brain_Tumor_Segmentation/data',
        'output_dir': 'E:/Brain_Tumor_Segmentation/results',
        'num_samples': 80
    }

    inferencer = TumorSegmentationInference(
        model_path=config['model_path'],
        data_root=config['data_root']
    )

    inferencer.visualize_segmentation(
        output_dir=config['output_dir'],
        num_samples=config['num_samples']
    )


if __name__ == "__main__":
    main()