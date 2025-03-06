import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import amp
from data_loader import get_dataloader
from model import MiniUNet3D
from tqdm import tqdm
import os
from datetime import datetime
import random
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt  # 保留用于可视化


class DiceLoss(nn.Module):
    """自定义多类别 Dice 损失，解决类不平衡"""

    def __init__(self, num_classes=4, smooth=1):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)  # 转换为概率分布
        target = target.long()  # 确保 target 是长整型
        batch_size, _, depth, height, width = pred.shape
        dice_loss = 0.0

        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]  # 形状 [batch_size, depth, height, width]
            target_cls = (target == cls).float()  # 形状 [batch_size, depth, height, width]

            intersection = (pred_cls * target_cls).sum(dim=(1, 2, 3))  # 沿深度、高度、宽度求和
            union = pred_cls.sum(dim=(1, 2, 3)) + target_cls.sum(dim=(1, 2, 3))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1.0 - dice.mean()

        return dice_loss / self.num_classes


class DiceCELoss(nn.Module):
    """多类别 Dice + CrossEntropy 混合损失，解决类不平衡"""

    def __init__(self, num_classes=4, weights=[0.1, 0.4, 0.3, 0.2]):  # 更高的权重给 NCR
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = 0.7
        self.ce_weight = 0.3
        self.weights = torch.tensor(weights).cuda()  # 调整权重以平衡 NCR
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, pred, target):
        target = target.squeeze(1).long()  # 去掉通道维度，确保为长整型
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def calculate_dice(pred, target, label_map={0: 0, 1: 1, 2: 2, 4: 3}, num_classes=4):
    """计算各类别Dice系数，支持自定义标签映射"""
    dice_scores = []
    pred_mapped = pred.argmax(dim=1)
    target_mapped = target.squeeze(1)

    for cls in range(num_classes):
        orig_label = label_map.get(cls, cls)
        pred_cls = (pred_mapped == orig_label)
        target_cls = (target_mapped == orig_label)

        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float()

        dice = (2.0 * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())

    return np.array(dice_scores)


def train_epoch(model, loader, optimizer, scaler, device, train_losses, accumulation_steps=8):
    """单个训练epoch，使用梯度累积模拟大批量大小"""
    torch.cuda.empty_cache()  # 在 epoch 开始时清空缓
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Training", leave=False, disable=False)  # 显示进度条

    for step, (images, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)  # 使用 non_blocking 加速数据传输
        labels = labels.to(device, non_blocking=True).long()

        with amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # 平均损失到每个小批量

        scaler.scale(loss).backward()  # 累积梯度

        if (step + 1) % accumulation_steps == 0:  # 每 accumulation_steps 个小批量更新一次
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # 累加损失
        progress.set_postfix(loss=loss.item() * accumulation_steps,
                             lr=scheduler.get_last_lr()[0])  # 使用 scheduler.get_last_lr()

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    print(f"Epoch {len(train_losses)} 训练损失: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, loader, device, label_map={0: 0, 1: 1, 2: 2, 4: 3}):
    torch.cuda.empty_cache()  # 清空缓存
    """验证过程"""
    model.eval()
    dice_ncr, dice_ed, dice_et = [], [], []  # Label 1, 2, 4 分别对应 NCR, ED, ET

    for images, labels in tqdm(loader, desc="Validation", disable=False):  # 显示进度条
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        with amp.autocast('cuda'):
            outputs = model(images)
        scores = calculate_dice(outputs, labels, label_map)

        dice_ncr.append(scores[1])  # Label 1: NCR
        dice_ed.append(scores[2])  # Label 2: ED
        dice_et.append(scores[3])  # Label 4: ET

    avg_dice_ncr = np.mean(dice_ncr)
    avg_dice_ed = np.mean(dice_ed)
    avg_dice_et = np.mean(dice_et)
    avg_dice = (avg_dice_ncr + avg_dice_ed + avg_dice_et) / 3
    print(
        f"验证 - Epoch {len(val_dices) * 5}: Dice NCR: {avg_dice_ncr:.4f}, Dice ED: {avg_dice_ed:.4f}, Dice ET: {avg_dice_et:.4f}, Avg Dice: {avg_dice:.4f}")
    return avg_dice_ncr, avg_dice_ed, avg_dice_et


class EarlyStopper:
    """智能早停控制器"""

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_epoch = 0
        self.best_weights = None

    def __call__(self, model, current_score, epoch):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def visualize_prediction(model, val_loader, checkpoint_dir, timestamp):
    """可视化随机一个样本的预测结果，并添加后处理平滑边界"""
    model.eval()
    idx = random.randint(0, len(val_loader) - 1)
    images, labels = val_loader.dataset[idx]
    images = images.unsqueeze(0).to(device, non_blocking=True)
    labels = labels.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.softmax(outputs, dim=1).cpu().numpy()  # 使用 softmax 转换为概率
        # 应用高斯平滑以改善边界
        preds_smoothed = ndimage.gaussian_filter(preds, sigma=1.0)
        preds_final = np.argmax(preds_smoothed, axis=1).astype(float)  # 转换为类别标签
        preds = torch.from_numpy(preds_final).to(device).unsqueeze(1)  # 调整形状为 [1, 1, 96, 96, 96]

    true_label = labels.cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(true_label[:, :, true_label.shape[2] // 2], cmap='gray')
    axes[0].set_title('True Label')
    axes[1].imshow(preds[0, 0].cpu().numpy()[:, :, preds.shape[3] // 2], cmap='jet')
    axes[1].set_title('Predicted Label')
    axes[2].imshow(images[0, 0].cpu().numpy()[:, :, images.shape[3] // 2], cmap='gray')
    axes[2].set_title('T1 Image')
    plt.tight_layout()
    vis_path = f"{checkpoint_dir}/prediction_vis_{timestamp}.png"
    plt.savefig(vis_path, dpi=150)  # 降低 DPI 减少文件大小
    plt.close()
    print(f"Prediction visualization saved to {vis_path}")


if __name__ == "__main__":
    # 训练配置
    config = {
        "data_root": "E:/Brain_Tumor_Segmentation/data",
        "batch_size": 2,
        "num_workers": 8,
        "lr": 2e-4,
        "epochs": 200,
        "patch_size": (96, 96, 96),
        "patience": 20,
        "min_delta": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 初始化系统组件
    device = config["device"]
    model = MiniUNet3D(in_channels=4, num_classes=4).to(device)
    criterion = DiceCELoss(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=5e-5)  # 添加权重衰减防止过拟合
    scaler = amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)  # 动态调整学习率

    # 确保检查点目录存在
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory '{checkpoint_dir}' is ready.")

    # 初始化早停器
    early_stopper = EarlyStopper(
        patience=config["patience"],
        min_delta=config["min_delta"]
    )

    # 数据加载
    train_loader = get_dataloader(config["data_root"],
                                  batch_size=config["batch_size"],
                                  phase="train",
                                  num_workers=config["num_workers"])
    val_loader = get_dataloader(config["data_root"],
                                batch_size=2,
                                phase="val",
                                num_workers=config["num_workers"])

    # 训练循环
    best_dice = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_losses = []
    val_dices = []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}, Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # 训练阶段，使用梯度累积模拟更大批量
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, train_losses)

        # 验证阶段
        if epoch % 5 == 0:
            dice_ncr, dice_ed, dice_et = validate(model, val_loader, device)
            avg_dice = (dice_ncr + dice_ed + dice_et) / 3
            print(
                f"Validation - Epoch {epoch}: Dice NCR: {dice_ncr:.4f}, Dice ED: {dice_ed:.4f}, Dice ET: {dice_et:.4f}, Avg Dice: {avg_dice:.4f}")

            # 更新学习率调度器
            scheduler.step(avg_dice)

            # 早停判断
            if early_stopper(model, avg_dice, epoch):
                print(
                    f"\nEarly stopping triggered! No improvement after {epoch} epochs, best epoch: {early_stopper.best_epoch}")
                model.load_state_dict(early_stopper.best_weights)
                break

            # 保存检查点（仅在当前为最佳时）
            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model_{timestamp}.pth")
                print(f"Saved best model to {checkpoint_dir}/best_model_{timestamp}.pth")

            # 可视化预测结果（每 5 个 epoch）
            visualize_prediction(model, val_loader, checkpoint_dir, timestamp)

    # 训练结束后保存最终模型
    final_model_path = f"{checkpoint_dir}/final_model_{timestamp}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed! Final model saved to: {final_model_path}")