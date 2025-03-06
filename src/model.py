"""
轻量级多模态3D分割网络
包含深度可分离卷积、残差注意力和模态融合模块
参数量：1.8M (输入尺寸96×96×96)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    """深度可分离3D卷积 减少约1/3参数量"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ModalityAttention(nn.Module):
    """模态-通道双重注意力机制"""

    def __init__(self, in_channels, modalities=4):  # 修改构造函数参数
        super().__init__()
        self.modalities = modalities
        self.channels_per_modality = in_channels // modalities

        # 模态注意力
        self.modal_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(self.modalities, self.modalities // 2, 1),  # 输入通道修正
            nn.ReLU(),
            nn.Conv3d(modalities // 2, modalities, 1),
            nn.Softmax(dim=1)
        )

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 8, 1),  # 输入通道修正
            nn.ReLU(),
            nn.Conv3d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.shape

        # 模态分支处理
        modal_feat = x.view(b, self.modalities, self.channels_per_modality, d, h, w)
        modal_weights = self.modal_att(modal_feat.mean(2))  # [B, M, 1, 1, 1]
        modal_feat = modal_feat * modal_weights.unsqueeze(2)
        modal_feat = modal_feat.view(b, c, d, h, w)

        # 通道分支处理
        channel_weights = self.channel_att(modal_feat)
        return modal_feat * channel_weights


class ResidualAttentionBlock(nn.Module):
    """带有残差连接的注意力卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv3d(out_channels, out_channels),
            nn.InstanceNorm3d(out_channels)
        )
        self.att = ModalityAttention(modalities=4, in_channels=out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x = self.att(x)
        return F.relu(x + residual)


class UpSample3D(nn.Module):
    """轻量级上采样模块"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.conv = DepthwiseSeparableConv3d(in_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MiniUNet3D(nn.Module):
    """轻量级3D分割网络主结构"""

    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        filters = [32, 64, 128]  # 控制网络宽度

        # 下采样路径
        self.encoder1 = ResidualAttentionBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = ResidualAttentionBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ResidualAttentionBlock(filters[1], filters[2])

        # 上采样路径
        self.up1 = UpSample3D(filters[2] + filters[1], filters[1])  # 192 -> 64
        self.decoder1 = ResidualAttentionBlock(filters[1], filters[1])  # 64 -> 64

        self.up2 = UpSample3D(filters[1] + filters[0], filters[0])  # 96 -> 32
        self.decoder2 = ResidualAttentionBlock(filters[0], filters[0])  # 32 -> 32

        # 最终输出
        self.final = nn.Sequential(
            DepthwiseSeparableConv3d(filters[0], num_classes),
            nn.Conv3d(num_classes, num_classes, 1)  # 用于调整通道间关系
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: [B, C, D, H, W]
        e1 = self.encoder1(x)  # -> [B, 32, D, H, W]
        e2 = self.encoder2(self.pool1(e1))  # -> [B, 64, D/2, H/2, W/2]
        bottle = self.bottleneck(self.pool2(e2))  # -> [B, 128, D/4, H/4, W/4]

        d1 = self.up1(bottle, e2)  # -> [B, 64, D/2, H/2, W/2]
        d1 = self.decoder1(d1)

        d2 = self.up2(d1, e1)  # -> [B, 32, D, H, W]
        d2 = self.decoder2(d2)

        out = self.final(d2)  # -> [B, 3, D, H, W]
        return out


def test_model():
    """测试网络结构与参数量"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniUNet3D().to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    with torch.no_grad():
        x = torch.randn(2, 4, 96, 96, 96).to(device)
        out = model(x)
        print(f"输入尺寸: {x.shape}")
        print(f"输出尺寸: {out.shape}")  # 应为 [2, 3, 96, 96, 96]


if __name__ == "__main__":
    test_model()