"""改进版 HybridSN：BatchNorm + 残差连接 + 全局平均池化分类头。

本模块定义 ``ImprovedHybridSN``，用于在原始 HybridSN 基础上完成「改进模型
结构 / 提升精度」的进阶实验。相较原始 HybridSN，本模型的三处改动为：

1. **BatchNorm**：每个卷积层之后、ReLU 之前插入批归一化，稳定训练、加速收敛；
2. **残差连接**：每个 3D 卷积块与 2D 卷积块增加投影短接（projection shortcut），
   通道数变化时用同尺寸卷积投影对齐维度，改善梯度流动与特征复用；
3. **全局平均池化分类头**：用 ``AdaptiveAvgPool2d`` 取代原始
   ``Flatten(18496) -> Linear(256) -> Linear(128)`` 的庞大全连接头，
   使可训练参数量从约 484 万降至约 15 万（约 -97%）。

3D/2D 卷积结构（8/16/32 通道、7/5/3 光谱卷积核、3×3 空间卷积核、valid 填充）
与原始 HybridSN 保持一致，从而把比较严格限定在上述三处改动上。

输入输出接口与原始 HybridSN 一致：``N x 1 x 15 x 25 x 25 -> N x 9 logits``。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualBlock3D(nn.Module):
    """带投影短接的 3D 残差卷积块：``out = ReLU(BN(Conv(x)) + shortcut(x))``。

    主路径为 ``Conv3d -> BatchNorm``，短接在通道数变化时使用同尺寸卷积投影，
    输出空间/光谱尺寸与主路径完全一致，因此可逐元素相加。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spectral_kernel: int,
        spatial_kernel: int,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
            padding=0,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            projection = [
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
                    padding=0,
                )
            ]
            if use_batch_norm:
                projection.append(nn.BatchNorm3d(out_channels))
            self.shortcut = nn.Sequential(*projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.batch_norm(self.conv(x)) + self.shortcut(x))


class _ResidualBlock2D(nn.Module):
    """带投影短接的 2D 残差卷积块，结构与 3D 版本对应。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            projection = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)]
            if use_batch_norm:
                projection.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.batch_norm(self.conv(x)) + self.shortcut(x))


class ImprovedHybridSN(nn.Module):
    """BatchNorm + 残差连接 + 全局平均池化分类头的改进 HybridSN。

    Parameters
    ----------
    input_bands / patch_size / num_classes:
        PCA15、25×25 patch、9 分类的固定接口，与原始 HybridSN 一致。
    conv3d_channels / spectral_kernel_sizes / spatial_kernel_size / conv2d_channels:
        卷积结构超参数，默认与原始 HybridSN 完全相同。
    dense_units:
        全局平均池化之后的可选小型全连接瓶颈；默认为空，即直接 ``Linear(64, 9)``。
    dropout:
        仅作用于可选全连接瓶颈（默认无瓶颈时无效）。
    batch_normalization / residual_connections:
        控制是否启用批归一化与残差连接；改进模型默认两者均开启。
    """

    def __init__(
        self,
        *,
        input_bands: int = 15,
        patch_size: int = 25,
        num_classes: int = 9,
        conv3d_channels: tuple[int, ...] = (8, 16, 32),
        spectral_kernel_sizes: tuple[int, ...] = (7, 5, 3),
        spatial_kernel_size: int = 3,
        conv2d_channels: int = 64,
        dense_units: tuple[int, ...] = (),
        dropout: float = 0.4,
        batch_normalization: bool = True,
        residual_connections: bool = True,
    ) -> None:
        super().__init__()
        if len(conv3d_channels) != len(spectral_kernel_sizes):
            raise ValueError("conv3d_channels 与 spectral_kernel_sizes 必须等长")
        if not conv3d_channels:
            raise ValueError("conv3d_channels 不能为空")

        self.input_bands = int(input_bands)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)

        # ---- 3D 光谱-空间特征提取 ----
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, spectral_kernel in zip(
            conv3d_channels, spectral_kernel_sizes, strict=True
        ):
            if residual_connections:
                blocks.append(
                    _ResidualBlock3D(
                        in_channels,
                        out_channels,
                        spectral_kernel,
                        spatial_kernel_size,
                        batch_normalization,
                    )
                )
            else:
                plain = [
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=(spectral_kernel, spatial_kernel_size, spatial_kernel_size),
                        padding=0,
                    )
                ]
                if batch_normalization:
                    plain.append(nn.BatchNorm3d(out_channels))
                plain.append(nn.ReLU(inplace=True))
                blocks.append(nn.Sequential(*plain))
            in_channels = out_channels
        self.conv3d = nn.Sequential(*blocks)

        # 有效卷积后的光谱深度与空间尺寸
        spectral_depth = input_bands - sum(value - 1 for value in spectral_kernel_sizes)
        spatial_depth = patch_size - len(spectral_kernel_sizes) * (spatial_kernel_size - 1)

        # ---- 2D 高层空间特征提取 ----
        conv2d_in = conv3d_channels[-1] * spectral_depth
        if residual_connections:
            self.conv2d = _ResidualBlock2D(
                conv2d_in, conv2d_channels, spatial_kernel_size, batch_normalization
            )
        else:
            plain = [nn.Conv2d(conv2d_in, conv2d_channels, kernel_size=spatial_kernel_size, padding=0)]
            if batch_normalization:
                plain.append(nn.BatchNorm2d(conv2d_channels))
            plain.append(nn.ReLU(inplace=True))
            self.conv2d = nn.Sequential(*plain)

        # ---- 全局平均池化 + 轻量分类头 ----
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        head: list[nn.Module] = []
        in_features = conv2d_channels
        for out_features in dense_units:
            head.append(nn.Linear(in_features, out_features))
            head.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                head.append(nn.Dropout(dropout))
            in_features = out_features
        self.head = nn.Sequential(*head) if head else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)                      # (N, C, D, H, W)
        batch, channels, depth, height, width = x.shape
        x = x.reshape(batch, channels * depth, height, width)  # (N, C*D, H, W)
        x = self.conv2d(x)                      # (N, C2, H', W')
        x = self.pool(x)                        # (N, C2, 1, 1)
        x = torch.flatten(x, start_dim=1)       # (N, C2)
        x = self.head(x)
        return self.classifier(x)               # (N, num_classes) logits


__all__ = ["ImprovedHybridSN"]
