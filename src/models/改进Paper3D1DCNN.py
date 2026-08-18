"""改进版论文 3D-1D-CNN：BatchNorm + 全局平均池化分类头（轻量化改进）。

本模块定义 ``ImprovedPaper3D1DCNN``，用于在复现论文 ``Paper3D1DCNN`` 基础上
完成「改进模型结构 / 提升精度」的选做加分实验。相较原始论文 3D-1D-CNN，改动为：

1. **BatchNorm**：每个 3D 卷积层、每个 1D 卷积层之后、ReLU 之前插入批归一化，
   稳定训练、加速收敛；
2. **全局平均池化分类头**：用 ``AdaptiveAvgPool1d`` 取代原始
   ``Flatten(24 * L) -> Linear(C)`` 的末尾全连接，把 24 通道的序列在长度维平均
   为 24 维向量后直接 ``Linear(24, C)``，移除参数量最集中的末尾全连接层。

3D 卷积主干（1→4→8→16→32→64 通道、光谱核 7、空间核 3、valid 填充）与两层 1D
卷积（64→48→24、核 7）与原始论文完全一致，从而把比较严格限定在「BN + GAP」
两处改动上。输入输出接口与原始一致：``N x 1 x B x 11 x 11 -> N x C logits``。

注：此处**不引入残差短接**。残差块在通道数变化时需要投影卷积对齐维度，会为
本就轻量的 3D-1D-CNN 额外增加卷积参数；论文模型末端全连接较小，GAP 的参数量
收益会被投影卷积抵消。因此轻量改进以「BN（近零参数成本）+ GAP（移除末尾全连接）」
为主，确保改进方向真正指向更小参数量。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PAPER_PATCH_SIZE = 11
PAPER_SPECTRAL_KERNEL = 7
PAPER_SPATIAL_KERNEL = 3


class ImprovedPaper3D1DCNN(nn.Module):
    """BatchNorm + 全局平均池化分类头的改进论文 3D-1D-CNN。

    Parameters
    ----------
    spectral_bands / num_classes / patch_size:
        原始光谱波段数、类别数与 11x11 patch，与原始 Paper3D1DCNN 一致。
    dropout:
        仅作用于 3D 主干之后的 Dropout，默认 0.5（与论文一致）。
    batch_normalization:
        控制是否启用批归一化；改进模型默认开启。
    """

    def __init__(
        self,
        spectral_bands: int,
        num_classes: int,
        patch_size: int = PAPER_PATCH_SIZE,
        dropout: float = 0.5,
        batch_normalization: bool = True,
    ) -> None:
        super().__init__()
        if patch_size != PAPER_PATCH_SIZE:
            raise ValueError(f"严格复现要求 patch_size={PAPER_PATCH_SIZE}")
        if spectral_bands <= 42:
            raise ValueError("spectral_bands 必须 > 42（与 Paper3D1DCNN 一致）")
        if num_classes < 2:
            raise ValueError("num_classes 必须 >= 2")

        self.spectral_bands = spectral_bands
        self.num_classes = num_classes
        self.patch_size = patch_size

        # ---- 五层 3D 主干（1->4->8->16->32->64，核 7x3x3，valid）----
        channels = (1, 4, 8, 16, 32, 64)
        conv3d: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            conv3d.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(PAPER_SPECTRAL_KERNEL, PAPER_SPATIAL_KERNEL, PAPER_SPATIAL_KERNEL),
                    padding=0,
                )
            )
            if batch_normalization:
                conv3d.append(nn.BatchNorm3d(out_channels))
            conv3d.append(nn.ReLU(inplace=True))
        self.features3d = nn.Sequential(*conv3d)
        self.drop_conv = nn.Dropout(dropout)

        # ---- 两层 1D 卷积（64->48->24，核 7，valid）----
        conv1d: list[nn.Module] = [nn.Conv1d(64, 48, kernel_size=7)]
        if batch_normalization:
            conv1d.append(nn.BatchNorm1d(48))
        conv1d.append(nn.ReLU(inplace=True))
        conv1d.append(nn.Conv1d(48, 24, kernel_size=7))
        if batch_normalization:
            conv1d.append(nn.BatchNorm1d(24))
        conv1d.append(nn.ReLU(inplace=True))
        self.conv1d = nn.Sequential(*conv1d)

        # ---- 全局平均池化 + 轻量分类头 ----
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(24, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[2] != self.spectral_bands:
            raise ValueError(
                f"期望输入 [N,1,{self.spectral_bands},11,11]，实际 {tuple(x.shape)}"
            )
        x = self.features3d(x)          # (N, 64, B-30, 1, 1)
        x = self.drop_conv(x)
        x = x.squeeze(-1).squeeze(-1)   # (N, 64, B-30)
        x = self.conv1d(x)              # (N, 24, B-42)
        x = self.pool(x)                # (N, 24, 1)
        x = torch.flatten(x, 1)         # (N, 24)
        return self.classifier(x)       # (N, C) logits


__all__ = ["ImprovedPaper3D1DCNN"]
