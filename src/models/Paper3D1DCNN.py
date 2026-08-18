"""论文复现模型：三维卷积神经网络用于机载高光谱树种分类。

复现自论文：

    Zhang, B., Zhao, L., & Zhang, X. (2020).
    "Three-dimensional convolutional neural network model for tree species
    classification using airborne hyperspectral images."
    Remote Sensing of Environment, 247, 111938.

本模块包含论文中的两个模型：

1. ``Paper3DCNN``   —— 基础模型：五层 3D 卷积 + 全连接分类头；
2. ``Paper3D1DCNN`` —— 轻量化改进：五层 3D 卷积主干 + 两层 1D 卷积，
   用一维卷积替代参数量集中的 ``6080 -> 128`` 全连接映射。

输入约定（与 HybridSN 走不同预处理支路，本分支**不做 PCA**）：

    x: [N, 1, B, 11, 11]

其中 N 为 batch size，B 为原始有效光谱波段数，空间 patch 为 11 x 11。

重要：

- 本论文模型使用**原始光谱波段**输入，模型分支不做 PCA；
- 末端输出 logits，不要加 Softmax，直接交给 ``nn.CrossEntropyLoss``；
- 五层有效 3D 卷积（spectral kernel=7, spatial kernel=3, padding=0）使
  光谱维每层减少 6、空间维每层减少 2，B 需 > 30（3D-1D-CNN 还需 > 42）。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


PAPER_PATCH_SIZE = 11
PAPER_SPECTRAL_KERNEL = 7
PAPER_SPATIAL_KERNEL = 3


def _check_common_args(spectral_bands: int, num_classes: int, patch_size: int) -> None:
    """校验论文模型的公共超参数合法性。"""
    if patch_size != PAPER_PATCH_SIZE:
        raise ValueError(
            f"严格复现要求 patch_size={PAPER_PATCH_SIZE}，但得到 {patch_size}。"
        )
    if spectral_bands <= 30:
        raise ValueError(
            "五层有效 3D 卷积（光谱核 7）会把光谱维减 30，因此 spectral_bands 必须 > 30。"
        )
    if num_classes < 2:
        raise ValueError("num_classes 必须 >= 2。")


def _check_input(x: torch.Tensor, spectral_bands: int) -> None:
    """校验输入 tensor 的维度约定 [N, 1, B, 11, 11]。"""
    if x.ndim != 5:
        raise ValueError(f"期望 5 维输入 [N,1,B,11,11]，实际 shape={tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"期望通道维 C=1，实际 C={x.shape[1]}")
    if x.shape[2] != spectral_bands:
        raise ValueError(
            f"模型按 B={spectral_bands} 构建，但输入 B={x.shape[2]}。"
        )
    if tuple(x.shape[-2:]) != (PAPER_PATCH_SIZE, PAPER_PATCH_SIZE):
        raise ValueError(
            f"期望空间 patch {PAPER_PATCH_SIZE}x{PAPER_PATCH_SIZE}，"
            f"实际 {tuple(x.shape[-2:])}。"
        )


class _FiveLayer3DBackbone(nn.Module):
    """论文两个模型共用的五层有效 3D 卷积主干。"""

    def __init__(self) -> None:
        super().__init__()
        k = (PAPER_SPECTRAL_KERNEL, PAPER_SPATIAL_KERNEL, PAPER_SPATIAL_KERNEL)
        self.conv1 = nn.Conv3d(1, 4, kernel_size=k)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=k)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=k)
        self.conv4 = nn.Conv3d(16, 32, kernel_size=k)
        self.conv5 = nn.Conv3d(32, 64, kernel_size=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


class Paper3DCNN(nn.Module):
    """论文基础模型：五层 3D-CNN + 全连接分类头。

    原论文 B=125, C=12 时的逐层 shape：

        [N,1,125,11,11]
        -> [N,4,119,9,9]
        -> [N,8,113,7,7]
        -> [N,16,107,5,5]
        -> [N,32,101,3,3]
        -> [N,64,95,1,1]
        -> Flatten(6080)
        -> FC(128)
        -> FC(12)

    B=125, C=12 时可训练参数量为 951,652，与论文表 4 一致。
    """

    def __init__(
        self,
        spectral_bands: int,
        num_classes: int,
        patch_size: int = PAPER_PATCH_SIZE,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _check_common_args(spectral_bands, num_classes, patch_size)

        self.spectral_bands = spectral_bands
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.features3d = _FiveLayer3DBackbone()
        self.drop_conv = nn.Dropout(dropout)

        # 五层有效光谱卷积（k=7）：B -> B - 5*(7-1) = B - 30。
        spectral_after_3d = spectral_bands - 30
        # 五层有效空间卷积（k=3）：11 -> 1，故 flatten 维度为 64 * (B-30)。
        flat_dim = 64 * spectral_after_3d

        self.fc1 = nn.Linear(flat_dim, 128)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.spectral_bands)
        x = self.features3d(x)
        x = self.drop_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)  # logits


class Paper3D1DCNN(nn.Module):
    """论文改进模型：五层 3D-CNN 主干 + 两层 1D-CNN。

    原论文 B=125, C=12 时的逐层 shape：

        [N,1,125,11,11]
        -> [N,64,95,1,1]
        -> 重排为 [N,64,95]
        -> Conv1d(64->48, k=7): [N,48,89]
        -> Conv1d(48->24, k=7): [N,24,83]
        -> Flatten(1992)
        -> FC(12)

    B=125, C=12 时可训练参数量为 225,292，约为基础 3D-CNN 的 23.7%。
    """

    def __init__(
        self,
        spectral_bands: int,
        num_classes: int,
        patch_size: int = PAPER_PATCH_SIZE,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _check_common_args(spectral_bands, num_classes, patch_size)
        if spectral_bands <= 42:
            raise ValueError(
                "两层有效 Conv1d（k=7）会把 3D 之后的序列再减 12；"
                "对 3D-1D-CNN，spectral_bands 必须 > 42。"
            )

        self.spectral_bands = spectral_bands
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.features3d = _FiveLayer3DBackbone()
        self.drop_conv = nn.Dropout(dropout)

        self.conv1d_1 = nn.Conv1d(64, 48, kernel_size=7)
        self.conv1d_2 = nn.Conv1d(48, 24, kernel_size=7)

        # B -> B-30 后接两层 k=7 有效 1D 卷积：再减 12。
        sequence_after_1d = spectral_bands - 42
        self.fc = nn.Linear(24 * sequence_after_1d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.spectral_bands)
        x = self.features3d(x)
        x = self.drop_conv(x)

        # 论文 patch 固定为 11x11，故此刻空间尺寸已为 1x1。
        # [N,64,B-30,1,1] -> [N,64,B-30]，匹配 PyTorch Conv1d 的 [N,C,L]。
        x = x.squeeze(-1).squeeze(-1)
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)  # logits


def count_trainable_parameters(model: nn.Module) -> int:
    """返回模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_paper_model(
    name: str,
    spectral_bands: int,
    num_classes: int,
    dropout: float = 0.5,
) -> nn.Module:
    """按名称构建论文模型，供统一训练脚本分派使用。"""
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"3dcnn", "paper3d", "paper3dcnn"}:
        return Paper3DCNN(spectral_bands, num_classes, dropout=dropout)
    if key in {"3d1dcnn", "paper3d1d", "paper3d1dcnn"}:
        return Paper3D1DCNN(spectral_bands, num_classes, dropout=dropout)
    raise ValueError(f"未知论文模型名称：{name}")


__all__ = [
    "Paper3DCNN",
    "Paper3D1DCNN",
    "build_paper_model",
    "count_trainable_parameters",
]
