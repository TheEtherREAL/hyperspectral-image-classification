"""HybridSN 论文基线模型。

本模块只定义模型结构，不包含数据预处理、损失函数、优化器、训练循环或
评价逻辑。首个固定接口面向 Pavia University 的 PCA15 + patch25 路线：

    input:  N x 1 x 15 x 25 x 25
    output: N x 9 logits

结构依据 Roy et al. 的 HybridSN 原论文及作者公开 Keras 实现。PyTorch 的
``Conv3d`` 卷积核顺序为 ``(spectral_depth, height, width)``，因此论文中的
空间-空间-光谱核 ``3 x 3 x 7`` 在此写作 ``(7, 3, 3)``。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HybridSN(nn.Module):
    """论文 HybridSN 在 Pavia University 上的 PyTorch 基线实现。

    该模型严格使用固定的 PCA15、25 x 25 patch 和九分类接口。输出为未经
    Softmax 的 logits，以便直接交给 ``torch.nn.CrossEntropyLoss``。

    Parameters
    ----------
    num_classes:
        输出类别数。首个 Pavia University 基线固定为 9。
    dropout:
        两个全连接隐藏层之后的 Dropout 概率。论文实现使用 0.4。
    """

    INPUT_CHANNELS = 1
    INPUT_BANDS = 15
    PATCH_SIZE = 25
    NUM_CLASSES = 9
    RESHAPED_CHANNELS = 32 * 3
    FLATTENED_FEATURES = 64 * 17 * 17

    def __init__(self, *, num_classes: int = NUM_CLASSES, dropout: float = 0.4) -> None:
        super().__init__()
        if num_classes != self.NUM_CLASSES:
            raise ValueError(
                "the first Pavia University HybridSN baseline requires num_classes=9"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        # PyTorch Conv3d uses N x C x D x H x W. All convolutions use the
        # paper's stride=1 and valid padding (padding=0).
        self.conv3d_1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=(7, 3, 3),
            stride=1,
            padding=0,
        )
        self.conv3d_2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=(5, 3, 3),
            stride=1,
            padding=0,
        )
        self.conv3d_3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0,
        )

        # After the third 3D convolution the tensor is N x 32 x 3 x 19 x 19.
        # Merging feature channels and residual spectral depth gives 96 2D
        # channels, matching the HybridSN 3D-to-2D transition.
        self.conv2d = nn.Conv2d(
            in_channels=self.RESHAPED_CHANNELS,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
        )

        self.fc1 = nn.Linear(self.FLATTENED_FEATURES, 256)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the Glorot/Xavier initialization employed by the Keras model."""

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "HybridSN expects a 5D tensor shaped N x 1 x 15 x 25 x 25, "
                f"got {tuple(x.shape)}"
            )
        expected = (
            self.INPUT_CHANNELS,
            self.INPUT_BANDS,
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        )
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                "HybridSN expects each sample to have shape 1 x 15 x 25 x 25, "
                f"got {tuple(x.shape[1:])}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one nine-class logit vector for every input patch."""

        self._validate_input(x)

        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))

        batch_size, channels, spectral_depth, height, width = x.shape
        x = x.reshape(batch_size, channels * spectral_depth, height, width)

        x = F.relu(self.conv2d(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.classifier(x)


__all__ = ["HybridSN"]
