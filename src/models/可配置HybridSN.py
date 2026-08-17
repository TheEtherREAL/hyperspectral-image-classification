"""YAML-friendly HybridSN and mutually exclusive classification objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HybridSNArchitecture:
    """Architecture knobs that may be changed between controlled experiments."""

    conv3d_channels: tuple[int, ...] = (8, 16, 32)
    spectral_kernel_sizes: tuple[int, ...] = (7, 5, 3)
    spatial_kernel_size: int = 3
    conv2d_channels: int = 64
    dense_units: tuple[int, ...] = (256, 128)
    dropout: float = 0.4
    batch_normalization: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HybridSNArchitecture":
        model = values.get("model", values)
        architecture = model.get("architecture", model)
        instance = cls(
            conv3d_channels=tuple(
                int(value) for value in architecture.get("conv3d_channels", (8, 16, 32))
            ),
            spectral_kernel_sizes=tuple(
                int(value)
                for value in architecture.get("spectral_kernel_sizes", (7, 5, 3))
            ),
            spatial_kernel_size=int(architecture.get("spatial_kernel_size", 3)),
            conv2d_channels=int(architecture.get("conv2d_channels", 64)),
            dense_units=tuple(
                int(value) for value in architecture.get("dense_units", (256, 128))
            ),
            dropout=float(model.get("dropout", architecture.get("dropout", 0.4))),
            batch_normalization=bool(
                architecture.get(
                    "batch_normalization", model.get("batch_normalization", False)
                )
            ),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if not self.conv3d_channels or len(self.conv3d_channels) != len(
            self.spectral_kernel_sizes
        ):
            raise ValueError("3D channel and spectral-kernel lists must be non-empty/aligned")
        if any(value < 1 for value in self.conv3d_channels + self.spectral_kernel_sizes):
            raise ValueError("all 3D channel/kernel values must be positive")
        if self.spatial_kernel_size < 1 or self.spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be a positive odd integer")
        if self.conv2d_channels < 1 or not self.dense_units:
            raise ValueError("conv2d_channels and dense_units must be positive")
        if any(value < 1 for value in self.dense_units):
            raise ValueError("dense_units must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")


class ConfigurableHybridSN(nn.Module):
    """HybridSN whose widths, kernels and dense layers come from YAML.

    The output is always raw logits. Softmax/sigmoid belongs to the objective
    and probability conversion, not to the network body; this avoids unstable
    combinations such as applying an activation before a logits-based loss.
    """

    def __init__(
        self,
        *,
        input_bands: int,
        patch_size: int,
        num_classes: int,
        architecture: HybridSNArchitecture | None = None,
    ) -> None:
        super().__init__()
        self.input_bands = int(input_bands)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.architecture = architecture or HybridSNArchitecture()
        self.architecture.validate()
        if min(self.input_bands, self.patch_size, self.num_classes) < 1:
            raise ValueError("input_bands, patch_size and num_classes must be positive")

        conv3d: list[nn.Module] = []
        in_channels = 1
        for out_channels, spectral_kernel in zip(
            self.architecture.conv3d_channels,
            self.architecture.spectral_kernel_sizes,
            strict=True,
        ):
            conv3d.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(
                        spectral_kernel,
                        self.architecture.spatial_kernel_size,
                        self.architecture.spatial_kernel_size,
                    ),
                    padding=0,
                )
            )
            if self.architecture.batch_normalization:
                conv3d.append(nn.BatchNorm3d(out_channels))
            conv3d.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv3d = nn.Sequential(*conv3d)

        spectral_depth = self.input_bands - sum(
            value - 1 for value in self.architecture.spectral_kernel_sizes
        )
        spatial_depth = self.patch_size - len(
            self.architecture.spectral_kernel_sizes
        ) * (self.architecture.spatial_kernel_size - 1)
        if spectral_depth < 1 or spatial_depth < self.architecture.spatial_kernel_size:
            raise ValueError(
                "input bands/patch are too small for configured valid convolutions: "
                f"bands={self.input_bands}, patch={self.patch_size}"
            )

        conv2d_modules: list[nn.Module] = [
            nn.Conv2d(
                self.architecture.conv3d_channels[-1] * spectral_depth,
                self.architecture.conv2d_channels,
                kernel_size=self.architecture.spatial_kernel_size,
                padding=0,
            )
        ]
        if self.architecture.batch_normalization:
            conv2d_modules.append(nn.BatchNorm2d(self.architecture.conv2d_channels))
        conv2d_modules.append(nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(*conv2d_modules)

        final_spatial = spatial_depth - self.architecture.spatial_kernel_size + 1
        flattened = self.architecture.conv2d_channels * final_spatial * final_spatial
        dense: list[nn.Module] = []
        in_features = flattened
        for out_features in self.architecture.dense_units:
            dense.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.architecture.dropout),
                ]
            )
            in_features = out_features
        self.dense = nn.Sequential(*dense)
        self.classifier = nn.Linear(in_features, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (1, self.input_bands, self.patch_size, self.patch_size)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected N x {expected}, got {tuple(x.shape)}")
        x = self.conv3d(x)
        batch, channels, depth, height, width = x.shape
        x = x.reshape(batch, channels * depth, height, width)
        x = self.conv2d(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(self.dense(x))


class SigmoidOneVsRestLoss(nn.Module):
    """BCE-with-logits against one-hot labels for the sigmoid ablation."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = F.one_hot(labels, num_classes=self.num_classes).to(logits.dtype)
        return self.loss(logits, targets)


def build_classification_objective(name: str, num_classes: int) -> nn.Module:
    key = str(name).strip().lower()
    if key == "softmax":
        return nn.CrossEntropyLoss()
    if key == "sigmoid":
        return SigmoidOneVsRestLoss(num_classes)
    raise ValueError("classification objective must be 'softmax' or 'sigmoid'")


def probabilities_from_logits(logits: torch.Tensor, objective: str) -> torch.Tensor:
    key = str(objective).strip().lower()
    if key == "softmax":
        return torch.softmax(logits, dim=1)
    if key == "sigmoid":
        return torch.sigmoid(logits)
    raise ValueError("classification objective must be 'softmax' or 'sigmoid'")


__all__ = [
    "ConfigurableHybridSN",
    "HybridSNArchitecture",
    "SigmoidOneVsRestLoss",
    "build_classification_objective",
    "probabilities_from_logits",
]
