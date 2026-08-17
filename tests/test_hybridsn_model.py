"""HybridSN 论文基线的结构、张量和梯度验收。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.HybridSN模型 import HybridSN


def test_pavia_architecture_and_parameter_count_match_paper() -> None:
    model = HybridSN()

    assert (model.conv3d_1.in_channels, model.conv3d_1.out_channels) == (1, 8)
    assert model.conv3d_1.kernel_size == (7, 3, 3)
    assert (model.conv3d_2.in_channels, model.conv3d_2.out_channels) == (8, 16)
    assert model.conv3d_2.kernel_size == (5, 3, 3)
    assert (model.conv3d_3.in_channels, model.conv3d_3.out_channels) == (16, 32)
    assert model.conv3d_3.kernel_size == (3, 3, 3)
    assert all(layer.padding == (0, 0, 0) for layer in (
        model.conv3d_1,
        model.conv3d_2,
        model.conv3d_3,
    ))
    assert all(layer.stride == (1, 1, 1) for layer in (
        model.conv3d_1,
        model.conv3d_2,
        model.conv3d_3,
    ))

    assert (model.conv2d.in_channels, model.conv2d.out_channels) == (96, 64)
    assert model.conv2d.kernel_size == (3, 3)
    assert model.conv2d.padding == (0, 0)
    assert model.conv2d.stride == (1, 1)
    assert (model.fc1.in_features, model.fc1.out_features) == (18_496, 256)
    assert (model.fc2.in_features, model.fc2.out_features) == (256, 128)
    assert (model.classifier.in_features, model.classifier.out_features) == (128, 9)
    assert model.dropout1.p == pytest.approx(0.4)
    assert model.dropout2.p == pytest.approx(0.4)

    assert not any(isinstance(module, nn.Softmax) for module in model.modules())
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for module in model.modules()
    )
    assert not any(
        isinstance(
            module,
            (
                nn.MaxPool1d,
                nn.MaxPool2d,
                nn.MaxPool3d,
                nn.AvgPool1d,
                nn.AvgPool2d,
                nn.AvgPool3d,
            ),
        )
        for module in model.modules()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_844_793


def test_forward_layer_shapes_and_logits_contract() -> None:
    model = HybridSN().eval()
    observed: dict[str, tuple[int, ...]] = {}
    handles = []
    handles.append(
        model.conv2d.register_forward_pre_hook(
            lambda _module, inputs: observed.__setitem__(
                "reshape_3d_to_2d", tuple(inputs[0].shape)
            )
        )
    )
    for name in (
        "conv3d_1",
        "conv3d_2",
        "conv3d_3",
        "conv2d",
        "fc1",
        "fc2",
        "classifier",
    ):
        layer = getattr(model, name)
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_name=name: observed.__setitem__(
                    layer_name, tuple(output.shape)
                )
            )
        )

    with torch.inference_mode():
        logits = model(torch.zeros(2, 1, 15, 25, 25))
    for handle in handles:
        handle.remove()

    assert observed == {
        "conv3d_1": (2, 8, 9, 23, 23),
        "conv3d_2": (2, 16, 5, 21, 21),
        "conv3d_3": (2, 32, 3, 19, 19),
        "reshape_3d_to_2d": (2, 96, 19, 19),
        "conv2d": (2, 64, 17, 17),
        "fc1": (2, 256),
        "fc2": (2, 128),
        "classifier": (2, 9),
    }
    assert logits.shape == (2, 9)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2))


def test_cross_entropy_backward_produces_finite_gradients() -> None:
    torch.manual_seed(1442)
    model = HybridSN().train()
    inputs = torch.randn(1, 1, 15, 25, 25)
    labels = torch.tensor([3], dtype=torch.long)

    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 15, 25, 25),
        (2, 3, 15, 25, 25),
        (2, 1, 14, 25, 25),
        (2, 1, 15, 23, 23),
    ],
)
def test_rejects_inputs_outside_fixed_contract(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="HybridSN expects"):
        HybridSN()(torch.zeros(shape))


@pytest.mark.parametrize("num_classes", [1, 8, 10])
def test_rejects_non_pavia_class_counts(num_classes: int) -> None:
    with pytest.raises(ValueError, match="num_classes=9"):
        HybridSN(num_classes=num_classes)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.1])
def test_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="0 <= dropout < 1"):
        HybridSN(dropout=dropout)
