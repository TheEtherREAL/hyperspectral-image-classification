"""论文复现模型 Paper3DCNN / Paper3D1DCNN / 改进版的参数与 shape 验收。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.Paper3D1DCNN import (
    Paper3DCNN,
    Paper3D1DCNN,
    build_paper_model,
    count_trainable_parameters,
)
from src.models.改进Paper3D1DCNN import ImprovedPaper3D1DCNN


def test_paper_parameter_counts_match_original_paper() -> None:
    # 论文原设置 B=125, C=12。
    model_3d = Paper3DCNN(spectral_bands=125, num_classes=12)
    model_3d1d = Paper3D1DCNN(spectral_bands=125, num_classes=12)

    assert count_trainable_parameters(model_3d) == 951_652
    assert count_trainable_parameters(model_3d1d) == 225_292


def test_paper_output_shapes_match_original_paper() -> None:
    x = torch.randn(2, 1, 125, 11, 11)
    with torch.inference_mode():
        y_3d = Paper3DCNN(125, 12)(x)
        y_3d1d = Paper3D1DCNN(125, 12)(x)

    assert y_3d.shape == (2, 12)
    assert y_3d1d.shape == (2, 12)


def test_paper_models_return_raw_logits_without_softmax_or_batchnorm() -> None:
    for model in (Paper3DCNN(125, 12), Paper3D1DCNN(125, 12)):
        assert not any(isinstance(m, (nn.Softmax, nn.Sigmoid)) for m in model.modules())
        assert not any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in model.modules()
        )


@pytest.mark.parametrize(
    ("bands", "classes"),
    [(103, 9), (200, 16), (204, 16)],
)
def test_paper_models_accept_three_dataset_band_and_class_counts(
    bands: int, classes: int
) -> None:
    x = torch.randn(2, 1, bands, 11, 11)
    for model in (Paper3DCNN(bands, classes), Paper3D1DCNN(bands, classes)):
        with torch.inference_mode():
            logits = model(x)
        assert logits.shape == (2, classes)
        assert torch.isfinite(logits).all()


def test_paper_models_backward_produces_finite_gradients() -> None:
    torch.manual_seed(1442)
    for model in (Paper3DCNN(103, 9), Paper3D1DCNN(103, 9)):
        model.train()
        loss = F.cross_entropy(model(torch.randn(4, 1, 103, 11, 11)), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("paper3dcnn", Paper3DCNN),
        ("3D-CNN", Paper3DCNN),
        ("paper3d1dcnn", Paper3D1DCNN),
        ("3d_1d_cnn", Paper3D1DCNN),
    ],
)
def test_build_paper_model_factory(name: str, expected_type: type) -> None:
    assert isinstance(build_paper_model(name, spectral_bands=103, num_classes=9), expected_type)


def test_build_paper_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="未知论文模型名称"):
        build_paper_model("resnet", spectral_bands=103, num_classes=9)


@pytest.mark.parametrize("bands", [30, 31, 42])
def test_paper_3d1dcnn_rejects_too_few_bands(bands: int) -> None:
    with pytest.raises(ValueError):
        Paper3D1DCNN(spectral_bands=bands, num_classes=9)


def test_improved_paper_model_is_lighter_and_has_batchnorm() -> None:
    original = Paper3D1DCNN(spectral_bands=200, num_classes=16)
    improved = ImprovedPaper3D1DCNN(spectral_bands=200, num_classes=16)

    x = torch.randn(2, 1, 200, 11, 11)
    with torch.inference_mode():
        y_original = original(x)
        y_improved = improved(x)

    assert y_original.shape == y_improved.shape == (2, 16)
    assert count_trainable_parameters(improved) < count_trainable_parameters(original)
    assert any(isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)) for m in improved.modules())
