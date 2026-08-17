from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.可配置HybridSN import (
    ConfigurableHybridSN,
    HybridSNArchitecture,
    SigmoidOneVsRestLoss,
    build_classification_objective,
    probabilities_from_logits,
)


def test_default_configurable_model_matches_coursework_contract() -> None:
    model = ConfigurableHybridSN(input_bands=15, patch_size=25, num_classes=9)
    with torch.inference_mode():
        logits = model(torch.zeros(2, 1, 15, 25, 25))

    assert logits.shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_844_793
    assert not any(isinstance(module, (nn.Softmax, nn.Sigmoid)) for module in model.modules())


def test_yaml_architecture_knobs_change_shape_without_changing_output_contract() -> None:
    architecture = HybridSNArchitecture.from_mapping(
        {
            "model": {
                "dropout": 0.2,
                "architecture": {
                    "conv3d_channels": [6, 12],
                    "spectral_kernel_sizes": [5, 3],
                    "spatial_kernel_size": 3,
                    "conv2d_channels": 24,
                    "dense_units": [64],
                    "batch_normalization": True,
                },
            }
        }
    )
    model = ConfigurableHybridSN(
        input_bands=15,
        patch_size=25,
        num_classes=16,
        architecture=architecture,
    ).eval()
    with torch.inference_mode():
        logits = model(torch.randn(3, 1, 15, 25, 25))

    assert logits.shape == (3, 16)
    assert any(isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)) for module in model.modules())


@pytest.mark.parametrize("objective", ["softmax", "sigmoid"])
def test_classification_objectives_produce_finite_loss_and_predictions(
    objective: str,
) -> None:
    torch.manual_seed(1442)
    logits = torch.randn(4, 9, requires_grad=True)
    labels = torch.tensor([0, 2, 5, 8], dtype=torch.long)
    criterion = build_classification_objective(objective, num_classes=9)
    loss = criterion(logits, labels)
    loss.backward()
    probabilities = probabilities_from_logits(logits.detach(), objective)

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert probabilities.shape == logits.shape
    assert torch.equal(probabilities.argmax(dim=1), logits.detach().argmax(dim=1))
    if objective == "softmax":
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4))
    else:
        assert isinstance(criterion, SigmoidOneVsRestLoss)
        assert torch.all((0.0 < probabilities) & (probabilities < 1.0))


def test_configurable_model_rejects_inputs_outside_declared_contract() -> None:
    model = ConfigurableHybridSN(input_bands=15, patch_size=25, num_classes=9)
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(1, 1, 14, 25, 25))
