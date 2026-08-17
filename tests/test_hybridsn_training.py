"""Tests for reusable baseline train/inference helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training.hybridsn_baseline import infer_loader, train_one_epoch


class TinyTraceableDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.tensor(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input": self.inputs[index],
            "label": self.labels[index],
            "raw_label": self.labels[index] + 1,
            "coordinate": torch.tensor([index, 0], dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def test_train_and_inference_helpers_preserve_traceability() -> None:
    torch.manual_seed(7)
    loader = DataLoader(TinyTraceableDataset(), batch_size=2, shuffle=False)
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    train_stats = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        non_blocking=False,
    )
    output = infer_loader(
        model,
        loader,
        torch.device("cpu"),
        criterion=criterion,
        non_blocking=False,
    )

    assert train_stats["samples"] == 4
    assert np.isfinite(train_stats["loss"])
    assert train_stats["seconds"] > 0
    assert output.labels.shape == (4,)
    assert output.predictions.shape == (4,)
    np.testing.assert_array_equal(output.sample_indices, np.arange(4))
    np.testing.assert_array_equal(output.coordinates[:, 0], np.arange(4))
    assert output.loss is not None and np.isfinite(output.loss)
    assert output.accuracy is not None and 0.0 <= output.accuracy <= 1.0
    assert output.throughput_samples_per_second > 0
