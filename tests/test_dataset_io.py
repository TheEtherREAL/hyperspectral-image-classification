from pathlib import Path

import numpy as np
import pytest

from src.datasets.数据读取 import validate_dataset
from src.datasets.数据集注册 import DatasetSpec


SPEC = DatasetSpec(
    name="toy",
    data_file="toy.mat",
    label_file="toy_gt.mat",
    data_key="toy",
    label_key="toy_gt",
    class_names=("A", "B"),
)


def test_validate_dataset_accepts_contiguous_labels() -> None:
    cube = np.zeros((2, 3, 4), dtype=np.uint16)
    labels = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
    validate_dataset(cube, labels, SPEC)


def test_validate_dataset_rejects_spatial_mismatch() -> None:
    cube = np.zeros((2, 3, 4), dtype=np.uint16)
    labels = np.zeros((3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="spatial shape"):
        validate_dataset(cube, labels, SPEC)
