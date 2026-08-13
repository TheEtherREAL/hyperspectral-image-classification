"""安全读取和验证高光谱 MATLAB 数据 / Safe HSI MATLAB loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .数据集注册 import DatasetSpec


def public_mat_keys(path: Path) -> list[str]:
    """List user variables in a MATLAB v5 file."""
    return sorted(key for key in loadmat(path).keys() if not key.startswith("__"))


def load_dataset(raw_dir: Path, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load a cube and label map using registry-defined variable names."""
    data_path = raw_dir / spec.data_file
    label_path = raw_dir / spec.label_file
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    data_values = loadmat(data_path)
    label_values = loadmat(label_path)
    if spec.data_key not in data_values:
        raise KeyError(f"{spec.data_key!r} not found in {data_path.name}")
    if spec.label_key not in label_values:
        raise KeyError(f"{spec.label_key!r} not found in {label_path.name}")

    cube = np.asarray(data_values[spec.data_key])
    labels = np.asarray(label_values[spec.label_key])
    validate_dataset(cube, labels, spec)
    return cube, labels


def validate_dataset(cube: np.ndarray, labels: np.ndarray, spec: DatasetSpec) -> None:
    """Validate basic geometry and label invariants before preprocessing."""
    if cube.ndim != 3:
        raise ValueError(f"{spec.name}: expected H×W×B cube, got {cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"{spec.name}: expected H×W labels, got {labels.shape}")
    if cube.shape[:2] != labels.shape:
        raise ValueError(
            f"{spec.name}: cube spatial shape {cube.shape[:2]} != label shape {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{spec.name}: labels must be integer, got {labels.dtype}")

    unique = np.unique(labels)
    if unique.min() != 0:
        raise ValueError(f"{spec.name}: background label 0 is missing")
    expected = np.arange(len(spec.class_names) + 1)
    if not np.array_equal(unique, expected):
        raise ValueError(f"{spec.name}: labels {unique.tolist()} != {expected.tolist()}")
