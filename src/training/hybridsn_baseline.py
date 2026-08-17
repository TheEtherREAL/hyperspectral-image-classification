"""Reusable training, inference and checkpoint helpers for HybridSN."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.高光谱预处理 import HSITensorDataset, PreprocessingConfig


@dataclass(frozen=True)
class ModelReadyArtifact:
    """Validated bridge between frozen preprocessing and model execution."""

    path: Path
    schema_version: str
    dataset_name: str
    split_protocol: str
    split_seed: int
    config_fingerprint: str
    transformed_cube: np.ndarray
    coordinates: np.ndarray
    raw_labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    class_names: tuple[str, ...]
    patch_size: int
    padding_mode: str
    padding_value: float
    num_classes: int

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.transformed_cube.shape[:2])

    @property
    def output_bands(self) -> int:
        return int(self.transformed_cube.shape[2])


@dataclass(frozen=True)
class InferenceOutput:
    loss: float | None
    accuracy: float | None
    labels: np.ndarray
    predictions: np.ndarray
    sample_indices: np.ndarray
    coordinates: np.ndarray
    elapsed_seconds: float
    throughput_samples_per_second: float
    peak_memory_allocated_bytes: int | None
    peak_memory_reserved_bytes: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_ready_artifact(
    path: Path,
    preprocessing_config: PreprocessingConfig,
) -> ModelReadyArtifact:
    """Load and strictly validate the frozen model-ready NPZ artifact."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema_version",
            "dataset_name",
            "split_protocol",
            "split_seed",
            "config_fingerprint",
            "transformed_cube",
            "coordinates",
            "raw_labels",
            "train_indices",
            "validation_indices",
            "test_indices",
            "class_names",
            "patch_size",
            "padding_mode",
            "padding_value",
            "num_classes",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"model-ready artifact is missing arrays: {sorted(missing)}")
        values = {name: saved[name].copy() for name in required}

    artifact = ModelReadyArtifact(
        path=path,
        schema_version=str(values["schema_version"].item()),
        dataset_name=str(values["dataset_name"].item()),
        split_protocol=str(values["split_protocol"].item()),
        split_seed=int(values["split_seed"].item()),
        config_fingerprint=str(values["config_fingerprint"].item()),
        transformed_cube=np.asarray(values["transformed_cube"]),
        coordinates=np.asarray(values["coordinates"], dtype=np.int32),
        raw_labels=np.asarray(values["raw_labels"], dtype=np.int16),
        train_indices=np.asarray(values["train_indices"], dtype=np.int64),
        validation_indices=np.asarray(values["validation_indices"], dtype=np.int64),
        test_indices=np.asarray(values["test_indices"], dtype=np.int64),
        class_names=tuple(str(value) for value in values["class_names"].tolist()),
        patch_size=int(values["patch_size"].item()),
        padding_mode=str(values["padding_mode"].item()),
        padding_value=float(values["padding_value"].item()),
        num_classes=int(values["num_classes"].item()),
    )
    _validate_model_ready_artifact(artifact, preprocessing_config)
    return artifact


def _validate_model_ready_artifact(
    artifact: ModelReadyArtifact,
    config: PreprocessingConfig,
) -> None:
    if artifact.schema_version != "1.0":
        raise ValueError(f"unsupported model-ready schema: {artifact.schema_version}")
    if artifact.dataset_name != config.dataset_name:
        raise ValueError("model-ready dataset does not match the configuration")
    if artifact.split_protocol != config.split_protocol:
        raise ValueError("model-ready split protocol does not match the configuration")
    if artifact.split_seed != config.split_seed:
        raise ValueError("model-ready split seed does not match the configuration")
    if artifact.config_fingerprint != config.fingerprint():
        raise ValueError("model-ready preprocessing fingerprint does not match")
    if artifact.transformed_cube.ndim != 3:
        raise ValueError("transformed_cube must have shape H x W x bands")
    if artifact.transformed_cube.dtype != np.dtype(config.output_dtype):
        raise ValueError("transformed_cube dtype does not match the configuration")
    if not np.isfinite(artifact.transformed_cube).all():
        raise ValueError("transformed_cube contains NaN or infinite values")
    if config.reducer != "none" and artifact.output_bands != config.n_components:
        raise ValueError("transformed spectral dimension does not match n_components")
    if artifact.patch_size != config.patch_size:
        raise ValueError("model-ready patch size does not match the configuration")
    if artifact.padding_mode != config.padding_mode:
        raise ValueError("model-ready padding mode does not match the configuration")
    if not np.isclose(artifact.padding_value, config.padding_value):
        raise ValueError("model-ready padding value does not match the configuration")
    if artifact.coordinates.shape != (artifact.raw_labels.size, 2):
        raise ValueError("coordinates and raw_labels are not aligned")
    if artifact.num_classes != len(artifact.class_names):
        raise ValueError("num_classes and class_names are inconsistent")
    if np.any(artifact.raw_labels < 1) or np.any(artifact.raw_labels > artifact.num_classes):
        raise ValueError("raw labels must be in 1..num_classes")
    rows, columns = artifact.coordinates.T
    if np.any(rows < 0) or np.any(rows >= artifact.image_shape[0]):
        raise ValueError("model-ready artifact contains an invalid row")
    if np.any(columns < 0) or np.any(columns >= artifact.image_shape[1]):
        raise ValueError("model-ready artifact contains an invalid column")
    assigned = np.concatenate(
        [artifact.train_indices, artifact.validation_indices, artifact.test_indices]
    )
    expected = np.arange(artifact.raw_labels.size, dtype=np.int64)
    if assigned.size != expected.size or not np.array_equal(np.sort(assigned), expected):
        raise ValueError("train/validation/test indices must partition every labeled sample")
    if np.unique(assigned).size != assigned.size:
        raise ValueError("train/validation/test index partitions overlap")


def build_datasets(artifact: ModelReadyArtifact) -> dict[str, HSITensorDataset | None]:
    def make(indices: np.ndarray) -> HSITensorDataset | None:
        if indices.size == 0:
            return None
        return HSITensorDataset(
            artifact.transformed_cube,
            artifact.coordinates,
            artifact.raw_labels,
            indices,
            representation="patch",
            patch_size=artifact.patch_size,
            padding_mode=artifact.padding_mode,
            padding_value=artifact.padding_value,
        )

    return {
        "train": make(artifact.train_indices),
        "validation": make(artifact.validation_indices),
        "test": make(artifact.test_indices),
    }


def build_loaders(
    artifact: ModelReadyArtifact,
    *,
    batch_size: int,
    loader_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[dict[str, DataLoader | None], torch.Generator]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    datasets = build_datasets(artifact)
    generator = torch.Generator().manual_seed(loader_seed)
    loaders: dict[str, DataLoader | None] = {}
    for name, dataset in datasets.items():
        loaders[name] = None if dataset is None else DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return loaders, generator


def build_all_labeled_loader(
    artifact: ModelReadyArtifact,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = HSITensorDataset(
        artifact.transformed_cube,
        artifact.coordinates,
        artifact.raw_labels,
        np.arange(artifact.raw_labels.size, dtype=np.int64),
        representation="patch",
        patch_size=artifact.patch_size,
        padding_mode=artifact.padding_mode,
        padding_value=artifact.padding_value,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, float | int]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    _synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = int(labels.size(0))
        loss_sum += float(loss.detach()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        sample_count += count
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "samples": sample_count,
        "seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
    }


@torch.inference_mode()
def infer_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: nn.Module | None,
    non_blocking: bool,
) -> InferenceOutput:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_sum = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    sample_indices_all: list[np.ndarray] = []
    coordinates_all: list[np.ndarray] = []
    _synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        logits = model(inputs)
        if criterion is not None:
            loss_sum += float(criterion(logits, labels)) * int(labels.size(0))
        labels_all.append(labels.cpu().numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
        sample_indices_all.append(batch["sample_index"].numpy())
        coordinates_all.append(batch["coordinate"].numpy())
    _synchronize(device)
    elapsed = time.perf_counter() - started
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    sample_indices = np.concatenate(sample_indices_all)
    coordinates = np.concatenate(coordinates_all)
    accuracy = float(np.mean(labels == predictions)) if criterion is not None else None
    return InferenceOutput(
        loss=loss_sum / labels.size if criterion is not None else None,
        accuracy=accuracy,
        labels=labels,
        predictions=predictions,
        sample_indices=sample_indices,
        coordinates=coordinates,
        elapsed_seconds=elapsed,
        throughput_samples_per_second=labels.size / elapsed,
        peak_memory_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        peak_memory_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
    )


@torch.inference_mode()
def benchmark_model_compute(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    non_blocking: bool,
    warmup_iterations: int = 5,
    measured_iterations: int = 20,
) -> dict[str, float | int]:
    if warmup_iterations < 0 or measured_iterations < 1:
        raise ValueError("benchmark iteration counts are invalid")
    model.eval()
    batch = next(iter(loader))
    inputs = batch["input"].to(device, non_blocking=non_blocking)
    for _ in range(warmup_iterations):
        model(inputs)
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(measured_iterations):
        model(inputs)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    samples = int(inputs.size(0)) * measured_iterations
    return {
        "batch_size": int(inputs.size(0)),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "elapsed_seconds": elapsed,
        "milliseconds_per_batch": elapsed * 1000.0 / measured_iterations,
        "milliseconds_per_sample": elapsed * 1000.0 / samples,
        "throughput_samples_per_second": samples / elapsed,
    }


def capture_rng_state(data_loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": data_loader_generator.get_state(),
    }


def restore_rng_state(state: Mapping[str, Any], data_loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    data_loader_generator.set_state(state["data_loader_generator"])


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    effective_config: Mapping[str, Any],
    preprocessing_fingerprint: str,
    history: list[dict[str, Any]],
    data_loader_generator: torch.Generator,
    selected_by: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "1.0",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "effective_config": dict(effective_config),
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "history": history,
            "rng_state": capture_rng_state(data_loader_generator),
            "selected_by": selected_by,
            "test_set_evaluated": False,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    data_loader_generator: torch.Generator | None,
    expected_preprocessing_fingerprint: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("preprocessing_fingerprint") != expected_preprocessing_fingerprint:
        raise ValueError("checkpoint preprocessing fingerprint does not match the data")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if data_loader_generator is not None and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"], data_loader_generator)
    return checkpoint


def write_json(path: Path, values: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
    Path(path).write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "InferenceOutput",
    "ModelReadyArtifact",
    "benchmark_model_compute",
    "build_all_labeled_loader",
    "build_datasets",
    "build_loaders",
    "infer_loader",
    "load_checkpoint",
    "load_model_ready_artifact",
    "save_checkpoint",
    "sha256_file",
    "train_one_epoch",
    "write_json",
]
