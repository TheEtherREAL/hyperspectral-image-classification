"""高光谱像元的确定性分层划分 / Deterministic stratified splits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


SPLIT_SCHEMA_VERSION = "1.0"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_IDS = {name: split_id for split_id, name in enumerate(SPLIT_NAMES)}


@dataclass(frozen=True)
class SplitProtocol:
    """Human- and machine-readable definition of one split protocol."""

    name: str
    purpose: str
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float

    def target_fractions(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }


@dataclass(frozen=True)
class PixelSplit:
    """A complete assignment of every labeled pixel to one partition."""

    protocol: SplitProtocol
    coordinates: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def indices_by_split(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "test": self.test_indices,
        }

    def split_ids(self) -> np.ndarray:
        output = np.full(self.labels.shape[0], fill_value=255, dtype=np.uint8)
        for split_name, indices in self.indices_by_split().items():
            output[indices] = SPLIT_IDS[split_name]
        return output


def labeled_pixels(
    label_map: np.ndarray,
    *,
    background_label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based ``(row, column)`` coordinates and original labels."""
    label_map = np.asarray(label_map)
    if label_map.ndim != 2:
        raise ValueError(f"expected a 2D label map, got shape {label_map.shape}")
    if not np.issubdtype(label_map.dtype, np.integer):
        raise TypeError(f"labels must be integer, got {label_map.dtype}")

    coordinates = np.argwhere(label_map != background_label).astype(np.int32, copy=False)
    labels = label_map[coordinates[:, 0], coordinates[:, 1]].astype(np.int16, copy=False)
    if labels.size == 0:
        raise ValueError("label map has no labeled pixels")
    return coordinates, labels


def create_fixed_protocol_splits(
    label_map: np.ndarray,
    *,
    seed: int = 1442,
    background_label: int = 0,
) -> dict[str, PixelSplit]:
    """Create the paper-compatible and fair-comparison Pavia protocols.

    Both protocols share the same stratified 70% test set. The remaining
    30% pool is the complete training set for ``paper30`` and is split 80/20
    into train/validation partitions for ``fair24_6_70``.
    """
    coordinates, labels = labeled_pixels(label_map, background_label=background_label)
    all_indices = np.arange(labels.size, dtype=np.int64)

    paper_train, shared_test = train_test_split(
        all_indices,
        train_size=0.30,
        test_size=0.70,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    fair_train, fair_validation = train_test_split(
        paper_train,
        train_size=0.80,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=labels[paper_train],
    )

    empty = np.empty(0, dtype=np.int64)
    paper = PixelSplit(
        protocol=SplitProtocol(
            name="paper30",
            purpose="HybridSN paper-compatible 30% train / 70% test reproduction",
            seed=seed,
            train_fraction=0.30,
            validation_fraction=0.0,
            test_fraction=0.70,
        ),
        coordinates=coordinates,
        labels=labels,
        train_indices=np.sort(paper_train.astype(np.int64, copy=False)),
        validation_indices=empty,
        test_indices=np.sort(shared_test.astype(np.int64, copy=False)),
    )
    fair = PixelSplit(
        protocol=SplitProtocol(
            name="fair24_6_70",
            purpose="24% train / 6% validation / 70% test fair model comparison",
            seed=seed,
            train_fraction=0.24,
            validation_fraction=0.06,
            test_fraction=0.70,
        ),
        coordinates=coordinates,
        labels=labels,
        train_indices=np.sort(fair_train.astype(np.int64, copy=False)),
        validation_indices=np.sort(fair_validation.astype(np.int64, copy=False)),
        test_indices=np.sort(shared_test.astype(np.int64, copy=False)),
    )

    for split in (paper, fair):
        validate_pixel_split(split)
    if not np.array_equal(paper.test_indices, fair.test_indices):
        raise AssertionError("the two protocols must share an identical test set")
    if not np.array_equal(
        paper.train_indices,
        np.sort(np.concatenate((fair.train_indices, fair.validation_indices))),
    ):
        raise AssertionError("fair train+validation must equal the paper30 training pool")

    return {paper.protocol.name: paper, fair.protocol.name: fair}


def validate_pixel_split(split: PixelSplit) -> None:
    """Validate disjointness, full coverage, coordinates and class coverage."""
    coordinates = np.asarray(split.coordinates)
    labels = np.asarray(split.labels)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape N×2, got {coordinates.shape}")
    if labels.ndim != 1 or labels.shape[0] != coordinates.shape[0]:
        raise ValueError("labels must be a length-N vector aligned with coordinates")
    if np.any(labels <= 0):
        raise ValueError("background or negative labels must not enter a split")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise ValueError("labeled pixel coordinates must be unique")

    expected_indices = np.arange(labels.size, dtype=np.int64)
    partition_arrays: list[np.ndarray] = []
    expected_classes = np.unique(labels)
    for split_name, indices in split.indices_by_split().items():
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"{split_name} indices must be a 1D integer array")
        if np.any(indices < 0) or np.any(indices >= labels.size):
            raise ValueError(f"{split_name} contains out-of-range indices")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{split_name} contains duplicate indices")
        if indices.size and not np.array_equal(np.unique(labels[indices]), expected_classes):
            raise ValueError(f"{split_name} does not contain every class")
        partition_arrays.append(indices.astype(np.int64, copy=False))

    assigned = np.concatenate(partition_arrays)
    if assigned.size != labels.size or not np.array_equal(np.sort(assigned), expected_indices):
        raise ValueError("partitions must be disjoint and cover every labeled pixel exactly once")
    if np.any(split.split_ids() == 255):
        raise ValueError("one or more labeled pixels have no split assignment")


def split_statistics(
    split: PixelSplit,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Create overall and per-class statistics for a split artifact."""
    class_ids = np.unique(split.labels)
    expected_class_ids = np.arange(1, len(class_names) + 1)
    if not np.array_equal(class_ids, expected_class_ids):
        raise ValueError(
            f"split labels {class_ids.tolist()} do not match class names 1..{len(class_names)}"
        )

    rows: list[dict[str, Any]] = []
    target_fractions = split.protocol.target_fractions()
    for split_name, indices in split.indices_by_split().items():
        rows.append(
            {
                "protocol": split.protocol.name,
                "seed": split.protocol.seed,
                "split": split_name,
                "class_id": "ALL",
                "class_name": "All labeled pixels",
                "samples": int(indices.size),
                "class_total": int(split.labels.size),
                "target_fraction": target_fractions[split_name],
                "actual_fraction_of_class": float(indices.size / split.labels.size),
            }
        )
        partition_labels = split.labels[indices]
        for class_id, class_name in enumerate(class_names, start=1):
            class_total = int(np.count_nonzero(split.labels == class_id))
            class_samples = int(np.count_nonzero(partition_labels == class_id))
            rows.append(
                {
                    "protocol": split.protocol.name,
                    "seed": split.protocol.seed,
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "samples": class_samples,
                    "class_total": class_total,
                    "target_fraction": target_fractions[split_name],
                    "actual_fraction_of_class": float(class_samples / class_total),
                }
            )
    return rows


def statistics_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert tabular split statistics to a compact JSON structure."""
    overall: dict[str, Any] = {}
    by_class: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["class_id"] == "ALL":
            overall[str(row["split"])] = {
                "samples": int(row["samples"]),
                "target_fraction": float(row["target_fraction"]),
                "actual_fraction": float(row["actual_fraction_of_class"]),
            }
            continue
        class_key = str(row["class_id"])
        class_entry = by_class.setdefault(
            class_key,
            {
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "total": int(row["class_total"]),
                "splits": {},
            },
        )
        class_entry["splits"][str(row["split"])] = {
            "samples": int(row["samples"]),
            "actual_fraction": float(row["actual_fraction_of_class"]),
        }
    return {"overall": overall, "by_class": list(by_class.values())}


def write_split_artifacts(
    split: PixelSplit,
    *,
    output_dir: Path,
    dataset_name: str,
    class_names: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write NPZ assignments, JSON metadata and CSV statistics."""
    validate_pixel_split(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}__{split.protocol.name}__seed{split.protocol.seed}"
    paths = {
        "npz": output_dir / f"{stem}.npz",
        "metadata": output_dir / f"{stem}.json",
        "statistics": output_dir / f"{stem}__stats.csv",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing split artifacts: {existing}")

    indices_by_split = split.indices_by_split()
    np.savez_compressed(
        paths["npz"],
        schema_version=np.asarray(SPLIT_SCHEMA_VERSION),
        dataset_name=np.asarray(dataset_name),
        protocol_name=np.asarray(split.protocol.name),
        seed=np.asarray(split.protocol.seed, dtype=np.int64),
        coordinates=split.coordinates.astype(np.int32, copy=False),
        labels=split.labels.astype(np.int16, copy=False),
        split_ids=split.split_ids(),
        split_id_names=np.asarray(SPLIT_NAMES),
        train_indices=split.train_indices,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        train_coordinates=split.coordinates[split.train_indices],
        train_labels=split.labels[split.train_indices],
        validation_coordinates=split.coordinates[split.validation_indices],
        validation_labels=split.labels[split.validation_indices],
        test_coordinates=split.coordinates[split.test_indices],
        test_labels=split.labels[split.test_indices],
    )

    rows = split_statistics(split, class_names)
    with paths["statistics"].open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "protocol": {
                "name": split.protocol.name,
                "purpose": split.protocol.purpose,
                "seed": split.protocol.seed,
                "target_fractions": split.protocol.target_fractions(),
            },
            "artifacts": {name: path.name for name, path in paths.items()},
            "statistics": statistics_summary(rows),
        }
    )
    paths["metadata"].write_text(
        json.dumps(complete_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths
