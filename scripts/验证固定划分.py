"""独立验证 Pavia University 固定划分 / Verify frozen splits."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.datasets.固定划分 import SPLIT_IDS, create_fixed_protocol_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DATASET_NAME = "pavia_university"
SEED = 345
PROTOCOLS = ("paper30", "fair24_6_70")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(protocol_name: str, label_map: np.ndarray) -> dict[str, np.ndarray]:
    stem = f"{DATASET_NAME}__{protocol_name}__seed{SEED}"
    npz_path = SPLIT_DIR / f"{stem}.npz"
    metadata_path = SPLIT_DIR / f"{stem}.json"
    statistics_path = SPLIT_DIR / f"{stem}__stats.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["protocol"]["name"] != protocol_name or metadata["protocol"]["seed"] != SEED:
        raise AssertionError(f"{protocol_name}: protocol metadata mismatch")
    for file_name, file_metadata in metadata["dataset"]["source_files"].items():
        if sha256(RAW_DIR / file_name) != file_metadata["sha256"]:
            raise AssertionError(f"{protocol_name}: source hash mismatch for {file_name}")

    with np.load(npz_path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}

    coordinates = arrays["coordinates"]
    labels = arrays["labels"]
    rows, columns = coordinates.T
    if coordinates.shape != (np.count_nonzero(label_map), 2):
        raise AssertionError(f"{protocol_name}: labeled coordinate coverage mismatch")
    np.testing.assert_array_equal(labels, label_map[rows, columns])
    np.testing.assert_array_equal(np.unique(labels), np.arange(1, 10))

    partition_indices = {
        "train": arrays["train_indices"],
        "validation": arrays["validation_indices"],
        "test": arrays["test_indices"],
    }
    all_indices = np.concatenate(list(partition_indices.values()))
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(labels.size))
    if np.unique(all_indices).size != labels.size:
        raise AssertionError(f"{protocol_name}: partitions overlap")

    expected_split_ids = np.full(labels.size, 255, dtype=np.uint8)
    counts: dict[str, int] = {}
    for split_name, indices in partition_indices.items():
        counts[split_name] = int(indices.size)
        expected_split_ids[indices] = SPLIT_IDS[split_name]
        if indices.size:
            np.testing.assert_array_equal(np.unique(labels[indices]), np.arange(1, 10))
        np.testing.assert_array_equal(
            arrays[f"{split_name}_coordinates"],
            coordinates[indices],
        )
        np.testing.assert_array_equal(arrays[f"{split_name}_labels"], labels[indices])
    np.testing.assert_array_equal(arrays["split_ids"], expected_split_ids)

    with statistics_path.open("r", newline="", encoding="utf-8-sig") as stream:
        statistics = list(csv.DictReader(stream))
    overall_rows = {row["split"]: row for row in statistics if row["class_id"] == "ALL"}
    for split_name, count in counts.items():
        if int(overall_rows[split_name]["samples"]) != count:
            raise AssertionError(f"{protocol_name}: CSV count mismatch for {split_name}")
        if metadata["statistics"]["overall"][split_name]["samples"] != count:
            raise AssertionError(f"{protocol_name}: JSON count mismatch for {split_name}")

    rebuilt = create_fixed_protocol_splits(label_map, seed=SEED)[protocol_name]
    np.testing.assert_array_equal(rebuilt.coordinates, coordinates)
    np.testing.assert_array_equal(rebuilt.labels, labels)
    np.testing.assert_array_equal(rebuilt.split_ids(), arrays["split_ids"])

    print(
        f"protocol={protocol_name} counts={counts} "
        f"npz_sha256={sha256(npz_path)} verification=passed"
    )
    return partition_indices


def main() -> None:
    label_map = np.asarray(loadmat(RAW_DIR / "PaviaU_gt.mat")["paviaU_gt"])
    verified = {name: verify_protocol(name, label_map) for name in PROTOCOLS}

    np.testing.assert_array_equal(
        verified["paper30"]["test"],
        verified["fair24_6_70"]["test"],
    )
    np.testing.assert_array_equal(
        verified["paper30"]["train"],
        np.sort(
            np.concatenate(
                (
                    verified["fair24_6_70"]["train"],
                    verified["fair24_6_70"]["validation"],
                )
            )
        ),
    )
    print("cross_protocol_relationship=passed")


if __name__ == "__main__":
    main()
