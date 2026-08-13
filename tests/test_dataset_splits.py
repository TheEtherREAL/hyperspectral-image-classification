import json
from dataclasses import replace

import numpy as np
import pytest

from src.datasets.固定划分 import (
    create_fixed_protocol_splits,
    validate_pixel_split,
    write_split_artifacts,
)


CLASS_NAMES = ("one", "two", "three", "four")


@pytest.fixture
def label_map() -> np.ndarray:
    background = np.zeros(20, dtype=np.uint8)
    labeled = np.repeat(np.arange(1, 5, dtype=np.uint8), 100)
    return np.concatenate((background, labeled)).reshape(21, 20)


def test_partitions_are_disjoint_and_cover_all_labeled_pixels(label_map: np.ndarray) -> None:
    splits = create_fixed_protocol_splits(label_map, seed=345)
    labeled_count = int(np.count_nonzero(label_map))

    for split in splits.values():
        partitions = split.indices_by_split()
        index_sets = {name: set(indices.tolist()) for name, indices in partitions.items()}

        assert index_sets["train"].isdisjoint(index_sets["validation"])
        assert index_sets["train"].isdisjoint(index_sets["test"])
        assert index_sets["validation"].isdisjoint(index_sets["test"])
        assert set().union(*index_sets.values()) == set(range(labeled_count))


def test_every_nonempty_partition_contains_every_class(label_map: np.ndarray) -> None:
    expected_classes = {1, 2, 3, 4}
    for split in create_fixed_protocol_splits(label_map, seed=345).values():
        for indices in split.indices_by_split().values():
            if indices.size:
                assert set(np.unique(split.labels[indices]).tolist()) == expected_classes


def test_same_seed_reproduces_every_assignment(label_map: np.ndarray) -> None:
    first = create_fixed_protocol_splits(label_map, seed=345)
    second = create_fixed_protocol_splits(label_map, seed=345)

    for protocol_name in first:
        np.testing.assert_array_equal(first[protocol_name].coordinates, second[protocol_name].coordinates)
        np.testing.assert_array_equal(first[protocol_name].labels, second[protocol_name].labels)
        np.testing.assert_array_equal(first[protocol_name].split_ids(), second[protocol_name].split_ids())


def test_default_seed_is_the_frozen_seed_345(label_map: np.ndarray) -> None:
    default = create_fixed_protocol_splits(label_map)
    explicit = create_fixed_protocol_splits(label_map, seed=345)

    for protocol_name in default:
        assert default[protocol_name].protocol.seed == 345
        np.testing.assert_array_equal(
            default[protocol_name].split_ids(), explicit[protocol_name].split_ids()
        )


def test_protocols_share_test_set_and_training_pool(label_map: np.ndarray) -> None:
    splits = create_fixed_protocol_splits(label_map, seed=345)
    paper = splits["paper30"]
    fair = splits["fair24_6_70"]

    np.testing.assert_array_equal(paper.test_indices, fair.test_indices)
    np.testing.assert_array_equal(
        paper.train_indices,
        np.sort(np.concatenate((fair.train_indices, fair.validation_indices))),
    )


def test_coordinates_and_original_labels_remain_aligned(label_map: np.ndarray) -> None:
    split = create_fixed_protocol_splits(label_map, seed=345)["fair24_6_70"]
    rows, columns = split.coordinates.T

    assert split.coordinates.dtype == np.int32
    assert split.labels.dtype == np.int16
    np.testing.assert_array_equal(split.labels, label_map[rows, columns])
    assert np.all(split.labels > 0)


def test_validator_rejects_overlapping_incomplete_partitions(label_map: np.ndarray) -> None:
    fair = create_fixed_protocol_splits(label_map, seed=345)["fair24_6_70"]
    invalid = replace(fair, validation_indices=fair.train_indices)

    with pytest.raises(ValueError, match="disjoint and cover"):
        validate_pixel_split(invalid)


def test_artifacts_save_assignments_statistics_and_metadata(
    label_map: np.ndarray,
    tmp_path,
) -> None:
    split = create_fixed_protocol_splits(label_map, seed=345)["fair24_6_70"]
    paths = write_split_artifacts(
        split,
        output_dir=tmp_path,
        dataset_name="toy",
        class_names=CLASS_NAMES,
        metadata={"dataset": {"source_files": {"toy_gt.mat": {"sha256": "abc"}}}},
    )

    with np.load(paths["npz"], allow_pickle=False) as artifact:
        expected_keys = {
            "coordinates",
            "labels",
            "split_ids",
            "train_indices",
            "validation_indices",
            "test_indices",
            "train_coordinates",
            "train_labels",
            "validation_coordinates",
            "validation_labels",
            "test_coordinates",
            "test_labels",
        }
        assert expected_keys.issubset(artifact.files)
        np.testing.assert_array_equal(
            artifact["train_coordinates"],
            artifact["coordinates"][artifact["train_indices"]],
        )
        np.testing.assert_array_equal(
            artifact["train_labels"],
            artifact["labels"][artifact["train_indices"]],
        )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1.0"
    assert metadata["protocol"]["seed"] == 345
    assert metadata["statistics"]["overall"]["train"]["samples"] == 96

    statistics_lines = paths["statistics"].read_text(encoding="utf-8-sig").splitlines()
    assert len(statistics_lines) == 1 + 3 * (1 + len(CLASS_NAMES))
