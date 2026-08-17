"""Tests for reusable classification metrics and spatial prediction maps."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)


def test_classification_summary_keeps_all_classes_and_matches_hand_calculation() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 1], dtype=np.int64)

    summary = classification_summary(labels, predictions, num_classes=3)

    np.testing.assert_array_equal(
        summary.confusion_matrix,
        np.asarray([[1, 1, 0], [0, 2, 0], [0, 0, 0]]),
    )
    np.testing.assert_array_equal(summary.support, np.asarray([2, 2, 0]))
    np.testing.assert_allclose(summary.per_class_accuracy, [0.5, 1.0, 0.0])
    assert summary.overall_accuracy == pytest.approx(0.75)
    assert summary.average_accuracy == pytest.approx(0.5)
    assert summary.kappa == pytest.approx(0.5)


def test_summary_dictionary_preserves_label_contract() -> None:
    summary = classification_summary(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        num_classes=2,
    )

    values = summary.to_dict(("class-a", "class-b"))

    assert values["per_class"][0] == {
        "class_index": 0,
        "raw_label": 1,
        "class_name": "class-a",
        "support": 1,
        "accuracy": 1.0,
    }
    assert values["confusion_matrix"] == [[1, 0], [0, 1]]


def test_build_classification_map_uses_sample_identity_not_loader_order() -> None:
    coordinates = np.asarray([[0, 1], [2, 2], [1, 0]], dtype=np.int32)

    output = build_classification_map(
        (3, 3),
        coordinates,
        sample_indices=np.asarray([2, 0]),
        predictions=np.asarray([1, 0]),
    )

    expected = np.zeros((3, 3), dtype=np.int16)
    expected[1, 0] = 2
    expected[0, 1] = 1
    np.testing.assert_array_equal(output, expected)


def test_spatial_overlap_audit_counts_any_and_same_class_centers() -> None:
    coordinates = np.asarray([[1, 1], [1, 3], [3, 1], [3, 3]], dtype=np.int32)
    raw_labels = np.asarray([1, 1, 2, 2], dtype=np.int16)

    audit = spatial_overlap_audit(
        (5, 5),
        coordinates,
        raw_labels,
        train_indices=np.asarray([0, 2]),
        query_indices=np.asarray([1, 3]),
        patch_size=3,
        class_names=("one", "two"),
    )

    assert audit["any_training_center_in_query_patch"]["fraction_with_at_least_one"] == 0.0
    assert audit["same_class_training_center_in_query_patch"]["fraction_with_at_least_one"] == 0.0

    wider = spatial_overlap_audit(
        (5, 5),
        coordinates,
        raw_labels,
        train_indices=np.asarray([0, 2]),
        query_indices=np.asarray([1, 3]),
        patch_size=5,
        class_names=("one", "two"),
    )
    assert wider["any_training_center_in_query_patch"]["fraction_with_at_least_one"] == 1.0
    assert wider["same_class_training_center_in_query_patch"]["fraction_with_at_least_one"] == 1.0


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([], []),
        ([0, 1], [0]),
        ([0, 2], [0, 1]),
        ([0, 1], [0, -1]),
    ],
)
def test_classification_summary_rejects_invalid_inputs(labels, predictions) -> None:
    with pytest.raises(ValueError):
        classification_summary(
            np.asarray(labels),
            np.asarray(predictions),
            num_classes=2,
        )
