"""Classification metrics shared by HSI baselines and later model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClassificationSummary:
    """Complete fixed-class classification summary.

    Labels are zero based at the model boundary.  The confusion matrix keeps
    every configured class, including classes with no predictions.
    """

    overall_accuracy: float
    average_accuracy: float
    kappa: float
    per_class_accuracy: np.ndarray
    support: np.ndarray
    confusion_matrix: np.ndarray

    def to_dict(self, class_names: Sequence[str]) -> dict[str, object]:
        if len(class_names) != self.per_class_accuracy.size:
            raise ValueError("class_names length does not match the metric summary")
        return {
            "oa": self.overall_accuracy,
            "aa": self.average_accuracy,
            "kappa": self.kappa,
            "per_class": [
                {
                    "class_index": index,
                    "raw_label": index + 1,
                    "class_name": str(name),
                    "support": int(self.support[index]),
                    "accuracy": float(self.per_class_accuracy[index]),
                }
                for index, name in enumerate(class_names)
            ],
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def classification_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
) -> ClassificationSummary:
    """Compute OA, AA, Cohen's kappa, per-class accuracy and confusion matrix."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if labels.size == 0:
        raise ValueError("at least one labeled sample is required")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    for name, values in (("labels", labels), ("predictions", predictions)):
        if np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError(f"{name} must be in the range 0..num_classes-1")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support != 0,
    )
    total = int(matrix.sum())
    observed = float(np.trace(matrix) / total)
    expected = float(np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / (total * total))
    kappa = 0.0 if np.isclose(1.0 - expected, 0.0) else (observed - expected) / (1.0 - expected)
    return ClassificationSummary(
        overall_accuracy=observed,
        average_accuracy=float(per_class.mean()),
        kappa=float(kappa),
        per_class_accuracy=per_class,
        support=support,
        confusion_matrix=matrix,
    )


def build_classification_map(
    image_shape: tuple[int, int],
    coordinates: np.ndarray,
    sample_indices: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Place zero-based predictions into a raw-label map (background remains 0)."""

    coordinates = np.asarray(coordinates, dtype=np.int64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape N x 2")
    if sample_indices.size != predictions.size:
        raise ValueError("sample_indices and predictions must be aligned")
    if np.any(sample_indices < 0) or np.any(sample_indices >= coordinates.shape[0]):
        raise ValueError("sample_indices contain an out-of-range value")
    selected = coordinates[sample_indices]
    if np.any(selected[:, 0] < 0) or np.any(selected[:, 0] >= image_shape[0]):
        raise ValueError("row coordinate is outside image_shape")
    if np.any(selected[:, 1] < 0) or np.any(selected[:, 1] >= image_shape[1]):
        raise ValueError("column coordinate is outside image_shape")

    output = np.zeros(image_shape, dtype=np.int16)
    output[selected[:, 0], selected[:, 1]] = predictions.astype(np.int16) + 1
    return output


def _window_counts(
    image_shape: tuple[int, int],
    center_coordinates: np.ndarray,
    query_coordinates: np.ndarray,
    radius: int,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.int32)
    if center_coordinates.size:
        np.add.at(mask, (center_coordinates[:, 0], center_coordinates[:, 1]), 1)
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    rows, columns = query_coordinates.T
    row_start = np.maximum(rows - radius, 0)
    column_start = np.maximum(columns - radius, 0)
    row_stop = np.minimum(rows + radius + 1, image_shape[0])
    column_stop = np.minimum(columns + radius + 1, image_shape[1])
    return (
        integral[row_stop, column_stop]
        - integral[row_start, column_stop]
        - integral[row_stop, column_start]
        + integral[row_start, column_start]
    )


def spatial_overlap_audit(
    image_shape: tuple[int, int],
    coordinates: np.ndarray,
    raw_labels: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    patch_size: int,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Quantify spatial dependence induced by random pixel splits and patches.

    This does not claim label leakage in the preprocessing implementation.  It
    records how often a query patch spatially contains training center pixels,
    which is essential context when interpreting very high random-split scores.
    """

    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    coordinates = np.asarray(coordinates, dtype=np.int64)
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    if coordinates.shape != (raw_labels.size, 2):
        raise ValueError("coordinates and raw_labels are not aligned")
    if not train_indices.size or not query_indices.size:
        raise ValueError("train and query partitions must both be non-empty")
    if len(class_names) < int(raw_labels.max()):
        raise ValueError("class_names does not cover every raw label")

    radius = patch_size // 2
    train_coordinates = coordinates[train_indices]
    query_coordinates = coordinates[query_indices]
    query_labels = raw_labels[query_indices]
    any_counts = _window_counts(image_shape, train_coordinates, query_coordinates, radius)
    same_class_counts = np.zeros(query_indices.size, dtype=np.int64)
    per_class: list[dict[str, object]] = []
    for raw_label, class_name in enumerate(class_names, start=1):
        train_for_class = train_coordinates[raw_labels[train_indices] == raw_label]
        query_mask = query_labels == raw_label
        class_counts = _window_counts(
            image_shape,
            train_for_class,
            query_coordinates[query_mask],
            radius,
        )
        same_class_counts[query_mask] = class_counts
        per_class.append(
            {
                "raw_label": raw_label,
                "class_name": str(class_name),
                "query_samples": int(query_mask.sum()),
                "with_same_class_train_center": int(np.count_nonzero(class_counts)),
                "fraction_with_same_class_train_center": float(np.mean(class_counts > 0)),
                "mean_same_class_train_centers_in_patch": float(np.mean(class_counts)),
            }
        )

    def summarize(counts: np.ndarray) -> dict[str, object]:
        return {
            "query_samples": int(counts.size),
            "with_at_least_one": int(np.count_nonzero(counts)),
            "fraction_with_at_least_one": float(np.mean(counts > 0)),
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "maximum": int(counts.max()),
        }

    return {
        "interpretation": (
            "Spatial context overlap under the fixed random-pixel split; this is "
            "not a claim that test labels entered preprocessing or model selection."
        ),
        "patch_size": patch_size,
        "patch_radius": radius,
        "any_training_center_in_query_patch": summarize(any_counts),
        "same_class_training_center_in_query_patch": summarize(same_class_counts),
        "per_class": per_class,
    }


__all__ = [
    "ClassificationSummary",
    "build_classification_map",
    "classification_summary",
    "spatial_overlap_audit",
]
