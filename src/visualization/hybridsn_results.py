"""Deterministic result figures for the HybridSN baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


HSI_CLASS_COLORS = (
    "#000000",  # background / not evaluated
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#6a3d9a",
)


def save_learning_curves(
    history: Sequence[Mapping[str, float]],
    path: Path,
    *,
    loss_title: str = "Training objective loss",
) -> None:
    if not history:
        raise ValueError("history must contain at least one epoch")
    epochs = [int(record["epoch"]) for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    axes[0].plot(epochs, [record["train_loss"] for record in history], label="Train", lw=1.8)
    axes[1].plot(
        epochs,
        [record["train_accuracy"] for record in history],
        label="Train",
        lw=1.8,
    )
    if "validation_loss" in history[0]:
        axes[0].plot(
            epochs,
            [record["validation_loss"] for record in history],
            label="Validation",
            lw=1.8,
        )
        axes[1].plot(
            epochs,
            [record["validation_accuracy"] for record in history],
            label="Validation",
            lw=1.8,
        )
    axes[0].set(title=loss_title, xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0.0, 1.01)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("matrix shape does not match class_names")
    figure, axis = plt.subplots(figsize=(8.2, 7.2), dpi=180)
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_per_class_accuracy(
    accuracies: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    accuracies = np.asarray(accuracies, dtype=np.float64)
    if accuracies.shape != (len(class_names),):
        raise ValueError("accuracies shape does not match class_names")
    figure, axis = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    bars = axis.bar(np.arange(len(class_names)), accuracies * 100.0, color="#2878B5")
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=35, ha="right")
    axis.set_ylim(0, 105)
    axis.set(ylabel="Accuracy (%)", title="Per-class test accuracy")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, accuracies, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value * 100.0 + 1.2, 102.0),
            f"{value * 100.0:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_classification_maps(
    ground_truth: np.ndarray,
    test_prediction_map: np.ndarray,
    all_labeled_prediction_map: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    *,
    title: str = "Hyperspectral image classification",
) -> None:
    maps = [
        np.asarray(ground_truth),
        np.asarray(test_prediction_map),
        np.asarray(all_labeled_prediction_map),
    ]
    if not all(array.shape == maps[0].shape for array in maps):
        raise ValueError("all classification maps must share the same shape")
    if len(class_names) + 1 <= len(HSI_CLASS_COLORS):
        colors = HSI_CLASS_COLORS[: len(class_names) + 1]
    else:
        colors = ("#000000",) + tuple(
            plt.get_cmap("tab20")(index) for index in range(len(class_names))
        )
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(class_names) + 1.5), cmap.N)
    titles = (
        "Ground truth (labeled pixels)",
        "Test predictions only",
        "Predictions for all labeled pixels",
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 7.2), dpi=180)
    image = None
    for axis, data, title in zip(axes, maps, titles, strict=True):
        image = axis.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    colorbar = figure.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        ticks=np.arange(len(class_names) + 1),
    )
    colorbar.ax.set_yticklabels(["Background / hidden", *class_names], fontsize=8)
    figure.suptitle(title, fontsize=14)
    figure.subplots_adjust(left=0.01, right=0.86, top=0.91, bottom=0.02, wspace=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "save_classification_maps",
    "save_confusion_matrix",
    "save_learning_curves",
    "save_per_class_accuracy",
]
