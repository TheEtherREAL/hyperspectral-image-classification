"""Consolidate executed coursework runs into report-ready CSV and PNG figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _box(axis: Any, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#234E8A",
        linewidth=1.4,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def save_architecture_diagram(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.5), dpi=180)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.16, 0.94, "Dimensionality reduction", ha="center", fontsize=14)
    axis.text(0.49, 0.94, "Spatial feature extraction / fusion", ha="center", fontsize=14)
    axis.text(0.83, 0.94, "Classifier", ha="center", fontsize=14)
    _box(axis, 0.015, 0.43, 0.13, 0.12, "HSI cube\nH×W×B", "#DCEBFF")
    for y, text in zip((0.72, 0.56, 0.40, 0.24), ("Raw bands", "PCA / LDA", "Uniform bands", "Fisher bands"), strict=True):
        _box(axis, 0.20, y, 0.17, 0.10, text, "#BBD4FF")
    _box(axis, 0.46, 0.62, 0.18, 0.14, "Traditional\nLBP + Gabor", "#BDE7D0")
    _box(axis, 0.46, 0.30, 0.18, 0.14, "Deep learning\n3D CNN → 2D CNN", "#FFD6A5")
    _box(axis, 0.76, 0.66, 0.17, 0.11, "SVM / XGBoost", "#D8C7FF")
    _box(axis, 0.76, 0.30, 0.17, 0.11, "Softmax / Sigmoid", "#D8C7FF")
    arrows = (
        ((0.145, 0.49), (0.20, 0.49)),
        ((0.37, 0.61), (0.46, 0.69)),
        ((0.37, 0.45), (0.46, 0.69)),
        ((0.37, 0.45), (0.46, 0.37)),
        ((0.64, 0.69), (0.76, 0.715)),
        ((0.64, 0.37), (0.76, 0.355)),
    )
    for start, stop in arrows:
        axis.add_patch(
            FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=16, color="#3567B7", lw=2)
        )
    axis.text(
        0.5,
        0.08,
        "Frozen protocol: stratified 24% train / 6% validation / 70% test; seed = 1442",
        ha="center",
        fontsize=11,
        color="#1F3B64",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def build_report_figures(project_root: Path, output_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_root = project_root / "coursework/outputs/stage2"
    stage3_root = project_root / "coursework/outputs/stage3/pavia_traditional"
    softmax = _json(stage2_root / "pavia_softmax_baseline/summary.json")
    sigmoid = _json(stage2_root / "pavia_sigmoid_ablation/summary.json")
    objectives = [softmax, sigmoid]
    _write_csv(output_dir / "hybridsn_objective_comparison.csv", objectives)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    labels = ["Softmax + CE", "Sigmoid + BCE"]
    x = np.arange(2)
    width = 0.26
    for offset, key, name, color in (
        (-width, "test_oa", "OA", "#2563EB"),
        (0.0, "test_aa", "AA", "#10B981"),
        (width, "test_kappa", "Kappa", "#F59E0B"),
    ):
        axes[0].bar(x + offset, [100 * row[key] for row in objectives], width, label=name, color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(99.5, 100.02)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("HybridSN objective ablation")
    axes[0].legend()
    axes[1].bar(x, [row["training_seconds"] for row in objectives], color=("#2563EB", "#F97316"))
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Accuracy comes with training-cost differences")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "hybridsn_objective_comparison.png", bbox_inches="tight")
    plt.close(figure)

    traditional = _csv(stage3_root / "traditional_method_comparison.csv")
    combined: list[dict[str, Any]] = []
    for row in traditional:
        combined.append(
            {
                "method": row["display_name"],
                "family": "traditional",
                "oa": float(row["test_oa"]),
                "aa": float(row["test_aa"]),
                "kappa": float(row["test_kappa"]),
                "training_seconds": float(row["training_seconds"]),
                "test_inference_seconds": float(row["test_inference_seconds"]),
                "model_size_bytes": int(row["model_size_bytes"]),
            }
        )
    for row, label in ((softmax, "HybridSN + Softmax"), (sigmoid, "HybridSN + Sigmoid")):
        checkpoint = Path(row["output_directory"]) / "checkpoint_best.pt"
        combined.append(
            {
                "method": label,
                "family": "deep learning",
                "oa": float(row["test_oa"]),
                "aa": float(row["test_aa"]),
                "kappa": float(row["test_kappa"]),
                "training_seconds": float(row["training_seconds"]),
                "test_inference_seconds": float(row["test_inference_seconds"]),
                "model_size_bytes": checkpoint.stat().st_size,
            }
        )
    _write_csv(output_dir / "all_method_comparison.csv", combined)
    positions = np.arange(len(combined))
    figure, axis = plt.subplots(figsize=(14, 6), dpi=180)
    for offset, key, label, color in (
        (-0.24, "oa", "OA", "#2563EB"),
        (0.0, "aa", "AA", "#10B981"),
        (0.24, "kappa", "Kappa", "#F59E0B"),
    ):
        axis.bar(positions + offset, [100 * row[key] for row in combined], 0.24, label=label, color=color)
    axis.set_xticks(positions, [row["method"] for row in combined], rotation=23, ha="right")
    axis.set_ylim(89, 100.2)
    axis.set_ylabel("Score (%)")
    axis.set_title("Traditional and deep-learning method comparison (same split, seed 1442)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "all_method_accuracy_comparison.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)
    label_offsets = {
        "PCA15 + SVM": (5, 5),
        "PCA15 + XGBoost": (5, -14),
        "PCA15 + LBP + SVM": (5, -16),
        "PCA15 + Gabor + SVM": (5, 6),
        "PCA15 + LBP + Gabor + SVM": (5, 8),
        "PCA15 + LBP + Gabor + XGBoost": (5, 6),
        "HybridSN + Softmax": (6, -14),
        "HybridSN + Sigmoid": (6, 8),
    }
    for row in combined:
        color = "#DC2626" if row["family"] == "deep learning" else "#2563EB"
        axis.scatter(
            row["test_inference_seconds"],
            100 * row["oa"],
            s=max(45, min(300, row["model_size_bytes"] / 12000)),
            color=color,
            alpha=0.8,
        )
        axis.annotate(
            row["method"],
            (row["test_inference_seconds"], 100 * row["oa"]),
            xytext=label_offsets.get(row["method"], (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Test inference time (s, log scale)")
    axis.set_ylabel("Test OA (%)")
    axis.set_title("Accuracy–efficiency trade-off (bubble size = model file size)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_efficiency_tradeoff.png", bbox_inches="tight")
    plt.close(figure)
    save_architecture_diagram(output_dir / "research_architecture.png")
    return {"objectives": objectives, "combined": combined}


__all__ = ["build_report_figures", "save_architecture_diagram"]
