"""Run architecture-guided traditional/deep HSI comparison groups.

This script keeps the fixed fair24_6_70 split and seed=1442, changes one
factor at a time, and imports already-completed HybridSN rows instead of
retraining them.  Test labels are never used for fitting or model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import yaml


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.高光谱预处理 import (
    HSIPreprocessingPipeline,
    PreprocessingConfig,
    load_hsi_data,
)
from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)
from src.experiments.spectral_preprocessing import fit_spectral_variant
from src.features.traditional_spatial import SpatialFeatureConfig, build_traditional_features
from src.visualization.hybridsn_results import (
    HSI_CLASS_COLORS,
    save_confusion_matrix,
    save_per_class_accuracy,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/模型训练/Pavia研究架构分组对比.yaml"
CLASS_NAMES_ZH = (
    "沥青路面",
    "草地",
    "砾石",
    "树木",
    "涂漆金属板",
    "裸土",
    "沥青材料",
    "自锁砖",
    "阴影",
)

METHOD_ORDER = (
    "R1_PCA15_SVM",
    "R2_LDA8_SVM",
    "R3_Uniform15_SVM",
    "S1_PCA15_LBP_SVM",
    "S2_PCA15_Gabor_SVM",
    "S3_PCA15_LBP_Gabor_SVM",
    "C1_PCA15_LBP_Gabor_HistGB",
    "D1_PCA15_HybridSN",
    "D2_Uniform15_HybridSN",
    "D3_Fisher15_HybridSN",
)

METHOD_INFO = {
    "R1_PCA15_SVM": ("PCA15 + SVM", "降维对比", "PCA15", "光谱", "RBF-SVM"),
    "R2_LDA8_SVM": ("LDA8 + SVM", "降维对比", "LDA8", "光谱", "RBF-SVM"),
    "R3_Uniform15_SVM": ("均匀15波段 + SVM", "降维对比", "均匀波段选择15", "光谱", "RBF-SVM"),
    "S1_PCA15_LBP_SVM": ("PCA15 + LBP + SVM", "空间特征消融", "PCA15", "光谱+LBP", "RBF-SVM"),
    "S2_PCA15_Gabor_SVM": ("PCA15 + Gabor + SVM", "空间特征消融", "PCA15", "光谱+Gabor", "RBF-SVM"),
    "S3_PCA15_LBP_Gabor_SVM": (
        "PCA15 + LBP + Gabor + SVM",
        "空间特征消融",
        "PCA15",
        "光谱+LBP+Gabor",
        "RBF-SVM",
    ),
    "C1_PCA15_LBP_Gabor_HistGB": (
        "PCA15 + LBP + Gabor + HistGB",
        "分类器对比",
        "PCA15",
        "光谱+LBP+Gabor",
        "HistGradientBoosting",
    ),
    "D1_PCA15_HybridSN": ("PCA15 + HybridSN", "深度学习对照", "PCA15", "25×25 patch", "HybridSN + CE"),
    "D2_Uniform15_HybridSN": (
        "均匀15波段 + HybridSN",
        "深度学习对照",
        "均匀波段选择15",
        "25×25 patch",
        "HybridSN + CE",
    ),
    "D3_Fisher15_HybridSN": (
        "Fisher15波段 + HybridSN",
        "深度学习对照",
        "Fisher波段选择15",
        "25×25 patch",
        "HybridSN + CE",
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行研究架构图对应的分组对比实验。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_base_config(config: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_name=str(config["dataset"]["name"]),
        split_protocol=str(config["dataset"]["split_protocol"]),
        split_seed=int(config["dataset"]["split_seed"]),
        standardization="none",
        reducer="none",
        n_components=None,
        representation="pixel",
        patch_size=1,
        padding_mode="constant",
        padding_value=0.0,
        output_dtype="float32",
    )


def fit_reduced_cubes(data: Any, config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    cubes: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    output_bands = int(config["reduction"]["pca_components"])

    for key, variant_key in (("pca15", "standard_pca15"), ("uniform15", "standard_uniform15")):
        started = time.perf_counter()
        result = fit_spectral_variant(
            data.cube,
            data.train_coordinates,
            data.train_labels,
            variant_key,
            output_bands=output_bands,
        )
        seconds[key] = time.perf_counter() - started
        cubes[key] = result.transformed_cube
        metadata[key] = result.metadata

    lda_components = int(config["reduction"]["lda_components"])
    lda_config = PreprocessingConfig(
        dataset_name=data.spec.name,
        split_protocol=str(config["dataset"]["split_protocol"]),
        split_seed=int(config["dataset"]["split_seed"]),
        standardization="standard",
        reducer="lda",
        n_components=lda_components,
        representation="pixel",
        patch_size=1,
        padding_mode="constant",
        padding_value=0.0,
        output_dtype="float32",
    )
    started = time.perf_counter()
    lda = HSIPreprocessingPipeline(lda_config).fit(data)
    seconds["lda8"] = time.perf_counter() - started
    cubes["lda8"] = lda.transformed_cube_
    metadata["lda8"] = lda.fit_metadata_
    return cubes, seconds, metadata


def build_feature_cache(
    data: Any,
    cubes: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, float]]:
    feature_config = SpatialFeatureConfig(
        component_count=int(config["spatial_features"]["component_count"]),
        lbp_bins=int(config["spatial_features"]["lbp_bins"]),
        pooling_window=int(config["spatial_features"]["pooling_window"]),
        gabor_orientations=tuple(float(v) for v in config["spatial_features"]["gabor_orientations"]),
        gabor_frequencies=tuple(float(v) for v in config["spatial_features"]["gabor_frequencies"]),
        gabor_sigma=float(config["spatial_features"]["gabor_sigma"]),
        gabor_gamma=float(config["spatial_features"]["gabor_gamma"]),
        gabor_radius=int(config["spatial_features"]["gabor_radius"]),
    )
    coordinates = data.coordinates
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    seconds: dict[str, float] = {}

    for key in ("pca15", "lda8", "uniform15"):
        started = time.perf_counter()
        matrices[key], names[key] = build_traditional_features(
            cubes[key], coordinates, include_spectral=True, config=feature_config
        )
        seconds[key] = time.perf_counter() - started

    for key, include_lbp, include_gabor in (
        ("pca15_lbp", True, False),
        ("pca15_gabor", False, True),
    ):
        started = time.perf_counter()
        matrices[key], names[key] = build_traditional_features(
            cubes["pca15"],
            coordinates,
            include_spectral=False,
            include_lbp=include_lbp,
            include_gabor=include_gabor,
            config=feature_config,
        )
        seconds[key] = time.perf_counter() - started

    matrices["pca15_spectral_lbp"] = np.ascontiguousarray(
        np.concatenate((matrices["pca15"], matrices["pca15_lbp"]), axis=1)
    )
    names["pca15_spectral_lbp"] = names["pca15"] + names["pca15_lbp"]
    seconds["pca15_spectral_lbp"] = seconds["pca15"] + seconds["pca15_lbp"]
    matrices["pca15_spectral_gabor"] = np.ascontiguousarray(
        np.concatenate((matrices["pca15"], matrices["pca15_gabor"]), axis=1)
    )
    names["pca15_spectral_gabor"] = names["pca15"] + names["pca15_gabor"]
    seconds["pca15_spectral_gabor"] = seconds["pca15"] + seconds["pca15_gabor"]
    matrices["pca15_spectral_lbp_gabor"] = np.ascontiguousarray(
        np.concatenate(
            (matrices["pca15"], matrices["pca15_lbp"], matrices["pca15_gabor"]),
            axis=1,
        )
    )
    names["pca15_spectral_lbp_gabor"] = (
        names["pca15"] + names["pca15_lbp"] + names["pca15_gabor"]
    )
    seconds["pca15_spectral_lbp_gabor"] = (
        seconds["pca15"] + seconds["pca15_lbp"] + seconds["pca15_gabor"]
    )
    return matrices, names, seconds


def make_classifier(classifier_key: str, config: dict[str, Any]) -> Any:
    seed = int(config["experiment"]["seed"])
    if classifier_key == "svm":
        values = config["classifiers"]["svm"]
        classifier = SVC(
            C=float(values["C"]),
            kernel=str(values["kernel"]),
            gamma=values["gamma"],
            cache_size=float(values["cache_size_mb"]),
            decision_function_shape="ovr",
        )
    elif classifier_key == "histgb":
        values = config["classifiers"]["histogram_gradient_boosting"]
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(values["learning_rate"]),
            max_iter=int(values["max_iter"]),
            max_leaf_nodes=int(values["max_leaf_nodes"]),
            l2_regularization=float(values["l2_regularization"]),
            early_stopping=bool(values["early_stopping"]),
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier: {classifier_key}")
    return make_pipeline(StandardScaler(), classifier)


def save_generic_classification_map(
    ground_truth: np.ndarray,
    test_map: np.ndarray,
    all_map: np.ndarray,
    path: Path,
    title: str,
) -> None:
    cmap = ListedColormap(HSI_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 10.5), cmap.N)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 7.2), dpi=180)
    image = None
    for axis, values, subtitle in zip(
        axes,
        (ground_truth, test_map, all_map),
        ("Ground truth", "Test predictions", "All labeled predictions"),
        strict=True,
    ):
        image = axis.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(subtitle)
        axis.axis("off")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02, ticks=np.arange(10))
    colorbar.ax.set_yticklabels(["背景", *CLASS_NAMES_ZH], fontsize=8)
    figure.suptitle(title, fontsize=14)
    figure.subplots_adjust(left=0.01, right=0.86, top=0.91, bottom=0.02, wspace=0.04)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_traditional_method(
    method_key: str,
    feature_key: str,
    classifier_key: str,
    *,
    data: Any,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    reduction_seconds: float,
    extraction_seconds: float,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    method_dir = output_dir / method_key
    method_dir.mkdir()
    train_indices = data.indices_by_split["train"]
    validation_indices = data.indices_by_split["validation"]
    test_indices = data.indices_by_split["test"]
    targets = data.labels.astype(np.int64) - 1
    model = make_classifier(classifier_key, config)

    started = time.perf_counter()
    model.fit(feature_matrix[train_indices], targets[train_indices])
    training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    validation_predictions = model.predict(feature_matrix[validation_indices])
    validation_seconds = time.perf_counter() - started
    validation = classification_summary(
        targets[validation_indices], validation_predictions, num_classes=len(data.spec.class_names)
    )

    started = time.perf_counter()
    test_predictions = model.predict(feature_matrix[test_indices])
    test_seconds = time.perf_counter() - started
    test = classification_summary(
        targets[test_indices], test_predictions, num_classes=len(data.spec.class_names)
    )

    started = time.perf_counter()
    all_predictions = model.predict(feature_matrix)
    all_seconds = time.perf_counter() - started

    model_path = method_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    model_size = model_path.stat().st_size
    estimator = model.steps[-1][1]
    support_vectors = int(estimator.n_support_.sum()) if isinstance(estimator, SVC) else None
    internal_iterations = (
        int(np.max(np.asarray(estimator.n_iter_)))
        if isinstance(estimator, HistGradientBoostingClassifier)
        else None
    )

    metrics = test.to_dict(data.spec.class_names)
    metrics.update(
        {
            "method_key": method_key,
            "validation_oa": validation.overall_accuracy,
            "test_set_used_for_fit_or_selection": False,
        }
    )
    write_json(method_dir / "metrics.json", metrics)
    performance = {
        "reduction_fit_and_full_cube_transform_seconds": reduction_seconds,
        "spatial_feature_extraction_seconds": extraction_seconds,
        "classifier_training_seconds": training_seconds,
        "validation_inference_seconds": validation_seconds,
        "test_inference_seconds": test_seconds,
        "test_throughput_samples_per_second": int(test_indices.size) / test_seconds,
        "all_labeled_inference_seconds": all_seconds,
        "model_file_bytes": model_size,
        "support_vectors": support_vectors,
        "histgb_internal_iterations": internal_iterations,
    }
    write_json(method_dir / "performance.json", performance)
    (method_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    np.savez_compressed(
        method_dir / "predictions_test.npz",
        labels=targets[test_indices].astype(np.int16),
        predictions=test_predictions.astype(np.int16),
        sample_indices=test_indices.astype(np.int64),
        coordinates=data.coordinates[test_indices].astype(np.int32),
    )

    ground_truth = data.label_map.astype(np.int16, copy=True)
    test_map = build_classification_map(
        data.label_map.shape, data.coordinates, test_indices, test_predictions
    )
    all_map = build_classification_map(
        data.label_map.shape,
        data.coordinates,
        np.arange(data.coordinates.shape[0]),
        all_predictions,
    )
    np.savez_compressed(
        method_dir / "classification_maps.npz",
        ground_truth=ground_truth,
        test_predictions=test_map,
        all_labeled_predictions=all_map,
    )
    save_confusion_matrix(test.confusion_matrix, CLASS_NAMES_ZH, method_dir / "confusion_matrix.png")
    save_per_class_accuracy(
        test.per_class_accuracy, CLASS_NAMES_ZH, method_dir / "per_class_accuracy.png"
    )
    save_generic_classification_map(
        ground_truth,
        test_map,
        all_map,
        method_dir / "classification_map.png",
        METHOD_INFO[method_key][0],
    )

    display_name, group, reducer, spatial, classifier = METHOD_INFO[method_key]
    row = {
        "method_key": method_key,
        "display_name": display_name,
        "primary_group": group,
        "reduction": reducer,
        "spatial_features": spatial,
        "classifier": classifier,
        "feature_dimension": int(feature_matrix.shape[1]),
        "validation_oa": validation.overall_accuracy,
        "test_oa": test.overall_accuracy,
        "test_aa": test.average_accuracy,
        "test_kappa": test.kappa,
        "test_errors": int(np.count_nonzero(targets[test_indices] != test_predictions)),
        "parameters_or_support_vectors": support_vectors,
        "preprocessing_seconds": reduction_seconds + extraction_seconds,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_seconds,
        "test_throughput_samples_per_second": int(test_indices.size) / test_seconds,
        "model_size_bytes": model_size,
        "source_directory": str(method_dir),
        "result_origin": "executed_in_this_run",
    }
    write_json(method_dir / "summary_row.json", row)
    print(
        f"[{method_key}] val={row['validation_oa']:.6f} testOA={row['test_oa']:.6f} "
        f"AA={row['test_aa']:.6f} train={training_seconds:.2f}s",
        flush=True,
    )
    return row


def find_deep_source(config: dict[str, Any]) -> Path:
    pattern = str(config["deep_results"]["source_pattern"])
    required = list(config["deep_results"]["variants"].values())
    candidates = sorted(
        (PROJECT_ROOT / "experiments").glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all(
            (candidate / variant / "status.json").is_file()
            and json.loads((candidate / variant / "status.json").read_text(encoding="utf-8"))[
                "status"
            ]
            == "complete"
            for variant in required
        ):
            return candidate
    raise FileNotFoundError(f"no complete deep comparison run matches {pattern!r}")


def load_deep_rows(config: dict[str, Any], source: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for method_key, variant in config["deep_results"]["variants"].items():
        variant_dir = source / str(variant)
        source_row = json.loads((variant_dir / "summary_row.json").read_text(encoding="utf-8"))
        metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
        performance = json.loads((variant_dir / "performance.json").read_text(encoding="utf-8"))
        display_name, group, reducer, spatial, classifier = METHOD_INFO[method_key]
        checkpoint = variant_dir / "checkpoint_best.pt"
        row = {
            "method_key": method_key,
            "display_name": display_name,
            "primary_group": group,
            "reduction": reducer,
            "spatial_features": spatial,
            "classifier": classifier,
            "feature_dimension": 15 * 25 * 25,
            "validation_oa": float(source_row["validation_oa"]),
            "test_oa": float(source_row["test_oa"]),
            "test_aa": float(source_row["test_aa"]),
            "test_kappa": float(source_row["test_kappa"]),
            "test_errors": int(source_row["test_errors"]),
            "parameters_or_support_vectors": int(source_row["parameters"]),
            "preprocessing_seconds": float(source_row["preprocessing_seconds"]),
            "training_seconds": float(source_row["training_seconds"]),
            "test_inference_seconds": float(source_row["test_inference_seconds"]),
            "test_throughput_samples_per_second": float(
                source_row["test_throughput_samples_per_second"]
            ),
            "model_size_bytes": checkpoint.stat().st_size,
            "source_directory": str(variant_dir),
            "result_origin": "reused_completed_seed1442_run",
        }
        rows.append(row)
        details[method_key] = {
            "metrics": metrics,
            "performance": performance,
            "classification_maps": variant_dir / "classification_maps.npz",
        }
    return rows, details


def save_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)


def save_metric_groups(output_dir: Path, rows: list[dict[str, Any]], groups: dict[str, list[str]]) -> None:
    lookup = {row["method_key"]: row for row in rows}
    titles = {
        "reduction_control": "A. 降维/波段选择（固定 SVM）",
        "spatial_ablation": "B. 空间特征消融（固定 PCA15 + SVM）",
        "classifier_control": "C. 分类器对比（固定融合特征）",
        "paradigm_control": "D. 传统方法与深度学习对照",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    for axis, group_key in zip(axes.reshape(-1), titles, strict=True):
        group_rows = [lookup[key] for key in groups[group_key]]
        labels = [row["display_name"] for row in group_rows]
        values = np.asarray([row["test_oa"] for row in group_rows]) * 100.0
        bars = axis.bar(np.arange(len(values)), values, color="#4472C4")
        lower = max(0.0, float(values.min()) - 3.0)
        axis.set_ylim(lower, min(100.2, float(values.max()) + 1.2))
        axis.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
        axis.set_ylabel("测试集 OA (%)")
        axis.set_title(titles[group_key])
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Pavia University 研究架构分组结果（fair24_6_70，seed=1442）", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "group_comparison_metrics.png", bbox_inches="tight")
    plt.close(figure)


def save_efficiency_plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 7), dpi=180)
    colors = {
        "RBF-SVM": "#4472C4",
        "HistGradientBoosting": "#ED7D31",
        "HybridSN + CE": "#70AD47",
    }
    for row in rows:
        axis.scatter(
            row["training_seconds"],
            row["test_oa"] * 100,
            s=75,
            color=colors[row["classifier"]],
            alpha=0.9,
        )
        axis.annotate(
            row["method_key"].split("_")[0],
            (row["training_seconds"], row["test_oa"] * 100),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("模型训练时间（秒，对数轴；不含预处理）")
    axis.set_ylabel("测试集 OA (%)")
    axis.set_title("精度—训练成本权衡")
    axis.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for label, color in colors.items()
    ]
    axis.legend(handles=handles, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_efficiency_tradeoff.png", bbox_inches="tight")
    plt.close(figure)


def save_per_class_heatmap(
    output_dir: Path,
    rows: list[dict[str, Any]],
    deep_details: dict[str, dict[str, Any]],
) -> None:
    values = []
    labels = []
    for row in rows:
        if row["method_key"] in deep_details:
            metrics = deep_details[row["method_key"]]["metrics"]
        else:
            metrics = json.loads(
                (output_dir / row["method_key"] / "metrics.json").read_text(encoding="utf-8")
            )
        values.append([item["accuracy"] for item in metrics["per_class"]])
        labels.append(row["display_name"])
    matrix = np.asarray(values) * 100.0
    figure, axis = plt.subplots(figsize=(14.5, 8.2), dpi=180)
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=max(0.0, matrix.min() - 1), vmax=100, aspect="auto")
    axis.set_xticks(np.arange(9), CLASS_NAMES_ZH, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("各方法逐类别测试准确率（%）")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_dir / "per_class_accuracy_heatmap.png", bbox_inches="tight")
    plt.close(figure)


def load_map_for_method(
    output_dir: Path, method_key: str, deep_details: dict[str, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        deep_details[method_key]["classification_maps"]
        if method_key in deep_details
        else output_dir / method_key / "classification_maps.npz"
    )
    with np.load(path) as artifact:
        return artifact["ground_truth"].copy(), artifact["all_labeled_predictions"].copy()


def save_representative_maps(
    output_dir: Path, deep_details: dict[str, dict[str, Any]]
) -> None:
    representative = (
        "R1_PCA15_SVM",
        "S3_PCA15_LBP_Gabor_SVM",
        "C1_PCA15_LBP_Gabor_HistGB",
        "D1_PCA15_HybridSN",
        "D2_Uniform15_HybridSN",
    )
    ground_truth, _ = load_map_for_method(output_dir, representative[0], deep_details)
    maps = [("Ground truth", ground_truth)]
    for key in representative:
        _, prediction = load_map_for_method(output_dir, key, deep_details)
        maps.append((METHOD_INFO[key][0], prediction))
    cmap = ListedColormap(HSI_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 10.5), cmap.N)
    figure, axes = plt.subplots(2, 3, figsize=(14, 15), dpi=180)
    for axis, (title, values) in zip(axes.reshape(-1), maps, strict=True):
        axis.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle("代表性方法的全部有标签像元分类图", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "representative_classification_maps.png", bbox_inches="tight")
    plt.close(figure)


def save_group_design(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(15, 8), dpi=180)
    axis.axis("off")
    box = dict(boxstyle="round,pad=0.55", facecolor="#4472C4", edgecolor="#2F5597", alpha=0.95)
    light_box = dict(boxstyle="round,pad=0.45", facecolor="#D9EAF7", edgecolor="#4472C4")
    axis.text(0.04, 0.5, "PaviaU\n高光谱立方体", ha="center", va="center", color="white", bbox=box, transform=axis.transAxes)
    axis.annotate("", xy=(0.17, 0.5), xytext=(0.09, 0.5), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.24, 0.72, "A 降维控制\nPCA15 / LDA8 / 均匀15波段", ha="center", va="center", bbox=light_box, transform=axis.transAxes)
    axis.text(0.24, 0.28, "固定条件\nfair24_6_70 + seed1442\n仅训练集拟合预处理", ha="center", va="center", bbox=light_box, transform=axis.transAxes)
    axis.annotate("", xy=(0.39, 0.63), xytext=(0.34, 0.67), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.48, 0.72, "B 空间消融\n光谱 / +LBP / +Gabor / 融合", ha="center", va="center", bbox=light_box, transform=axis.transAxes)
    axis.annotate("", xy=(0.61, 0.67), xytext=(0.57, 0.67), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.70, 0.72, "C 分类器控制\nRBF-SVM / HistGB", ha="center", va="center", bbox=light_box, transform=axis.transAxes)
    axis.annotate("", xy=(0.82, 0.67), xytext=(0.79, 0.67), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.91, 0.72, "传统分支结果\nOA / AA / Kappa\n时间 / 分类图", ha="center", va="center", color="white", bbox=box, transform=axis.transAxes)
    axis.annotate("", xy=(0.43, 0.32), xytext=(0.34, 0.32), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.54, 0.32, "D 深度学习对照\nPCA / 均匀波段 / Fisher波段\nHybridSN + CrossEntropy", ha="center", va="center", bbox=light_box, transform=axis.transAxes)
    axis.annotate("", xy=(0.82, 0.32), xytext=(0.67, 0.32), arrowprops=dict(arrowstyle="->", lw=2), xycoords=axis.transAxes)
    axis.text(0.91, 0.32, "深度分支结果\n验证集选 checkpoint\n测试集仅最终评估", ha="center", va="center", color="white", bbox=box, transform=axis.transAxes)
    axis.set_title("研究架构图对应的控制变量实验设计", fontsize=17, pad=20)
    figure.tight_layout()
    figure.savefig(output_dir / "architecture_group_design.png", bbox_inches="tight")
    plt.close(figure)


def percent_delta(left: dict[str, Any], right: dict[str, Any], key: str = "test_oa") -> float:
    return (float(right[key]) - float(left[key])) * 100.0


def write_experiment_record(
    output_dir: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    deep_source: Path,
) -> None:
    lookup = {row["method_key"]: row for row in rows}
    r1, r2, r3 = (lookup[key] for key in METHOD_ORDER[:3])
    s1, s2, s3 = (lookup[key] for key in METHOD_ORDER[3:6])
    c1 = lookup["C1_PCA15_LBP_Gabor_HistGB"]
    d1 = lookup["D1_PCA15_HybridSN"]
    best_validation = max(rows, key=lambda row: (row["validation_oa"], -METHOD_ORDER.index(row["method_key"])))
    best_test = max(rows, key=lambda row: row["test_oa"])
    lines = [
        "# Pavia University 研究架构分组对比实验记录",
        "",
        "> 固定划分：fair24_6_70；划分、模型及采样统一 seed=1442；所有统计预处理仅在训练中心像元拟合。",
        "",
        "## 1. 任务与公平协议",
        "",
        "- 数据不是无序点云，而是 `610×340×103` 的高光谱图像立方体；每个空间像元对应一条 103 波段光谱。",
        "- 当前模型输出的是中心像元所属的 9 个地物类别。把所有有标签像元的预测放回二维坐标后得到分类图。",
        "- 因而本实验本质是**像元/patch 中心分类**；外观类似语义分割，但不是端到端密集语义分割网络。",
        "- train=10,265，validation=2,567，test=29,944；测试集不用于拟合、调参或 checkpoint 选择。",
        "- 传统模型使用固定超参数；深度模型复用已经完成的 30 epoch、验证集选 checkpoint 的 seed1442 结果。",
        "",
        "## 2. 分组设计",
        "",
        "| 分组 | 控制变量 | 比较内容 |",
        "|---|---|---|",
        "| A 降维/波段选择 | 固定光谱特征 + RBF-SVM | PCA15、LDA8、均匀15波段 |",
        "| B 空间特征消融 | 固定 PCA15 + RBF-SVM | 光谱、+LBP、+Gabor、LBP+Gabor |",
        "| C 分类器 | 固定 PCA15 + LBP + Gabor | RBF-SVM、HistGradientBoosting |",
        "| D 方法范式 | 同一划分与种子 | 传统像元/手工空间特征、HybridSN patch |",
        "",
        "## 3. 全部结果",
        "",
        "| 编号 | 方法 | 特征维数 | Val OA | Test OA | Test AA | Kappa | 错分 | 训练/s | 测试推理/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_key'].split('_')[0]} | {row['display_name']} | {row['feature_dimension']} | "
            f"{row['validation_oa']:.4%} | {row['test_oa']:.4%} | {row['test_aa']:.4%} | "
            f"{row['test_kappa']:.6f} | {row['test_errors']} | {row['training_seconds']:.3f} | "
            f"{row['test_inference_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 分组分析",
            "",
            f"- A组：相对 PCA15+SVM，LDA8 的 Test OA 变化 {percent_delta(r1, r2):+.3f} 个百分点，均匀波段选择变化 {percent_delta(r1, r3):+.3f} 个百分点。该组只改变降维/选带方式。",
            f"- B组：相对纯 PCA 光谱 SVM，加入 LBP、Gabor、二者融合的 Test OA 分别变化 {percent_delta(r1, s1):+.3f}、{percent_delta(r1, s2):+.3f}、{percent_delta(r1, s3):+.3f} 个百分点。LBP 与 Gabor 不应默认“越多越好”，应以验证集判断互补性。",
            f"- C组：固定融合特征后，HistGB 相对 SVM 的 Test OA 变化 {percent_delta(s3, c1):+.3f} 个百分点；训练时间由 {s3['training_seconds']:.3f}s 变为 {c1['training_seconds']:.3f}s。",
            f"- D组：PCA15 HybridSN 相对纯 PCA15 SVM 的 Test OA 变化 {percent_delta(r1, d1):+.3f} 个百分点，相对手工融合 SVM 变化 {percent_delta(s3, d1):+.3f} 个百分点。深度模型参数量更大，并利用 25×25 空间上下文。",
            f"- 按 validation OA 的预设选择规则，当前最高为 **{best_validation['display_name']}**（{best_validation['validation_oa']:.4%}）。{best_test['display_name']} 的 Test OA 数值最高（{best_test['test_oa']:.4%}），这里只作最终描述，不用于反向选模。",
            "",
            "## 5. 分类头解释",
            "",
            "PaviaU 每个像元只有一个互斥类别，因此 HybridSN 使用 9 维 logits 与 CrossEntropyLoss；其数学上对应 Softmax 多分类。Sigmoid 将 9 类视为相互独立，适合多标签任务，不适合作为本课题主实验分类头，故未把不科学的 Sigmoid 组放入主表。",
            "",
            "## 6. 局限与建设性改进",
            "",
            "1. **先改划分再追模型。** 随机像元划分下，邻近 train/test patch 高度重叠，易得到过于乐观的结果。建议增加按连通区域或空间块划分，保留本表作为课程协议结果。",
            "2. **精确补做 XGBoost。** 当前项目环境未安装 `xgboost`，C1 是同属梯度提升树家族的 sklearn HistGradientBoosting，不得在报告中写成 XGBoost。安装后可复用同一融合特征接口增加真正的 XGBoost 组。",
            "3. **用验证集做小型参数网格。** SVM 的 C/gamma、LBP 窗口、Gabor 频率与方向应只在 validation 上选择；最终只评估一次 test。",
            "4. **做多种子稳健性。** 本轮遵从统一 seed1442；最终论文式结论建议固定同一批 3–5 个种子，报告均值±标准差，而不是只看单次差异。",
            "5. **优化 HybridSN。** 可尝试更小 patch、BatchNorm/残差、注意力或轻量光谱卷积，并同时报告参数量、显存、吞吐率，防止只追 OA。",
            "6. **关注难分类类别。** 结合逐类别热力图和混淆矩阵定位砾石/沥青材料/自锁砖等光谱或纹理相近类别，再决定做判别选带、类别重加权或边界增强。",
            "",
            "## 7. 结果来源",
            "",
            f"- 深度结果复用目录：`{deep_source}`。",
            "- 传统方法在本目录实际执行并保存模型、测试预测、混淆矩阵、逐类准确率和分类图。",
            "- `architecture_group_design.png` 为实验设计；`group_comparison_metrics.png` 为四组主结果；`accuracy_efficiency_tradeoff.png` 展示精度—耗时权衡。",
            "",
        ]
    )
    (output_dir / "实验记录.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    if seed != 1442 or seed != int(config["dataset"]["split_seed"]):
        raise ValueError("this comparison requires the unified experiment/split seed 1442")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT
        / str(config["experiment"]["output_root"])
        / f"hsi_architecture_comparison__fair24_6_70__seed1442__{timestamp}"
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(
        output_dir / "environment.json",
        {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": seed,
            "xgboost_installed": False,
            "xgboost_note": "HistGradientBoosting is a declared substitute, not XGBoost.",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
    )

    np.random.seed(seed)
    data = load_hsi_data(PROJECT_ROOT, create_base_config(config))
    write_json(
        output_dir / "split_counts.json",
        {name: int(indices.size) for name, indices in data.indices_by_split.items()},
    )
    write_json(
        output_dir / "spatial_overlap_audit_patch9.json",
        spatial_overlap_audit(
            data.label_map.shape,
            data.coordinates,
            data.labels,
            data.indices_by_split["train"],
            data.indices_by_split["test"],
            patch_size=int(config["spatial_features"]["pooling_window"]),
            class_names=data.spec.class_names,
        ),
    )

    print(f"OUTPUT_DIR={output_dir}", flush=True)
    cubes, reduction_seconds, reduction_metadata = fit_reduced_cubes(data, config)
    write_json(output_dir / "reduction_metadata.json", reduction_metadata)
    features, feature_names, extraction_seconds = build_feature_cache(data, cubes, config)
    feature_spec = {
        key: {"samples": int(matrix.shape[0]), "dimensions": int(matrix.shape[1])}
        for key, matrix in features.items()
    }
    write_json(output_dir / "feature_dimensions.json", feature_spec)

    traditional_specs = (
        ("R1_PCA15_SVM", "pca15", "svm", "pca15"),
        ("R2_LDA8_SVM", "lda8", "svm", "lda8"),
        ("R3_Uniform15_SVM", "uniform15", "svm", "uniform15"),
        ("S1_PCA15_LBP_SVM", "pca15_spectral_lbp", "svm", "pca15"),
        ("S2_PCA15_Gabor_SVM", "pca15_spectral_gabor", "svm", "pca15"),
        ("S3_PCA15_LBP_Gabor_SVM", "pca15_spectral_lbp_gabor", "svm", "pca15"),
        ("C1_PCA15_LBP_Gabor_HistGB", "pca15_spectral_lbp_gabor", "histgb", "pca15"),
    )
    rows: list[dict[str, Any]] = []
    for method_key, feature_key, classifier_key, reduction_key in traditional_specs:
        rows.append(
            run_traditional_method(
                method_key,
                feature_key,
                classifier_key,
                data=data,
                feature_matrix=features[feature_key],
                feature_names=feature_names[feature_key],
                reduction_seconds=reduction_seconds[reduction_key],
                extraction_seconds=extraction_seconds[feature_key],
                config=config,
                output_dir=output_dir,
            )
        )

    deep_source = find_deep_source(config)
    deep_rows, deep_details = load_deep_rows(config, deep_source)
    rows.extend(deep_rows)
    order = {key: index for index, key in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda row: order[row["method_key"]])
    save_summary_csv(output_dir, rows)
    write_json(output_dir / "group_definitions.json", config["groups"])
    save_group_design(output_dir)
    save_metric_groups(output_dir, rows, config["groups"])
    save_efficiency_plot(output_dir, rows)
    save_per_class_heatmap(output_dir, rows, deep_details)
    save_representative_maps(output_dir, deep_details)
    write_experiment_record(output_dir, config, rows, deep_source)
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "seed": seed,
            "methods": list(METHOD_ORDER),
        },
    )
    print(f"COMPLETE={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
