"""Traditional LBP/Gabor + SVM/XGBoost comparison on frozen stage-1 data."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import yaml

from src.datasets.高光谱预处理 import PreprocessingConfig
from src.evaluation.classification_metrics import build_classification_map, classification_summary
from src.features.traditional_spatial import SpatialFeatureConfig, build_traditional_features
from src.training.hybridsn_baseline import load_model_ready_artifact
from src.utils.reproducibility import seed_everything
from src.visualization.hybridsn_results import (
    save_classification_maps,
    save_confusion_matrix,
    save_per_class_accuracy,
)


SEED = 1442


@dataclass
class TraditionalContext:
    project_root: Path
    values: dict[str, Any]
    artifact: Any
    feature_matrices: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    extraction_seconds: dict[str, float]


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(values), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_traditional_context(project_root: Path, config_path: Path) -> TraditionalContext:
    project_root = Path(project_root).resolve()
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("stage-3 YAML must contain a mapping")
    values = dict(values)
    if int(values["experiment"]["seed"]) != SEED:
        raise ValueError("experiment.seed must equal 1442")
    manifest = json.loads(
        _resolve(project_root, values["stage1_manifest"]).read_text(encoding="utf-8")
    )
    selected = manifest["selected_artifact"]
    preprocessing = PreprocessingConfig(**selected["config"])
    artifact = load_model_ready_artifact(
        _resolve(project_root, selected["model_ready"]), preprocessing
    )
    if artifact.split_seed != SEED or artifact.split_protocol != "fair24_6_70":
        raise ValueError("stage-3 requires the frozen fair seed-1442 artifact")
    settings = SpatialFeatureConfig(
        component_count=int(values["spatial_features"].get("component_count", 3)),
        lbp_bins=int(values["spatial_features"].get("lbp_bins", 16)),
        pooling_window=int(values["spatial_features"].get("pooling_window", 9)),
        gabor_orientations=tuple(
            float(value) for value in values["spatial_features"]["gabor_orientations"]
        ),
        gabor_frequencies=tuple(
            float(value) for value in values["spatial_features"]["gabor_frequencies"]
        ),
        gabor_sigma=float(values["spatial_features"].get("gabor_sigma", 2.5)),
        gabor_gamma=float(values["spatial_features"].get("gabor_gamma", 0.5)),
        gabor_radius=int(values["spatial_features"].get("gabor_radius", 6)),
    )
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    seconds: dict[str, float] = {}
    families = {
        "spectral": (True, False, False),
        "spectral_lbp": (True, True, False),
        "spectral_gabor": (True, False, True),
        "spectral_lbp_gabor": (True, True, True),
    }
    for key, flags in families.items():
        started = time.perf_counter()
        matrices[key], names[key] = build_traditional_features(
            artifact.transformed_cube,
            artifact.coordinates,
            include_spectral=flags[0],
            include_lbp=flags[1],
            include_gabor=flags[2],
            config=settings,
        )
        seconds[key] = time.perf_counter() - started
    return TraditionalContext(
        project_root=project_root,
        values=values,
        artifact=artifact,
        feature_matrices=matrices,
        feature_names=names,
        extraction_seconds=seconds,
    )


def _classifier(name: str, values: Mapping[str, Any]) -> Any:
    if name == "svm":
        settings = values["classifiers"]["svm"]
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(settings.get("C", 16.0)),
                kernel=str(settings.get("kernel", "rbf")),
                gamma=settings.get("gamma", "scale"),
                cache_size=float(settings.get("cache_size_mb", 2048)),
                decision_function_shape="ovr",
            ),
        )
    if name == "xgboost":
        settings = values["classifiers"]["xgboost"]
        return XGBClassifier(
            n_estimators=int(settings.get("n_estimators", 200)),
            max_depth=int(settings.get("max_depth", 6)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            subsample=float(settings.get("subsample", 0.9)),
            colsample_bytree=float(settings.get("colsample_bytree", 0.9)),
            min_child_weight=float(settings.get("min_child_weight", 1.0)),
            reg_lambda=float(settings.get("reg_lambda", 1.0)),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method=str(settings.get("tree_method", "hist")),
            random_state=SEED,
            n_jobs=int(settings.get("n_jobs", -1)),
        )
    raise ValueError(f"unsupported classifier: {name}")


def run_traditional_method(
    context: TraditionalContext,
    method_key: str,
    output_root: Path,
) -> dict[str, Any]:
    if method_key not in context.values["methods"]:
        raise ValueError(f"unknown method: {method_key}")
    method = context.values["methods"][method_key]
    feature_key = str(method["features"])
    classifier_key = str(method["classifier"])
    matrix = context.feature_matrices[feature_key]
    artifact = context.artifact
    targets = artifact.raw_labels.astype(np.int64) - 1
    train_indices = artifact.train_indices
    validation_indices = artifact.validation_indices
    test_indices = artifact.test_indices
    model = _classifier(classifier_key, context.values)
    seed_everything(SEED)

    output_dir = Path(output_root).resolve() / method_key
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.fit(matrix[train_indices], targets[train_indices])
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    validation_predictions = model.predict(matrix[validation_indices])
    validation_seconds = time.perf_counter() - started
    validation = classification_summary(
        targets[validation_indices],
        validation_predictions,
        num_classes=artifact.num_classes,
    )
    started = time.perf_counter()
    test_predictions = model.predict(matrix[test_indices])
    test_seconds = time.perf_counter() - started
    test = classification_summary(
        targets[test_indices], test_predictions, num_classes=artifact.num_classes
    )
    started = time.perf_counter()
    all_predictions = model.predict(matrix)
    all_seconds = time.perf_counter() - started

    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    ground_truth = np.zeros(artifact.image_shape, dtype=np.int16)
    ground_truth[artifact.coordinates[:, 0], artifact.coordinates[:, 1]] = artifact.raw_labels
    test_map = build_classification_map(
        artifact.image_shape, artifact.coordinates, test_indices, test_predictions
    )
    all_map = build_classification_map(
        artifact.image_shape,
        artifact.coordinates,
        np.arange(artifact.raw_labels.size),
        all_predictions,
    )
    np.savez_compressed(
        output_dir / "predictions_and_maps.npz",
        test_labels=targets[test_indices],
        test_predictions=test_predictions,
        test_indices=test_indices,
        ground_truth=ground_truth,
        test_prediction_map=test_map,
        all_labeled_prediction_map=all_map,
    )
    save_confusion_matrix(
        test.confusion_matrix, artifact.class_names, output_dir / "confusion_matrix.png"
    )
    save_per_class_accuracy(
        test.per_class_accuracy, artifact.class_names, output_dir / "per_class_accuracy.png"
    )
    save_classification_maps(
        ground_truth,
        test_map,
        all_map,
        artifact.class_names,
        output_dir / "classification_maps.png",
        title=f"{artifact.dataset_name} - {method_key}",
    )
    metrics = test.to_dict(artifact.class_names)
    metrics.update(
        {
            "method": method_key,
            "validation_oa": validation.overall_accuracy,
            "test_set_used_for_fit_or_selection": False,
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "feature_names.txt").write_text(
        "\n".join(context.feature_names[feature_key]) + "\n", encoding="utf-8"
    )
    summary = {
        "method": method_key,
        "display_name": str(method.get("display_name", method_key)),
        "features": feature_key,
        "classifier": classifier_key,
        "seed": SEED,
        "feature_dimension": int(matrix.shape[1]),
        "validation_oa": validation.overall_accuracy,
        "test_oa": test.overall_accuracy,
        "test_aa": test.average_accuracy,
        "test_kappa": test.kappa,
        "test_errors": int(np.count_nonzero(targets[test_indices] != test_predictions)),
        "feature_extraction_seconds": context.extraction_seconds[feature_key],
        "training_seconds": training_seconds,
        "validation_inference_seconds": validation_seconds,
        "test_inference_seconds": test_seconds,
        "all_labeled_inference_seconds": all_seconds,
        "test_throughput_samples_per_second": int(test_indices.size) / test_seconds,
        "model_size_bytes": model_path.stat().st_size,
        "output_directory": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def save_method_comparison(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root = Path(output_root).resolve()
    with (output_root / "traditional_method_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [row["display_name"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5.8), dpi=180)
    for offset, key, name, color in (
        (-width, "test_oa", "OA", "#2563EB"),
        (0.0, "test_aa", "AA", "#10B981"),
        (width, "test_kappa", "Kappa", "#F59E0B"),
    ):
        axis.bar(
            positions + offset,
            [100 * row[key] for row in rows],
            width,
            label=name,
            color=color,
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(80, 101)
    axis.set_title("Traditional feature and classifier comparison (seed 1442)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "traditional_method_comparison.png", bbox_inches="tight")
    plt.close(figure)


def run_all_traditional(
    project_root: Path, config_path: Path, output_root: Path
) -> list[dict[str, Any]]:
    context = prepare_traditional_context(project_root, config_path)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method_key in context.values["methods"]:
        summary_path = output_root / method_key / "summary.json"
        if summary_path.is_file():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"reused complete result: {method_key}", flush=True)
        else:
            row = run_traditional_method(context, method_key, output_root)
            print(
                f"completed: {method_key} test_OA={100 * row['test_oa']:.4f}%",
                flush=True,
            )
        rows.append(row)
    save_method_comparison(rows, output_root)
    return rows


__all__ = [
    "TraditionalContext",
    "prepare_traditional_context",
    "run_all_traditional",
    "run_traditional_method",
    "save_method_comparison",
]
