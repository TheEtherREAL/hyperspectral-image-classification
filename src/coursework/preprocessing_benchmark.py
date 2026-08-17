"""Fair SVM benchmark across frozen raw/PCA/LDA/band-selection routes."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.datasets.高光谱预处理 import HSIPreprocessingPipeline, PreprocessingConfig, load_hsi_data
from src.evaluation.classification_metrics import classification_summary


def run_preprocessing_svm_benchmark(
    project_root: Path,
    stage1_manifest_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(stage1_manifest_path).read_text(encoding="utf-8"))
    if int(manifest["frozen_protocol"]["seed"]) != 1442:
        raise ValueError("benchmark requires seed 1442")
    first = next(iter(manifest["all_routes"].values()))
    data = load_hsi_data(project_root, PreprocessingConfig(**first["config"]))
    targets = data.labels.astype(np.int64) - 1
    rows: list[dict[str, Any]] = []
    for route_key, entry in manifest["all_routes"].items():
        route_dir = project_root / Path(entry["state"]).parent
        pipeline = HSIPreprocessingPipeline.load_state(
            route_dir / "preprocessing_state.npz", route_dir / "metadata.json"
        )
        started = time.perf_counter()
        transformed = pipeline.attach_transformed_cube(data.cube)
        transform_seconds = time.perf_counter() - started
        features = transformed[data.coordinates[:, 0], data.coordinates[:, 1], :]
        model = make_pipeline(
            StandardScaler(),
            SVC(C=16.0, kernel="rbf", gamma="scale", cache_size=2048),
        )
        started = time.perf_counter()
        model.fit(features[data.train_indices], targets[data.train_indices])
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        all_predictions = model.predict(features)
        all_inference_seconds = time.perf_counter() - started
        validation = classification_summary(
            targets[data.indices_by_split["validation"]],
            all_predictions[data.indices_by_split["validation"]],
            num_classes=len(data.spec.class_names),
        )
        test = classification_summary(
            targets[data.indices_by_split["test"]],
            all_predictions[data.indices_by_split["test"]],
            num_classes=len(data.spec.class_names),
        )
        model_path = output_dir / f"{route_key}__svm.joblib"
        joblib.dump(model, model_path, compress=3)
        row = {
            "route": route_key,
            "seed": 1442,
            "input_bands": int(data.cube.shape[2]),
            "output_bands": int(transformed.shape[2]),
            "validation_oa": validation.overall_accuracy,
            "test_oa": test.overall_accuracy,
            "test_aa": test.average_accuracy,
            "test_kappa": test.kappa,
            "test_errors": int(
                np.count_nonzero(
                    targets[data.indices_by_split["test"]]
                    != all_predictions[data.indices_by_split["test"]]
                )
            ),
            "full_cube_transform_seconds": transform_seconds,
            "training_seconds": training_seconds,
            "all_labeled_inference_seconds": all_inference_seconds,
            "model_size_bytes": model_path.stat().st_size,
            "test_set_used_for_fit_or_selection": False,
        }
        rows.append(row)
        print(f"{route_key}: OA={100 * row['test_oa']:.4f}%", flush=True)
    with (output_dir / "preprocessing_svm_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "preprocessing_svm_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    labels = [row["route"] for row in rows]
    positions = np.arange(len(rows))
    axes[0].bar(positions, [100 * row["test_oa"] for row in rows], color="#2563EB")
    axes[0].set_ylim(80, 101)
    axes[0].set_ylabel("Test OA (%)")
    axes[0].set_title("Preprocessing accuracy (same RBF-SVM)")
    axes[1].bar(
        positions,
        [row["training_seconds"] + row["all_labeled_inference_seconds"] for row in rows],
        color="#F59E0B",
    )
    axes[1].set_ylabel("Training + all-labeled inference (s)")
    axes[1].set_title("Accuracy/efficiency control")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "preprocessing_svm_comparison.png", bbox_inches="tight")
    plt.close(figure)
    return rows


__all__ = ["run_preprocessing_svm_benchmark"]
