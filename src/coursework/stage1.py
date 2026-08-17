"""Stage-1 dataset audit, frozen preprocessing routes and hand-off artifacts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.datasets.高光谱预处理 import (
    BandSelectionReducer,
    HSIPreprocessingPipeline,
    LDASpectralReducer,
    PCASpectralReducer,
    PreprocessingConfig,
    load_hsi_data,
)


SEED = 1442


def load_stage1_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("stage-1 YAML must contain a mapping")
    values = dict(values)
    dataset = values.get("dataset", {})
    if int(values.get("experiment", {}).get("seed", -1)) != SEED:
        raise ValueError("experiment.seed must be the frozen coursework seed 1442")
    if int(dataset.get("split_seed", -1)) != SEED:
        raise ValueError("dataset.split_seed must be the frozen coursework seed 1442")
    if dataset.get("split_protocol") != "fair24_6_70":
        raise ValueError("coursework comparisons require fair24_6_70")
    if values.get("selected_route") not in values.get("routes", {}):
        raise ValueError("selected_route must name one declared preprocessing route")
    return values


def route_config(values: Mapping[str, Any], route_key: str) -> PreprocessingConfig:
    route = values["routes"][route_key]
    reducer = str(route["reducer"]).lower()
    config = PreprocessingConfig(
        dataset_name=str(values["dataset"]["name"]),
        split_protocol=str(values["dataset"]["split_protocol"]),
        split_seed=int(values["dataset"]["split_seed"]),
        standardization=str(route.get("standardization", "standard")),
        reducer=reducer,
        n_components=(
            None if reducer == "none" else int(route.get("n_components", 15))
        ),
        whiten=bool(route.get("whiten", False)),
        band_selection_method=str(route.get("method", "fisher")),
        representation=str(values["spatial_preprocessing"]["representation"]),
        patch_size=int(values["spatial_preprocessing"]["patch_size"]),
        padding_mode=str(values["spatial_preprocessing"].get("padding_mode", "constant")),
        padding_value=float(values["spatial_preprocessing"].get("padding_value", 0.0)),
        output_dtype=str(values.get("output", {}).get("dtype", "float32")),
    )
    config.validate()
    return config


def _write_model_ready(
    path: Path,
    pipeline: HSIPreprocessingPipeline,
    data: Any,
) -> None:
    config = pipeline.config
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("1.0"),
        dataset_name=np.asarray(config.dataset_name),
        split_protocol=np.asarray(config.split_protocol),
        split_seed=np.asarray(config.split_seed, dtype=np.int64),
        config_fingerprint=np.asarray(config.fingerprint()),
        transformed_cube=np.ascontiguousarray(
            pipeline.transformed_cube_, dtype=config.output_dtype
        ),
        coordinates=np.ascontiguousarray(data.coordinates, dtype=np.int32),
        raw_labels=np.ascontiguousarray(data.labels, dtype=np.int16),
        train_indices=np.ascontiguousarray(data.train_indices, dtype=np.int64),
        validation_indices=np.ascontiguousarray(
            data.indices_by_split["validation"], dtype=np.int64
        ),
        test_indices=np.ascontiguousarray(data.indices_by_split["test"], dtype=np.int64),
        class_names=np.asarray(data.spec.class_names),
        patch_size=np.asarray(config.patch_size, dtype=np.int64),
        padding_mode=np.asarray(config.padding_mode),
        padding_value=np.asarray(config.padding_value, dtype=np.float32),
        num_classes=np.asarray(len(data.spec.class_names), dtype=np.int64),
    )
    temporary.replace(path)


def _rgb(cube: np.ndarray) -> np.ndarray:
    bands = np.rint(
        np.linspace(cube.shape[2] - 1, 0, 3, endpoint=True)
    ).astype(int)
    values = cube[:, :, bands].astype(np.float64)
    output = np.empty_like(values)
    for channel in range(3):
        low, high = np.percentile(values[:, :, channel], (2, 98))
        output[:, :, channel] = np.clip(
            (values[:, :, channel] - low) / max(high - low, np.finfo(float).eps), 0, 1
        )
    return output


def _save_dataset_overview(data: Any, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=170)
    axes[0].imshow(_rgb(data.cube))
    axes[0].set_title("Pseudo-RGB image")
    axes[1].imshow(data.label_map, cmap="tab20", interpolation="nearest")
    axes[1].set_title("Ground-truth labels (0 = background)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{data.spec.name}: cube {tuple(data.cube.shape)}")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _save_composition(data: Any, output: Path) -> None:
    class_ids = np.arange(1, len(data.spec.class_names) + 1)
    splits = ("train", "validation", "test")
    counts = np.asarray(
        [
            [
                np.count_nonzero(data.labels[data.indices_by_split[name]] == class_id)
                for class_id in class_ids
            ]
            for name in splits
        ]
    )
    figure, axis = plt.subplots(figsize=(12, 5.5), dpi=170)
    bottom = np.zeros(class_ids.size)
    for split_name, values, color in zip(
        splits, counts, ("#3B82F6", "#F59E0B", "#10B981"), strict=True
    ):
        axis.bar(class_ids, values, bottom=bottom, label=split_name, color=color)
        bottom += values
    axis.set_yscale("log")
    axis.set_xlabel("Class ID")
    axis.set_ylabel("Labeled pixels (log scale)")
    axis.set_title("Frozen class-stratified split: 24% / 6% / 70%, seed 1442")
    axis.set_xticks(class_ids)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _save_route_diagnostic(
    route_key: str,
    pipeline: HSIPreprocessingPipeline,
    data: Any,
    output: Path,
) -> None:
    reducer = pipeline.reducer
    figure, axis = plt.subplots(figsize=(8.5, 5), dpi=170)
    if isinstance(reducer, PCASpectralReducer):
        values = np.cumsum(reducer.explained_variance_ratio_)
        axis.plot(np.arange(1, values.size + 1), values, marker="o")
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("Cumulative explained variance ratio")
        axis.set_xlabel("Principal components")
    elif isinstance(reducer, BandSelectionReducer):
        if reducer.method == "fisher":
            axis.plot(np.arange(1, reducer.scores_.size + 1), reducer.scores_, lw=1.2)
            axis.scatter(
                reducer.selected_indices_ + 1,
                reducer.scores_[reducer.selected_indices_],
                color="#DC2626",
                label="selected",
            )
            axis.set_ylabel("Fisher score (train only)")
            axis.legend()
        else:
            axis.scatter(reducer.selected_indices_ + 1, np.ones(reducer.n_output_features))
            axis.set_yticks([])
        axis.set_xlabel("Original band number (1-based)")
    elif isinstance(reducer, LDASpectralReducer):
        coordinates = data.train_coordinates
        embedded = pipeline.transformed_cube_[coordinates[:, 0], coordinates[:, 1], :]
        for class_id in np.unique(data.train_labels):
            mask = data.train_labels == class_id
            axis.scatter(
                embedded[mask, 0],
                embedded[mask, min(1, embedded.shape[1] - 1)],
                s=4,
                alpha=0.35,
                label=str(class_id),
            )
        axis.set_xlabel("LDA component 1")
        axis.set_ylabel("LDA component 2")
        axis.legend(ncol=4, fontsize=7)
    else:
        train = data.cube[
            data.train_coordinates[:, 0], data.train_coordinates[:, 1], :
        ]
        axis.plot(train.mean(axis=0), color="#2563EB")
        axis.fill_between(
            np.arange(train.shape[1]),
            np.percentile(train, 25, axis=0),
            np.percentile(train, 75, axis=0),
            alpha=0.25,
        )
        axis.set_xlabel("Original band index")
        axis.set_ylabel("Reflectance / digital number")
    axis.set_title(route_key)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run_stage1(
    project_root: Path,
    config_path: Path,
    output_root: Path,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Fit all declared routes and publish a selected model-ready hand-off."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    values = load_stage1_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "figures"
    figures.mkdir(exist_ok=True)

    first_config = route_config(values, values["selected_route"])
    data = load_hsi_data(project_root, first_config)
    _save_dataset_overview(data, figures / "01_dataset_overview.png")
    _save_composition(data, figures / "02_class_and_split_distribution.png")

    rows: list[dict[str, Any]] = []
    route_artifacts: dict[str, Any] = {}
    for route_key in values["routes"]:
        config = route_config(values, route_key)
        route_dir = output_root / "routes" / route_key
        route_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        pipeline = HSIPreprocessingPipeline(config).fit(data)
        elapsed = time.perf_counter() - started
        pipeline.save_state(route_dir, overwrite=overwrite)
        _save_route_diagnostic(
            route_key, pipeline, data, figures / f"route_{route_key}.png"
        )
        model_ready = None
        if route_key in values.get("save_model_ready_routes", [values["selected_route"]]):
            model_ready = route_dir / "model_ready_dataset.npz"
            if model_ready.exists() and not overwrite:
                raise FileExistsError(model_ready)
            _write_model_ready(model_ready, pipeline, data)
        reducer = pipeline.reducer
        cumulative = None
        if isinstance(reducer, (PCASpectralReducer, LDASpectralReducer)):
            cumulative = float(reducer.explained_variance_ratio_.sum())
        selected = (
            (reducer.selected_indices_ + 1).tolist()
            if isinstance(reducer, BandSelectionReducer)
            else None
        )
        row = {
            "route": route_key,
            "standardization": config.standardization,
            "reducer": config.reducer,
            "band_selection_method": (
                config.band_selection_method if config.reducer == "band_selection" else ""
            ),
            "input_bands": int(data.cube.shape[2]),
            "output_bands": int(pipeline.output_bands),
            "fit_and_transform_seconds": elapsed,
            "cumulative_explained_variance_ratio": cumulative,
            "supervised_fit": config.reducer == "lda"
            or (
                config.reducer == "band_selection"
                and config.band_selection_method == "fisher"
            ),
            "validation_or_test_used_for_fit": False,
            "selected_band_numbers_one_based": json.dumps(selected),
        }
        rows.append(row)
        route_artifacts[route_key] = {
            "config": asdict(config),
            "config_fingerprint": config.fingerprint(),
            "state": str((route_dir / "preprocessing_state.npz").relative_to(project_root)),
            "metadata": str((route_dir / "metadata.json").relative_to(project_root)),
            "model_ready": (
                str(model_ready.relative_to(project_root)) if model_ready else None
            ),
        }

    csv_path = output_root / "preprocessing_route_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = route_artifacts[values["selected_route"]]
    handoff = {
        "schema_version": "1.0",
        "stage": "data_preprocessing",
        "dataset": values["dataset"],
        "frozen_protocol": {
            "train_fraction": 0.24,
            "validation_fraction": 0.06,
            "test_fraction": 0.70,
            "seed": SEED,
            "immutable_for_downstream_experiments": True,
        },
        "selected_route": values["selected_route"],
        "selected_artifact": selected,
        "all_routes": route_artifacts,
        "route_summary_csv": str(csv_path.relative_to(project_root)),
        "figures_directory": str(figures.relative_to(project_root)),
    }
    manifest = output_root / "stage1_manifest.json"
    manifest.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "selected_preprocessing.yaml").write_text(
        yaml.safe_dump(selected["config"], allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return {"manifest": handoff, "rows": rows, "data": data}


__all__ = ["SEED", "load_stage1_config", "route_config", "run_stage1"]
