"""Config-driven HybridSN training, validation, test and report artifacts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.datasets.高光谱预处理 import PreprocessingConfig
from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)
from src.models.可配置HybridSN import (
    ConfigurableHybridSN,
    HybridSNArchitecture,
    build_classification_objective,
)
from src.training.hybridsn_baseline import (
    benchmark_model_compute,
    build_all_labeled_loader,
    build_loaders,
    infer_loader,
    load_model_ready_artifact,
    train_one_epoch,
)
from src.utils.reproducibility import seed_everything
from src.visualization.hybridsn_results import (
    save_classification_maps,
    save_confusion_matrix,
    save_learning_curves,
    save_per_class_accuracy,
)


SEED = 1442


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(values), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_stage2_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("stage-2 YAML must contain a mapping")
    values = dict(values)
    seeds = (
        values.get("experiment", {}).get("seed"),
        values.get("dataset", {}).get("split_seed"),
        values.get("dataloader", {}).get("loader_seed"),
    )
    if any(int(seed) != SEED for seed in seeds):
        raise ValueError("experiment/split/loader seeds must all equal 1442")
    objective = str(values.get("classification", {}).get("objective", "softmax")).lower()
    if objective not in {"softmax", "sigmoid"}:
        raise ValueError("classification.objective must be softmax or sigmoid")
    return values


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _optimizer(model: torch.nn.Module, values: Mapping[str, Any]) -> torch.optim.Optimizer:
    training = values["training"]
    name = str(training.get("optimizer", "adam")).lower()
    kwargs = {
        "lr": float(training.get("learning_rate", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), momentum=float(training.get("momentum", 0.9)), **kwargs
        )
    raise ValueError(f"unsupported optimizer: {name}")


def run_stage2(
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one objective/architecture without using the test set for selection."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    values = load_stage2_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    stage1_manifest_path = _resolve(project_root, values["stage1_manifest"])
    stage1 = _read_json(stage1_manifest_path)
    if int(stage1["frozen_protocol"]["seed"]) != SEED:
        raise ValueError("stage-1 manifest is not seed 1442")
    if stage1["dataset"]["name"] != values["dataset"]["name"]:
        raise ValueError("stage-1 dataset does not match stage-2 config")
    selected = stage1["selected_artifact"]
    preprocessing = PreprocessingConfig(**selected["config"])
    artifact_path = _resolve(project_root, selected["model_ready"])
    artifact = load_model_ready_artifact(artifact_path, preprocessing)

    device_name = str(values.get("runtime", {}).get("device", "auto")).lower()
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    loader_values = values["dataloader"]
    batch_size = int(loader_values.get("batch_size", 256))
    num_workers = int(loader_values.get("num_workers", 0))
    pin_memory = bool(loader_values.get("pin_memory", True)) and device.type == "cuda"
    loaders, _ = build_loaders(
        artifact,
        batch_size=batch_size,
        loader_seed=SEED,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if loaders["validation"] is None:
        raise ValueError("fair comparison requires a non-empty validation set")

    architecture = HybridSNArchitecture.from_mapping(values)
    model = ConfigurableHybridSN(
        input_bands=artifact.output_bands,
        patch_size=artifact.patch_size,
        num_classes=artifact.num_classes,
        architecture=architecture,
    ).to(device)
    objective_name = str(values["classification"]["objective"]).lower()
    criterion = build_classification_objective(objective_name, artifact.num_classes)
    optimizer = _optimizer(model, values)
    epochs = int(values["training"].get("epochs", 30))
    patience = int(values["training"].get("early_stopping_patience", epochs))
    min_delta = float(values["training"].get("early_stopping_min_delta", 0.0))
    checkpoint = output_dir / "checkpoint_best.pt"

    history: list[dict[str, Any]] = []
    best_validation = -np.inf
    best_epoch = 0
    stale_epochs = 0
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            non_blocking=pin_memory,
        )
        validation = infer_loader(
            model,
            loaders["validation"],
            device,
            criterion=criterion,
            non_blocking=pin_memory,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(train["loss"]),
            "train_accuracy": float(train["accuracy"]),
            "validation_loss": float(validation.loss),
            "validation_accuracy": float(validation.accuracy),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(train["seconds"] + validation.elapsed_seconds),
        }
        history.append(record)
        if float(validation.accuracy) > best_validation + min_delta:
            best_validation = float(validation.accuracy)
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": best_validation,
                    "objective": objective_name,
                    "architecture": architecture.__dict__,
                    "seed": SEED,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train_acc={record['train_accuracy']:.6f} "
            f"val_acc={record['validation_accuracy']:.6f} "
            f"train_loss={record['train_loss']:.6f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break
    training_seconds = time.perf_counter() - training_started

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    validation = infer_loader(
        model,
        loaders["validation"],
        device,
        criterion=criterion,
        non_blocking=pin_memory,
    )
    test = infer_loader(
        model,
        loaders["test"],
        device,
        criterion=criterion,
        non_blocking=pin_memory,
    )
    all_loader = build_all_labeled_loader(
        artifact,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    all_output = infer_loader(
        model, all_loader, device, criterion=None, non_blocking=pin_memory
    )
    metrics = classification_summary(
        test.labels, test.predictions, num_classes=artifact.num_classes
    )
    validation_metrics = classification_summary(
        validation.labels, validation.predictions, num_classes=artifact.num_classes
    )

    image_shape = artifact.image_shape
    ground_truth = np.zeros(image_shape, dtype=np.int16)
    ground_truth[artifact.coordinates[:, 0], artifact.coordinates[:, 1]] = artifact.raw_labels
    test_map = build_classification_map(
        image_shape, artifact.coordinates, test.sample_indices, test.predictions
    )
    all_map = build_classification_map(
        image_shape, artifact.coordinates, all_output.sample_indices, all_output.predictions
    )
    np.savez_compressed(
        output_dir / "predictions_and_maps.npz",
        test_labels=test.labels,
        test_predictions=test.predictions,
        test_indices=test.sample_indices,
        test_coordinates=test.coordinates,
        ground_truth=ground_truth,
        test_prediction_map=test_map,
        all_labeled_prediction_map=all_map,
    )

    save_learning_curves(
        history,
        output_dir / "learning_curves.png",
        loss_title=(
            "Cross-entropy loss"
            if objective_name == "softmax"
            else "One-vs-rest BCE-with-logits loss"
        ),
    )
    save_confusion_matrix(
        metrics.confusion_matrix, artifact.class_names, output_dir / "confusion_matrix.png"
    )
    save_per_class_accuracy(
        metrics.per_class_accuracy,
        artifact.class_names,
        output_dir / "per_class_accuracy.png",
    )
    save_classification_maps(
        ground_truth,
        test_map,
        all_map,
        artifact.class_names,
        output_dir / "classification_maps.png",
        title=f"{artifact.dataset_name} - HybridSN ({objective_name})",
    )

    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metric_values = metrics.to_dict(artifact.class_names)
    metric_values.update(
        {
            "objective": objective_name,
            "best_epoch": best_epoch,
            "best_validation_oa": best_validation,
            "validation_oa_reloaded": validation_metrics.overall_accuracy,
            "test_set_used_for_fit_or_selection": False,
        }
    )
    _write_json(output_dir / "metrics.json", metric_values)
    overlap = spatial_overlap_audit(
        image_shape,
        artifact.coordinates,
        artifact.raw_labels,
        artifact.train_indices,
        artifact.test_indices,
        patch_size=artifact.patch_size,
        class_names=artifact.class_names,
    )
    _write_json(output_dir / "spatial_overlap_audit.json", overlap)
    benchmark = benchmark_model_compute(
        model,
        loaders["test"],
        device,
        non_blocking=pin_memory,
        warmup_iterations=5,
        measured_iterations=20,
    )
    performance = {
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": training_seconds,
        "epochs_completed": len(history),
        "validation_inference_seconds": validation.elapsed_seconds,
        "test_inference_seconds": test.elapsed_seconds,
        "test_throughput_samples_per_second": test.throughput_samples_per_second,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "compute_benchmark": benchmark,
    }
    _write_json(output_dir / "performance.json", performance)
    effective = dict(values)
    effective["resolved"] = {
        "device": str(device),
        "stage1_manifest": str(stage1_manifest_path),
        "model_ready_artifact": str(artifact_path),
        "input_bands": artifact.output_bands,
        "patch_size": artifact.patch_size,
        "num_classes": artifact.num_classes,
    }
    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "experiment": str(values["experiment"]["name"]),
        "objective": objective_name,
        "seed": SEED,
        "dataset": artifact.dataset_name,
        "split": artifact.split_protocol,
        "best_epoch": best_epoch,
        "validation_oa": best_validation,
        "test_oa": metrics.overall_accuracy,
        "test_aa": metrics.average_accuracy,
        "test_kappa": metrics.kappa,
        "test_errors": int(np.count_nonzero(test.labels != test.predictions)),
        "parameters": performance["parameters"],
        "training_seconds": training_seconds,
        "test_inference_seconds": test.elapsed_seconds,
        "test_throughput_samples_per_second": test.throughput_samples_per_second,
        "output_directory": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def compare_stage2_runs(run_directories: list[Path], output_csv: Path) -> list[dict[str, Any]]:
    rows = [_read_json(Path(directory) / "summary.json") for directory in run_directories]
    with Path(output_csv).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


__all__ = ["compare_stage2_runs", "load_stage2_config", "run_stage2"]
