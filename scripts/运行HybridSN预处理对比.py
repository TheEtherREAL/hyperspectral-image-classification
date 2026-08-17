"""Run a controlled HybridSN comparison across spectral preprocessing routes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.高光谱预处理 import PreprocessingConfig, load_hsi_data
from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)
from src.experiments.spectral_preprocessing import VARIANT_KEYS, fit_spectral_variant
from src.models.HybridSN模型 import HybridSN
from src.training.hybridsn_baseline import (
    ModelReadyArtifact,
    benchmark_model_compute,
    build_all_labeled_loader,
    build_loaders,
    infer_loader,
    sha256_file,
    train_one_epoch,
    write_json,
)
from src.utils.reproducibility import seed_everything
from src.visualization.hybridsn_results import (
    HSI_CLASS_COLORS,
    save_classification_maps,
    save_confusion_matrix,
    save_learning_curves,
    save_per_class_accuracy,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/模型训练/HybridSN_Pavia预处理对比.yaml"
PAVIA_CLASS_NAMES_ZH = (
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 HybridSN 光谱预处理公平对比。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, help="覆盖配置中的统一训练轮数。")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_KEYS,
        help="只运行指定的预处理路线；默认运行配置中的全部路线。",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true", help="冒烟时不评估测试集。")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="允许复用输出根目录；完整变体会跳过，残缺变体仍拒绝覆盖。",
    )
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_model_ready_artifact(path: Path, artifact: ModelReadyArtifact) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(artifact.schema_version),
        dataset_name=np.asarray(artifact.dataset_name),
        split_protocol=np.asarray(artifact.split_protocol),
        split_seed=np.asarray(artifact.split_seed, dtype=np.int64),
        config_fingerprint=np.asarray(artifact.config_fingerprint),
        transformed_cube=np.ascontiguousarray(artifact.transformed_cube, dtype=np.float32),
        coordinates=np.ascontiguousarray(artifact.coordinates, dtype=np.int32),
        raw_labels=np.ascontiguousarray(artifact.raw_labels, dtype=np.int16),
        train_indices=np.ascontiguousarray(artifact.train_indices, dtype=np.int64),
        validation_indices=np.ascontiguousarray(artifact.validation_indices, dtype=np.int64),
        test_indices=np.ascontiguousarray(artifact.test_indices, dtype=np.int64),
        class_names=np.asarray(artifact.class_names),
        patch_size=np.asarray(artifact.patch_size, dtype=np.int64),
        padding_mode=np.asarray(artifact.padding_mode),
        padding_value=np.asarray(artifact.padding_value, dtype=np.float32),
        num_classes=np.asarray(artifact.num_classes, dtype=np.int64),
    )


def save_comparison_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    validation_accuracy: float,
    variant_key: str,
    preprocessing_fingerprint: str,
    config: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "1.0",
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "variant_key": variant_key,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "test_set_used_for_model_selection": False,
        },
        temporary,
    )
    temporary.replace(path)


def prepare_artifact(data: Any, variant: Any, path: Path, config: Mapping[str, Any]) -> ModelReadyArtifact:
    spatial = config["spatial_preprocessing"]
    return ModelReadyArtifact(
        path=path,
        schema_version="comparison-1.0",
        dataset_name=data.spec.name,
        split_protocol=str(config["dataset"]["split_protocol"]),
        split_seed=int(config["dataset"]["split_seed"]),
        config_fingerprint=variant.fingerprint,
        transformed_cube=variant.transformed_cube,
        coordinates=data.coordinates,
        raw_labels=data.labels,
        train_indices=data.indices_by_split["train"],
        validation_indices=data.indices_by_split["validation"],
        test_indices=data.indices_by_split["test"],
        class_names=tuple(data.spec.class_names),
        patch_size=int(spatial["patch_size"]),
        padding_mode=str(spatial["padding_mode"]),
        padding_value=float(spatial["padding_value"]),
        num_classes=len(data.spec.class_names),
    )


def train_variant(
    *,
    variant_key: str,
    data: Any,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    skip_test: bool,
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=False, exist_ok=False)
    preprocessing_started = time.perf_counter()
    variant = fit_spectral_variant(
        data.cube,
        data.train_coordinates,
        data.train_labels,
        variant_key,
        output_bands=int(config["comparison"]["output_bands"]),
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    write_json(output_dir / "preprocessing_metadata.json", variant.metadata)
    np.savez_compressed(output_dir / "preprocessing_state.npz", **variant.state)

    artifact_path = output_dir / "model_ready_dataset.npz"
    artifact = prepare_artifact(data, variant, artifact_path, config)
    save_model_ready_artifact(artifact_path, artifact)
    assert artifact.output_bands == 15 and artifact.patch_size == 25

    seed = int(config["experiment"]["seed"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    pin_memory = device.type == "cuda"
    seed_everything(seed)
    loaders, _ = build_loaders(
        artifact,
        batch_size=batch_size,
        loader_seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if any(loaders[name] is None for name in ("train", "validation", "test")):
        raise RuntimeError("fair comparison requires non-empty train, validation and test")

    model = HybridSN(
        num_classes=artifact.num_classes,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 4_844_793:
        raise AssertionError("all comparison variants must use the identical HybridSN")

    history: list[dict[str, Any]] = []
    best_validation_accuracy = float("-inf")
    best_epoch = -1
    best_checkpoint = output_dir / "checkpoint_best.pt"
    training_peak_allocated = 0
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            non_blocking=pin_memory,
        )
        if device.type == "cuda":
            training_peak_allocated = max(
                training_peak_allocated, int(torch.cuda.max_memory_allocated(device))
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
            "train_loss": float(train_stats["loss"]),
            "train_accuracy": float(train_stats["accuracy"]),
            "validation_loss": float(validation.loss),
            "validation_accuracy": float(validation.accuracy),
            "epoch_seconds": float(train_stats["seconds"]),
        }
        history.append(record)
        if record["validation_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = record["validation_accuracy"]
            best_epoch = epoch
            save_comparison_checkpoint(
                best_checkpoint,
                model=model,
                epoch=epoch,
                validation_accuracy=best_validation_accuracy,
                variant_key=variant_key,
                preprocessing_fingerprint=variant.fingerprint,
                config=config,
            )
        print(
            f"[{variant_key}] epoch={epoch:03d}/{epochs} "
            f"train_loss={record['train_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"val_acc={record['validation_accuracy']:.4f} "
            f"best={best_validation_accuracy:.4f}@{best_epoch}",
            flush=True,
        )
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)
    training_seconds = time.perf_counter() - training_started
    save_learning_curves(history, output_dir / "learning_curves.png")

    if skip_test:
        write_json(
            output_dir / "status.json",
            {
                "status": "trained_without_test",
                "variant_key": variant_key,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_validation_accuracy,
            },
        )
        return None

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    if checkpoint["preprocessing_fingerprint"] != variant.fingerprint:
        raise ValueError("checkpoint and preprocessing fingerprint do not match")
    model.load_state_dict(checkpoint["model_state_dict"])
    benchmark = benchmark_model_compute(
        model,
        loaders["test"],
        device,
        non_blocking=pin_memory,
        warmup_iterations=int(config["evaluation"]["benchmark_warmup"]),
        measured_iterations=int(config["evaluation"]["benchmark_iterations"]),
    )
    test_output = infer_loader(
        model,
        loaders["test"],
        device,
        criterion=criterion,
        non_blocking=pin_memory,
    )
    summary = classification_summary(
        test_output.labels,
        test_output.predictions,
        num_classes=artifact.num_classes,
    )
    metrics = summary.to_dict(artifact.class_names)
    metrics.update(
        {
            "variant_key": variant_key,
            "display_name": variant.display_name,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "test_loss": test_output.loss,
            "test_samples": int(test_output.labels.size),
            "checkpoint": best_checkpoint.name,
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "preprocessing_fingerprint": variant.fingerprint,
            "test_set_used_for_model_selection": False,
        }
    )
    performance = {
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds_including_validation": training_seconds,
        "mean_train_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "training_peak_memory_allocated_bytes": (
            training_peak_allocated if device.type == "cuda" else None
        ),
        "trainable_parameters": parameter_count,
        "test_inference": {
            "elapsed_seconds": test_output.elapsed_seconds,
            "throughput_samples_per_second": test_output.throughput_samples_per_second,
            "peak_memory_allocated_bytes": test_output.peak_memory_allocated_bytes,
        },
        "model_compute_benchmark": benchmark,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "performance.json", performance)
    np.savez_compressed(
        output_dir / "predictions_test.npz",
        labels=test_output.labels.astype(np.int16),
        predictions=test_output.predictions.astype(np.int16),
        sample_indices=test_output.sample_indices.astype(np.int64),
        coordinates=test_output.coordinates.astype(np.int32),
    )
    save_confusion_matrix(summary.confusion_matrix, artifact.class_names, output_dir / "confusion_matrix.png")
    save_per_class_accuracy(
        summary.per_class_accuracy,
        artifact.class_names,
        output_dir / "per_class_accuracy.png",
    )

    ground_truth = np.zeros(artifact.image_shape, dtype=np.int16)
    ground_truth[artifact.coordinates[:, 0], artifact.coordinates[:, 1]] = artifact.raw_labels
    test_map = build_classification_map(
        artifact.image_shape,
        artifact.coordinates,
        test_output.sample_indices,
        test_output.predictions,
    )
    all_loader = build_all_labeled_loader(
        artifact,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    all_output = infer_loader(
        model,
        all_loader,
        device,
        criterion=None,
        non_blocking=pin_memory,
    )
    all_map = build_classification_map(
        artifact.image_shape,
        artifact.coordinates,
        all_output.sample_indices,
        all_output.predictions,
    )
    np.savez_compressed(
        output_dir / "classification_maps.npz",
        ground_truth=ground_truth,
        test_predictions=test_map,
        all_labeled_predictions=all_map,
    )
    save_classification_maps(
        ground_truth,
        test_map,
        all_map,
        artifact.class_names,
        output_dir / "classification_map.png",
    )
    performance["all_labeled_inference"] = {
        "samples": int(all_output.predictions.size),
        "elapsed_seconds": all_output.elapsed_seconds,
        "throughput_samples_per_second": all_output.throughput_samples_per_second,
    }
    write_json(output_dir / "performance.json", performance)
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "variant_key": variant_key,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )

    result_row = {
        "variant_key": variant_key,
        "display_name": variant.display_name,
        "standardization": variant.metadata["standardization"],
        "method": variant.metadata["method"],
        "whiten": variant.metadata["whiten"],
        "selected_bands_one_based": (
            ",".join(map(str, variant.metadata["selected_band_numbers_one_based"]))
            if variant.metadata["selected_band_numbers_one_based"] is not None
            else ""
        ),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation_oa": best_validation_accuracy,
        "test_oa": metrics["oa"],
        "test_aa": metrics["aa"],
        "test_kappa": metrics["kappa"],
        "test_errors": int(np.count_nonzero(test_output.labels != test_output.predictions)),
        "parameters": parameter_count,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_output.elapsed_seconds,
        "test_throughput_samples_per_second": test_output.throughput_samples_per_second,
    }
    write_json(output_dir / "summary_row.json", result_row)
    print(
        f"[{variant_key}] COMPLETE val={best_validation_accuracy:.6f} "
        f"OA={metrics['oa']:.6f} AA={metrics['aa']:.6f} "
        f"Kappa={metrics['kappa']:.6f}",
        flush=True,
    )
    del model, optimizer, loaders, all_loader, artifact, variant
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result_row


def load_completed_rows(output_dir: Path, variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for variant_key in variants:
        path = output_dir / variant_key / "summary_row.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def save_summary_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> None:
    frame = pd.DataFrame(rows)
    labels = frame["display_name"].tolist()
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#FFC000"][: len(frame)]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180)
    metric_specs = (
        ("validation_oa", "最佳验证集 OA", axes[0, 0]),
        ("test_oa", "测试集 OA", axes[0, 1]),
        ("test_aa", "测试集 AA", axes[1, 0]),
        ("test_kappa", "测试集 Kappa", axes[1, 1]),
    )
    for column, title, axis in metric_specs:
        values = frame[column].to_numpy(dtype=float)
        bars = axis.bar(np.arange(len(frame)), values, color=colors)
        lower = max(0.0, min(values) - 0.02)
        axis.set_ylim(lower, 1.005)
        axis.set_xticks(np.arange(len(frame)), labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.001,
                f"{value:.5f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(f"HybridSN 光谱预处理公平对比（fair24_6_70, seed{seed}）")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_metrics.png", bbox_inches="tight")
    plt.close(figure)

    per_class_rows = []
    for variant_key in frame["variant_key"]:
        metrics = json.loads((output_dir / variant_key / "metrics.json").read_text(encoding="utf-8"))
        per_class_rows.append([item["accuracy"] for item in metrics["per_class"]])
    per_class = np.asarray(per_class_rows)
    figure, axis = plt.subplots(figsize=(12.5, 5.5), dpi=180)
    image = axis.imshow(per_class * 100.0, cmap="YlGnBu", vmin=90, vmax=100, aspect="auto")
    axis.set_xticks(np.arange(9), PAVIA_CLASS_NAMES_ZH, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(frame)), labels)
    axis.set_title("逐类别测试准确率（%）")
    for row in range(per_class.shape[0]):
        for column in range(per_class.shape[1]):
            axis.text(column, row, f"{per_class[row, column] * 100:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_per_class_accuracy.png", bbox_inches="tight")
    plt.close(figure)


def save_combined_classification_maps(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    first_maps = np.load(output_dir / rows[0]["variant_key"] / "classification_maps.npz")
    ground_truth = first_maps["ground_truth"]
    maps = [("Ground truth", ground_truth)]
    for row in rows:
        with np.load(output_dir / row["variant_key"] / "classification_maps.npz") as values:
            maps.append((row["display_name"], values["all_labeled_predictions"].copy()))
    cmap = ListedColormap(HSI_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 10.5), cmap.N)
    figure, axes = plt.subplots(2, 3, figsize=(13, 15), dpi=180)
    for axis, (title, values) in zip(axes.reshape(-1), maps, strict=True):
        axis.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("PaviaU：不同光谱预处理的全部有标签像元分类图", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_classification_maps.png", bbox_inches="tight")
    plt.close(figure)


def write_experiment_record(output_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values(
        ["validation_oa", "best_epoch"], ascending=[False, True]
    )
    best_validation = float(frame["validation_oa"].max())
    tied = frame[np.isclose(frame["validation_oa"], best_validation, rtol=0.0, atol=1e-12)]
    tied_names = "、".join(f"**{name}**" for name in tied["display_name"])
    earliest_tied = tied.sort_values(["best_epoch", "variant_key"]).iloc[0]
    test_best = frame.sort_values(["test_oa", "variant_key"], ascending=[False, True]).iloc[0]
    training_seed = int(config["experiment"]["seed"])
    epochs = int(config["training"]["epochs"])
    lines = [
        "# HybridSN 光谱预处理对比实验记录",
        "",
        f"> 所有方法在同一 fair24_6_70 划分、统一 seed={training_seed}、{epochs} epoch 和相同 HybridSN 下比较；checkpoint 只由验证集 OA 选择。",
        "",
        "## 实验协议",
        "",
        f"- train={10265:,}，validation={2567:,}，test={29944:,}；split seed={config['dataset']['split_seed']}。",
        f"- 输入统一为 `N×1×15×25×25`，模型参数统一为 {int(frame['parameters'].iloc[0]):,}。",
        f"- Adam，lr={config['training']['learning_rate']}，batch={config['training']['batch_size']}，epochs={config['training']['epochs']}。",
        "- 标准化、PCA、Fisher 分数只使用训练中心像元拟合；验证集只选 checkpoint；测试集不参与选择。",
        "",
        "## 结果",
        "",
        "| 方法 | 最佳epoch | Validation OA | Test OA | Test AA | Kappa | 错分像元 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['display_name']} | {int(row['best_epoch'])} | {row['validation_oa']:.4%} | "
            f"{row['test_oa']:.4%} | {row['test_aa']:.4%} | {row['test_kappa']:.6f} | {int(row['test_errors'])} |"
        )
    lines.extend(
        [
            "",
            "## 验证集选择结论",
            "",
            f"按预先规定的 validation OA，{tied_names} 达到最高值（{best_validation:.4%}）。",
            (
                f"若预先增加“validation OA 相同时选择更早达到最佳值者”的规则，"
                f"则 {earliest_tied['display_name']}（第{int(earliest_tied['best_epoch'])}轮）优先；"
                "本实验未事后用测试结果破除验证集并列。"
                if len(tied) > 1
                else "本次 validation OA 存在唯一最高方法，无需追加并列判据。"
            ),
            f"{test_best['display_name']} 的测试 OA 在本次单 seed 中数值最高，只作描述，不能用测试集反向选择预处理方法。",
            "",
            "## 解释限制",
            "",
            "该划分仍是随机像元划分，25×25 patch 与训练区域高度重叠；结果适用于课程固定协议内部比较，不能等同于跨区域语义分割泛化。",
            "LDA8 未纳入纯预处理主表，因为原版 HybridSN 的连续 7/5/3 光谱卷积至少需要 13 个输入通道；加入 LDA8 必须改网络，构成混合变量实验。",
            "",
        ]
    )
    (output_dir / "实验记录.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        config["training"]["epochs"] = args.epochs
    variants = list(args.variants or config["comparison"]["variants"])
    if int(config["experiment"]["seed"]) != int(config["dataset"]["split_seed"]):
        raise ValueError(
            "experiment.seed must equal dataset.split_seed under the unified-seed protocol"
        )
    if len(set(variants)) != len(variants):
        raise ValueError("comparison variants must be unique")
    if any(key not in VARIANT_KEYS for key in variants):
        raise ValueError("config contains an unsupported comparison variant")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT
        / config["experiment"]["output_root"]
        / (
            f"hybridsn_preprocessing_comparison__{config['dataset']['split_protocol']}"
            f"__seed{int(config['experiment']['seed'])}__{timestamp}"
        )
    )
    if output_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.allow_existing_output)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    base_config = PreprocessingConfig(
        dataset_name=str(config["dataset"]["name"]),
        split_protocol=str(config["dataset"]["split_protocol"]),
        split_seed=int(config["dataset"]["split_seed"]),
        standardization="none",
        reducer="none",
        n_components=None,
        representation="patch",
        patch_size=int(config["spatial_preprocessing"]["patch_size"]),
        padding_mode=str(config["spatial_preprocessing"]["padding_mode"]),
        padding_value=float(config["spatial_preprocessing"]["padding_value"]),
        output_dtype="float32",
    )
    data = load_hsi_data(PROJECT_ROOT, base_config)
    device = select_device(args.device)
    environment = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
    }
    write_json(output_dir / "environment.json", environment)
    write_json(
        output_dir / "split_counts.json",
        {name: int(indices.size) for name, indices in data.indices_by_split.items()},
    )
    write_json(
        output_dir / "validation_spatial_overlap_audit.json",
        spatial_overlap_audit(
            data.label_map.shape,
            data.coordinates,
            data.labels,
            data.train_indices,
            data.indices_by_split["validation"],
            patch_size=base_config.patch_size,
            class_names=data.spec.class_names,
        ),
    )
    write_json(
        output_dir / "test_spatial_overlap_audit.json",
        spatial_overlap_audit(
            data.label_map.shape,
            data.coordinates,
            data.labels,
            data.train_indices,
            data.indices_by_split["test"],
            patch_size=base_config.patch_size,
            class_names=data.spec.class_names,
        ),
    )

    print(f"OUTPUT_DIR={output_dir}", flush=True)
    print(f"DEVICE={device} ({environment['gpu']})", flush=True)
    print(f"VARIANTS={variants}", flush=True)
    for variant_key in variants:
        variant_dir = output_dir / variant_key
        completed = variant_dir / "status.json"
        if completed.is_file():
            status = json.loads(completed.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                print(f"[{variant_key}] already complete; skipped", flush=True)
                continue
        if variant_dir.exists():
            raise FileExistsError(
                f"partial variant directory exists and will not be overwritten: {variant_dir}"
            )
        train_variant(
            variant_key=variant_key,
            data=data,
            config=config,
            output_dir=variant_dir,
            device=device,
            num_workers=args.num_workers,
            skip_test=args.skip_test,
        )

    if args.skip_test:
        print("Training smoke comparison completed without test evaluation.", flush=True)
        return 0
    rows = load_completed_rows(output_dir, variants)
    if len(rows) != len(variants):
        raise RuntimeError("not every requested comparison variant completed")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "comparison_summary.json", rows)
    save_summary_figures(output_dir, rows, seed=int(config["experiment"]["seed"]))
    save_combined_classification_maps(output_dir, rows)
    write_experiment_record(output_dir, config, rows)
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "variants": variants,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )
    print(frame[["display_name", "validation_oa", "test_oa", "test_aa", "test_kappa"]].to_string(index=False))
    print("COMPARISON_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
