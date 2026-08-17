"""运行改进 HybridSN（BatchNorm + 残差 + 全局平均池化）与原始 HybridSN 的公平对比。

在同一 fair24_6_70 划分、seed=1442、PCA15 + 25×25 patch、相同训练协议
（Adam / lr=1e-3 / batch=256 / 30 epoch / 验证集选 checkpoint）下，训练改进
模型并与已完成的原始 HybridSN 结果（standard_pca15）对比。

测试集只用于最终评估，不参与拟合、调参或 checkpoint 选择。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
)
from src.experiments.spectral_preprocessing import fit_spectral_variant
from src.models.改进HybridSN import ImprovedHybridSN
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
    save_classification_maps,
    save_confusion_matrix,
    save_learning_curves,
    save_per_class_accuracy,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/模型训练/HybridSN_Pavia改进对比.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行改进 HybridSN 与原始 HybridSN 的公平对比。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, help="覆盖配置中的统一训练轮数。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true", help="冒烟时不评估测试集。")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


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


def build_model(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    model = config["model"]
    return ImprovedHybridSN(
        input_bands=int(config["preprocessing"]["output_bands"]),
        patch_size=int(config["spatial_preprocessing"]["patch_size"]),
        num_classes=num_classes,
        conv3d_channels=tuple(int(v) for v in model["conv3d_channels"]),
        spectral_kernel_sizes=tuple(int(v) for v in model["spectral_kernel_sizes"]),
        spatial_kernel_size=int(model["spatial_kernel_size"]),
        conv2d_channels=int(model["conv2d_channels"]),
        dense_units=tuple(int(v) for v in model["dense_units"]),
        dropout=float(model["dropout"]),
        batch_normalization=bool(model["batch_normalization"]),
        residual_connections=bool(model["residual_connections"]),
    )


def find_reference_dir() -> Path:
    """定位已完成的原始 HybridSN（standard_pca15）结果目录。"""
    candidates = sorted(
        (PROJECT_ROOT / "experiments").glob(
            "hybridsn_preprocessing_comparison__fair24_6_70__seed1442__*/standard_pca15"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "metrics.json").is_file() and (candidate / "summary_row.json").is_file():
            return candidate
    raise FileNotFoundError("未找到已完成的原始 HybridSN standard_pca15 结果目录")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    validation_accuracy: float,
    config: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "1.0",
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "test_set_used_for_model_selection": False,
        },
        temporary,
    )
    temporary.replace(path)


def save_comparison_chart(
    path: Path,
    original: Mapping[str, Any],
    improved: Mapping[str, Any],
) -> None:
    """绘制原始 vs 改进 HybridSN 的 OA/AA/Kappa 对比柱状图（纵轴截断）。"""
    names = ["原始 HybridSN", "改进 HybridSN\n(BatchNorm+残差+GAP)"]
    oa = [original["test_oa"] * 100.0, improved["test_oa"] * 100.0]
    aa = [original["test_aa"] * 100.0, improved["test_aa"] * 100.0]
    kappa = [original["test_kappa"] * 100.0, improved["test_kappa"] * 100.0]
    x = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    axis.bar(x - width, oa, width, label="OA (%)", color="#2563EB")
    axis.bar(x, aa, width, label="AA (%)", color="#10B981")
    axis.bar(x + width, kappa, width, label="Kappa (%)", color="#F59E0B")
    for i in range(len(names)):
        axis.text(x[i] - width, oa[i] + 0.05, f"{oa[i]:.3f}", ha="center", va="bottom", fontsize=8)
        axis.text(x[i], aa[i] + 0.05, f"{aa[i]:.3f}", ha="center", va="bottom", fontsize=8)
        axis.text(x[i] + width, kappa[i] + 0.05, f"{kappa[i]:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_ylim(99.0, 100.25)  # 纵轴截断（不从 0 开始），放大细微差异
    axis.set_xticks(x)
    axis.set_xticklabels(names)
    axis.set_ylabel("得分 (%)")
    axis.set_title("原始 HybridSN 与改进 HybridSN 精度对比（Pavia University 测试集）")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def train_improved(
    *,
    data: Any,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    skip_test: bool,
) -> dict[str, Any]:
    preprocessing_started = time.perf_counter()
    variant = fit_spectral_variant(
        data.cube,
        data.train_coordinates,
        data.train_labels,
        str(config["preprocessing"]["variant"]),
        output_bands=int(config["preprocessing"]["output_bands"]),
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    write_json(output_dir / "preprocessing_metadata.json", variant.metadata)

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

    model = build_model(config, artifact.num_classes).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[improved] trainable_parameters={parameter_count:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

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
            model, loaders["train"], criterion, optimizer, device, non_blocking=pin_memory
        )
        if device.type == "cuda":
            training_peak_allocated = max(
                training_peak_allocated, int(torch.cuda.max_memory_allocated(device))
            )
        validation = infer_loader(
            model, loaders["validation"], device, criterion=criterion, non_blocking=pin_memory
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
            save_checkpoint(
                best_checkpoint,
                model=model,
                epoch=epoch,
                validation_accuracy=best_validation_accuracy,
                config=config,
            )
        print(
            f"[improved] epoch={epoch:03d}/{epochs} "
            f"train_loss={record['train_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"val_acc={record['validation_accuracy']:.4f} "
            f"best={best_validation_accuracy:.4f}@{best_epoch}",
            flush=True,
        )
        write_json(output_dir / "history.json", history)
    training_seconds = time.perf_counter() - training_started
    save_learning_curves(history, output_dir / "learning_curves.png")

    if skip_test:
        write_json(
            output_dir / "status.json",
            {
                "status": "trained_without_test",
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_validation_accuracy,
            },
        )
        return {}

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
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
        model, loaders["test"], device, criterion=criterion, non_blocking=pin_memory
    )
    summary = classification_summary(
        test_output.labels, test_output.predictions, num_classes=artifact.num_classes
    )
    metrics = summary.to_dict(artifact.class_names)
    metrics.update(
        {
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "test_loss": test_output.loss,
            "test_samples": int(test_output.labels.size),
            "trainable_parameters": parameter_count,
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
        summary.per_class_accuracy, artifact.class_names, output_dir / "per_class_accuracy.png"
    )

    ground_truth = np.zeros(artifact.image_shape, dtype=np.int16)
    ground_truth[artifact.coordinates[:, 0], artifact.coordinates[:, 1]] = artifact.raw_labels
    test_map = build_classification_map(
        artifact.image_shape, artifact.coordinates, test_output.sample_indices, test_output.predictions
    )
    all_loader = build_all_labeled_loader(
        artifact, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )
    all_output = infer_loader(model, all_loader, device, criterion=None, non_blocking=pin_memory)
    all_map = build_classification_map(
        artifact.image_shape, artifact.coordinates, all_output.sample_indices, all_output.predictions
    )
    np.savez_compressed(
        output_dir / "classification_maps.npz",
        ground_truth=ground_truth,
        test_predictions=test_map,
        all_labeled_predictions=all_map,
    )
    save_classification_maps(
        ground_truth, test_map, all_map, artifact.class_names, output_dir / "classification_map.png"
    )

    result_row = {
        "method_key": "D4_PCA15_ImprovedHybridSN",
        "display_name": "PCA15 + 改进 HybridSN",
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
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )
    print(
        f"[improved] COMPLETE val={best_validation_accuracy:.6f} "
        f"OA={metrics['oa']:.6f} AA={metrics['aa']:.6f} "
        f"Kappa={metrics['kappa']:.6f} params={parameter_count:,}",
        flush=True,
    )
    return result_row


def main() -> int:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        config["training"]["epochs"] = args.epochs
    if int(config["experiment"]["seed"]) != int(config["dataset"]["split_seed"]):
        raise ValueError("experiment.seed must equal dataset.split_seed under the unified-seed protocol")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT
        / config["experiment"]["output_root"]
        / f"hybridsn_improvement_comparison__{config['dataset']['split_protocol']}__seed{int(config['experiment']['seed'])}__{timestamp}"
    )
    variant_dir = output_root / "improved_bn_residual_gap"
    variant_dir.mkdir(parents=True, exist_ok=False)
    (output_root / "config.yaml").write_text(
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
    write_json(
        output_root / "environment.json",
        {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
    )

    print(f"OUTPUT_DIR={output_root}", flush=True)
    print(f"DEVICE={device}", flush=True)
    improved_row = train_improved(
        data=data,
        config=config,
        output_dir=variant_dir,
        device=device,
        num_workers=args.num_workers,
        skip_test=args.skip_test,
    )

    if args.skip_test:
        return 0

    # 与已完成的原始 HybridSN 结果对比
    reference_dir = find_reference_dir()
    original_metrics = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))
    original_summary = json.loads((reference_dir / "summary_row.json").read_text(encoding="utf-8"))
    improved_metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))

    comparison = {
        "protocol": {
            "split": "fair24_6_70",
            "seed": int(config["experiment"]["seed"]),
            "input": "N x 1 x 15 x 25 x 25",
            "optimizer": "adam",
            "learning_rate": float(config["training"]["learning_rate"]),
            "batch_size": int(config["training"]["batch_size"]),
            "epochs": int(config["training"]["epochs"]),
        },
        "original_hybridsn": {
            "source": str(reference_dir),
            "trainable_parameters": int(original_summary["parameters"]),
            "test_oa": original_metrics["oa"],
            "test_aa": original_metrics["aa"],
            "test_kappa": original_metrics["kappa"],
            "test_errors": int(original_summary["test_errors"]),
        },
        "improved_hybridsn": {
            "source": str(variant_dir),
            "trainable_parameters": int(improved_metrics["trainable_parameters"]),
            "test_oa": improved_metrics["oa"],
            "test_aa": improved_metrics["aa"],
            "test_kappa": improved_metrics["kappa"],
            "test_errors": improved_row["test_errors"],
        },
    }
    write_json(output_root / "comparison.json", comparison)
    save_comparison_chart(
        output_root / "original_vs_improved.png",
        comparison["original_hybridsn"],
        comparison["improved_hybridsn"],
    )

    original = comparison["original_hybridsn"]
    improved = comparison["improved_hybridsn"]
    print("\n===== 原始 vs 改进 HybridSN 对比 =====")
    print(
        f"原始  HybridSN: OA={original['test_oa']:.6f} AA={original['test_aa']:.6f} "
        f"Kappa={original['test_kappa']:.6f} params={original['trainable_parameters']:,}"
    )
    print(
        f"改进  HybridSN: OA={improved['test_oa']:.6f} AA={improved['test_aa']:.6f} "
        f"Kappa={improved['test_kappa']:.6f} params={improved['trainable_parameters']:,}"
    )
    delta_oa = (improved["test_oa"] - original["test_oa"]) * 100.0
    reduction = (1.0 - improved["trainable_parameters"] / original["trainable_parameters"]) * 100.0
    print(f"Test OA 变化: {delta_oa:+.4f} 个百分点；参数量下降 {reduction:.1f}%")
    print(f"COMPARISON_COMPLETE={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
