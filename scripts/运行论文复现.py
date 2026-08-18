"""论文复现与多模型对比的统一训练 / 推理 / 评估脚本。

本脚本在统一协议（fair24_6_70 + seed1442 + 统一评价指标）下运行任意一个模型
（论文复现模型 Paper3DCNN / Paper3D1DCNN / 改进版，或 HybridSN / 改进 HybridSN），
复用 ``src/training/hybridsn_baseline.py`` 的 DataLoader、训练循环、checkpoint 与
推理辅助函数，保证数据划分、随机种子与评价口径完全一致。

输入预处理按 config 走：论文模型用「原始波段 + 11x11 patch」（reducer=none），
HybridSN 用「PCA15 + 25x25 patch」，两者由 ``PreprocessingConfig`` 统一驱动。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("cannot locate 实验交付/pyproject.toml")


PROJECT_ROOT = find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.高光谱预处理 import HSIPreprocessingPipeline, PreprocessingConfig, load_hsi_data
from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)
from src.models.Paper3D1DCNN import Paper3DCNN, Paper3D1DCNN
from src.models.改进Paper3D1DCNN import ImprovedPaper3D1DCNN
from src.models.可配置HybridSN import ConfigurableHybridSN
from src.models.改进HybridSN import ImprovedHybridSN
from src.training.hybridsn_baseline import (
    benchmark_model_compute,
    build_all_labeled_loader,
    build_loaders,
    infer_loader,
    load_checkpoint,
    load_model_ready_artifact,
    save_checkpoint,
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

LOGGER = logging.getLogger("paper_reproduction")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行论文复现 / 多模型对比单次训练。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="新实验目录；默认在 experiments/ 下自动命名。")
    parser.add_argument("--epochs", type=int, help="调试用：覆盖 YAML 的 epoch 数。")
    parser.add_argument("--batch-size", type=int, help="调试用：覆盖 YAML 的 batch size。")
    parser.add_argument("--model", type=str, help="覆盖 YAML 的 model.name（用于同一预处理跑多个模型）。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true", help="只训练，不解封测试集。")
    parser.add_argument("--skip-all-labeled-map", action="store_true", help="跳过全图有标签像元推理图。")
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(value)


def configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def read_git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "status_short": run("status", "--short").splitlines(),
    }


def environment_record(
    *, device: torch.device, command: list[str], config_path: Path, model_ready_path: Path
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    record: dict[str, Any] = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "model_ready_artifact": {
            "path": str(model_ready_path),
            "sha256": sha256_file(model_ready_path),
        },
        "git": read_git_state(),
    }
    if cuda:
        properties = torch.cuda.get_device_properties(device)
        record["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
        }
    return record


def write_environment_files(output_dir: Path, values: Mapping[str, Any]) -> None:
    write_json(output_dir / "environment.json", values)
    lines = [
        f"recorded_at={values['recorded_at']}",
        f"command={' '.join(values['command'])}",
        f"platform={values['platform']}",
        f"python={values['python']}",
        f"numpy={values['numpy']}",
        f"torch={values['torch']}",
        f"cuda_available={values['cuda_available']}",
        f"torch_cuda_version={values['torch_cuda_version']}",
        f"selected_device={values['selected_device']}",
    ]
    if "gpu" in values:
        lines.append(f"gpu_name={values['gpu']['name']}")
        lines.append(f"gpu_total_memory_bytes={values['gpu']['total_memory_bytes']}")
    lines.append(f"config_sha256={values['config']['sha256']}")
    lines.append(f"model_ready_sha256={values['model_ready_artifact']['sha256']}")
    lines.append(f"git_commit={values['git']['commit']}")
    lines.append(f"git_branch={values['git']['branch']}")
    lines.append(f"git_dirty_files={len(values['git']['status_short'])}")
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_per_class_csv(path: Path, per_class: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)


def build_model(
    name: str,
    *,
    bands: int,
    patch_size: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """按名称构建模型。论文族用原始波段 + patch11，HybridSN 族用 PCA15 + patch25。"""
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"paper3dcnn", "3dcnn"}:
        return Paper3DCNN(bands, num_classes, dropout=dropout)
    if key in {"paper3d1dcnn", "3d1dcnn"}:
        return Paper3D1DCNN(bands, num_classes, dropout=dropout)
    if key in {"improvedpaper3d1dcnn", "paper3d1dcnnimproved"}:
        return ImprovedPaper3D1DCNN(bands, num_classes, dropout=dropout)
    if key in {"hybridsn", "originalhybridsn"}:
        # 默认架构即原始 HybridSN（8/16/32 通道、7/5/3 光谱核、256/128 全连接）。
        return ConfigurableHybridSN(input_bands=bands, patch_size=patch_size, num_classes=num_classes)
    if key in {"improvedhybridsn"}:
        return ImprovedHybridSN(input_bands=bands, patch_size=patch_size, num_classes=num_classes)
    raise ValueError(f"未知模型名称：{name}")


def ensure_model_ready(project_root: Path, config: PreprocessingConfig):
    """若模型就绪数据不存在，则先拟合预处理状态并生成 NPZ，再加载并返回。"""
    route_dir = project_root / "data/processed" / config.dataset_name / config.route_name()
    route_dir.mkdir(parents=True, exist_ok=True)
    state_path = route_dir / "preprocessing_state.npz"
    metadata_path = route_dir / "metadata.json"
    model_ready_path = route_dir / "model_ready_dataset.npz"

    if not model_ready_path.is_file():
        data = load_hsi_data(project_root, config)
        if state_path.is_file() and metadata_path.is_file():
            pipeline = HSIPreprocessingPipeline.load_state(state_path, metadata_path)
            if pipeline.config != config:
                raise ValueError("existing preprocessing state does not match the requested config")
            pipeline.attach_transformed_cube(data.cube)
        else:
            pipeline = HSIPreprocessingPipeline(config).fit(data)
            pipeline.save_state(route_dir, overwrite=True)
        temporary = route_dir / "model_ready_dataset.tmp.npz"
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("1.0"),
            dataset_name=np.asarray(config.dataset_name),
            split_protocol=np.asarray(config.split_protocol),
            split_seed=np.asarray(config.split_seed, dtype=np.int64),
            config_fingerprint=np.asarray(config.fingerprint()),
            transformed_cube=np.ascontiguousarray(pipeline.transformed_cube_, dtype=config.output_dtype),
            coordinates=np.ascontiguousarray(data.coordinates, dtype=np.int32),
            raw_labels=np.ascontiguousarray(data.labels, dtype=np.int16),
            train_indices=np.ascontiguousarray(data.indices_by_split["train"], dtype=np.int64),
            validation_indices=np.ascontiguousarray(data.indices_by_split["validation"], dtype=np.int64),
            test_indices=np.ascontiguousarray(data.indices_by_split["test"], dtype=np.int64),
            class_names=np.asarray(data.spec.class_names),
            patch_size=np.asarray(config.patch_size, dtype=np.int64),
            padding_mode=np.asarray(config.padding_mode),
            padding_value=np.asarray(config.padding_value, dtype=np.float32),
            num_classes=np.asarray(len(data.spec.class_names), dtype=np.int64),
        )
        temporary.replace(model_ready_path)
        LOGGER.info("已生成模型就绪数据：%s", model_ready_path)
    return load_model_ready_artifact(model_ready_path, config)


def build_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "run_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": "1.0", "files": files}


def write_experiment_record(
    path: Path,
    *,
    effective_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifact,
    history: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    performance: Mapping[str, Any],
    model_name: str,
    selected_checkpoint: Path,
) -> None:
    training = effective_config["training"]
    lines = [
        "# 论文复现 / 多模型对比实验记录",
        "",
        "> 本记录由正式运行脚本自动生成；评价数值只来自固定 test split。",
        "",
        "## 1. 实验目的",
        "",
        "在统一协议（fair24_6_70 + seed1442 + 统一指标）下训练并评价一个模型，用于论文复现与多模型 / 多数据集对比。",
        "",
        "## 2. 数据与预处理口径",
        "",
        f"- 数据集：{artifact.dataset_name}，预处理后影像 `{artifact.transformed_cube.shape[0]}×{artifact.transformed_cube.shape[1]}×{artifact.output_bands}`。",
        f"- 固定划分：`{artifact.split_protocol}`，seed={artifact.split_seed}；train={artifact.train_indices.size:,}，validation={artifact.validation_indices.size:,}，test={artifact.test_indices.size:,}。",
        f"- 输入：`N×1×{artifact.output_bands}×{artifact.patch_size}×{artifact.patch_size}`；类别数={artifact.num_classes}。",
        f"- 预处理指纹：`{artifact.config_fingerprint}`；模型就绪文件 SHA-256：`{environment['model_ready_artifact']['sha256']}`。",
        "",
        "## 3. 模型与训练设置",
        "",
        f"- 模型：`{model_name}`；可训练参数={performance['model']['trainable_parameters']:,}。",
        f"- 优化器：{training.get('optimizer', 'adam')}，learning rate={training['learning_rate']}，weight decay={training.get('weight_decay', 0.0)}。",
        f"- batch size={training['batch_size']}，epochs={len(history)}，训练 seed={effective_config['experiment']['seed']}。",
        f"- 设备：{environment.get('gpu', {}).get('name', environment['selected_device'])}；PyTorch {environment['torch']}。",
        f"- checkpoint 选择：`{selected_checkpoint.name}`（{performance['checkpoint_selection']}）。",
        "",
        "## 4. 结果与性能",
        "",
        f"- OA：{metrics['oa']:.4%}",
        f"- AA：{metrics['aa']:.4%}",
        f"- Cohen's Kappa：{metrics['kappa']:.6f}",
        f"- 测试损失：{metrics['test_loss']:.6f}",
        f"- 训练总计 {performance['training']['total_seconds']:.2f} s，平均每轮 {performance['training']['mean_epoch_seconds']:.2f} s。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    config_path = args.config.resolve()
    config_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config_values["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config_values["training"]["batch_size"] = args.batch_size
    if args.model is not None:
        config_values["model"]["name"] = args.model

    model_name = str(config_values["model"]["name"])
    epochs = int(config_values["training"]["epochs"])
    batch_size = int(config_values["training"]["batch_size"])
    training_seed = int(config_values["experiment"]["seed"])
    preprocessing_config = PreprocessingConfig.from_mapping(config_values)
    if training_seed != preprocessing_config.split_seed:
        raise ValueError("experiment.seed must equal dataset.split_seed under the unified-seed protocol")
    dropout = float(config_values["model"].get("dropout", 0.5))
    early_stop_patience = int(config_values["training"].get("early_stopping_patience", 0))
    early_stop_min_delta = float(config_values["training"].get("early_stopping_min_delta", 0.0))
    early_stop_metric = str(config_values["training"].get("early_stopping_metric", "accuracy")).strip().lower()
    if early_stop_metric not in {"accuracy", "loss"}:
        raise ValueError(f"unsupported early_stopping_metric: {early_stop_metric!r}")
    checkpoint_every = int(config_values["training"].get("checkpoint_every", 10))

    artifact = ensure_model_ready(PROJECT_ROOT, preprocessing_config)
    if artifact.num_classes != int(config_values["model"].get("num_classes", artifact.num_classes)):
        raise ValueError("model class count does not match the model-ready artifact")
    spatial_overlap = spatial_overlap_audit(
        artifact.image_shape,
        artifact.coordinates,
        artifact.raw_labels,
        artifact.train_indices,
        artifact.test_indices,
        patch_size=artifact.patch_size,
        class_names=artifact.class_names,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    default_name = (
        f"{artifact.dataset_name}__{model_name}__fair24_6_70__seed{training_seed}__{timestamp}"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / config_values["experiment"].get("output_root", "experiments") / default_name)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_logging(output_dir)

    effective_config_path = output_dir / "config.yaml"
    effective_config_path.write_text(
        yaml.safe_dump(config_values, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(config_path, output_dir / "config_source.yaml")
    write_json(output_dir / "spatial_overlap_audit.json", spatial_overlap)

    device = select_device(args.device)
    pin_memory = device.type == "cuda"
    seed_everything(training_seed)
    environment = environment_record(
        device=device,
        command=[sys.executable, *sys.argv],
        config_path=config_path,
        model_ready_path=artifact.path,
    )
    write_environment_files(output_dir, environment)

    loaders, loader_generator = build_loaders(
        artifact, batch_size=batch_size, loader_seed=training_seed,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    if loaders["train"] is None or loaders["test"] is None:
        raise RuntimeError("train and test splits must be non-empty")
    model = build_model(
        model_name,
        bands=artifact.output_bands,
        patch_size=artifact.patch_size,
        num_classes=artifact.num_classes,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = str(config_values["training"].get("optimizer", "adam")).lower()
    learning_rate = float(config_values["training"]["learning_rate"])
    weight_decay = float(config_values["training"].get("weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = float(config_values["training"].get("momentum", 0.0))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    history: list[dict[str, Any]] = []
    best_validation_accuracy = float("-inf")
    best_monitor_value = float("inf") if early_stop_metric == "loss" else float("-inf")
    selected_checkpoint = output_dir / "checkpoint_final.pt"
    selection_reason = "final epoch (no validation split)"
    early_stop_counter = 0

    LOGGER.info("输出目录：%s", output_dir)
    LOGGER.info("模型：%s | 设备：%s", model_name, environment.get("gpu", {}).get("name", str(device)))
    LOGGER.info(
        "数据：train=%d validation=%d test=%d，输入=1x%dx%dx%d，类别=%d",
        artifact.train_indices.size, artifact.validation_indices.size, artifact.test_indices.size,
        artifact.output_bands, artifact.patch_size, artifact.patch_size, artifact.num_classes,
    )
    LOGGER.info(
        "参数：%d；%s lr=%s batch=%d epochs=%d",
        trainable_parameters, optimizer_name, learning_rate, batch_size, epochs,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_wall_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, loaders["train"], criterion, optimizer, device, non_blocking=pin_memory)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(stats["loss"]),
            "train_accuracy": float(stats["accuracy"]),
            "epoch_seconds": float(stats["seconds"]),
            "train_samples_per_second": float(stats["samples_per_second"]),
        }
        improved = False
        if loaders["validation"] is not None:
            validation = infer_loader(model, loaders["validation"], device, criterion=criterion, non_blocking=pin_memory)
            record.update({"validation_loss": validation.loss, "validation_accuracy": validation.accuracy})
            # checkpoint 始终按验证集准确率选择（与 HybridSN 基线一致）。
            if validation.accuracy is not None and validation.accuracy > best_validation_accuracy:
                best_validation_accuracy = validation.accuracy
                selected_checkpoint = output_dir / "checkpoint_best.pt"
                selection_reason = "highest validation accuracy"
                save_checkpoint(
                    selected_checkpoint, epoch=epoch, model=model, optimizer=optimizer,
                    effective_config=config_values, preprocessing_fingerprint=artifact.config_fingerprint,
                    history=[*history, record], data_loader_generator=loader_generator, selected_by=selection_reason,
                )
            # 早停监视指标：loss 越小越好，accuracy 越大越好。
            monitor_value = validation.loss if early_stop_metric == "loss" else validation.accuracy
            if early_stop_metric == "loss":
                improved = monitor_value < best_monitor_value - early_stop_min_delta
            else:
                improved = monitor_value > best_monitor_value + early_stop_min_delta
            if improved:
                best_monitor_value = monitor_value
        history.append(record)
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)
        message = (
            f"epoch={epoch:03d}/{epochs} train_loss={record['train_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} seconds={record['epoch_seconds']:.2f}"
        )
        if "validation_accuracy" in record:
            message += f" val_loss={record['validation_loss']:.6f} val_acc={record['validation_accuracy']:.4f}"
        LOGGER.info(message)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir / "checkpoint_last.pt", epoch=epoch, model=model, optimizer=optimizer,
                effective_config=config_values, preprocessing_fingerprint=artifact.config_fingerprint,
                history=history, data_loader_generator=loader_generator, selected_by="recovery checkpoint",
            )
        if early_stop_patience > 0 and loaders["validation"] is not None:
            if improved:
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                LOGGER.info("早停触发：验证指标连续 %d 轮无改善，停止于 epoch %d。", early_stop_patience, epoch)
                break
    training_wall_seconds = time.perf_counter() - training_wall_started

    if not history:
        raise RuntimeError("no training history is available")
    save_checkpoint(
        output_dir / "checkpoint_final.pt", epoch=len(history), model=model, optimizer=optimizer,
        effective_config=config_values, preprocessing_fingerprint=artifact.config_fingerprint,
        history=history, data_loader_generator=loader_generator,
        selected_by="final epoch" if loaders["validation"] is None else "final epoch",
    )
    if loaders["validation"] is None:
        selected_checkpoint = output_dir / "checkpoint_final.pt"
    save_learning_curves(history, output_dir / "loss_curve.png")

    performance: dict[str, Any] = {
        "model": {
            "trainable_parameters": trainable_parameters,
            "parameter_bytes": parameter_bytes,
            "checkpoint_final_bytes": (output_dir / "checkpoint_final.pt").stat().st_size,
        },
        "training": {
            "epochs": len(history),
            "max_epochs_configured": epochs,
            "total_seconds": training_wall_seconds,
            "sum_epoch_seconds": float(sum(row["epoch_seconds"] for row in history)),
            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
            "mean_samples_per_second": float(np.mean([row["train_samples_per_second"] for row in history])),
            "peak_memory_allocated_bytes": (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
            "peak_memory_reserved_bytes": (int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None),
        },
        "checkpoint_selection": selection_reason,
    }

    if args.skip_test:
        performance["formal_test_evaluated"] = False
        write_json(output_dir / "performance.json", performance)
        write_json(output_dir / "run_manifest.json", build_manifest(output_dir))
        LOGGER.info("训练完成；按 --skip-test 要求未解封测试集。")
        return 0

    load_checkpoint(
        selected_checkpoint, model=model, optimizer=None, data_loader_generator=None,
        expected_preprocessing_fingerprint=artifact.config_fingerprint, device=device,
    )
    compute_benchmark = benchmark_model_compute(
        model, loaders["test"], device, non_blocking=pin_memory,
        warmup_iterations=args.benchmark_warmup, measured_iterations=args.benchmark_iterations,
    )
    test_output = infer_loader(model, loaders["test"], device, criterion=criterion, non_blocking=pin_memory)
    summary = classification_summary(test_output.labels, test_output.predictions, num_classes=artifact.num_classes)
    metrics = summary.to_dict(artifact.class_names)
    metrics.update(
        {
            "test_loss": test_output.loss,
            "test_samples": int(test_output.labels.size),
            "checkpoint": selected_checkpoint.name,
            "checkpoint_sha256": sha256_file(selected_checkpoint),
            "preprocessing_fingerprint": artifact.config_fingerprint,
            "test_set_used_for_model_selection": False,
        }
    )
    performance.update(
        {
            "formal_test_evaluated": True,
            "test_inference": {
                "elapsed_seconds": test_output.elapsed_seconds,
                "throughput_samples_per_second": test_output.throughput_samples_per_second,
                "milliseconds_per_sample": test_output.elapsed_seconds * 1000.0 / test_output.labels.size,
            },
            "model_compute_benchmark": compute_benchmark,
        }
    )

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "performance.json", performance)
    write_per_class_csv(output_dir / "per_class_accuracy.csv", metrics["per_class"])
    np.savez_compressed(
        output_dir / "predictions_test.npz",
        labels=test_output.labels.astype(np.int16),
        predictions=test_output.predictions.astype(np.int16),
        sample_indices=test_output.sample_indices.astype(np.int64),
        coordinates=test_output.coordinates.astype(np.int32),
    )
    save_confusion_matrix(summary.confusion_matrix, artifact.class_names, output_dir / "confusion_matrix.png")
    save_per_class_accuracy(summary.per_class_accuracy, artifact.class_names, output_dir / "per_class_accuracy.png")

    ground_truth = np.zeros(artifact.image_shape, dtype=np.int16)
    ground_truth[artifact.coordinates[:, 0], artifact.coordinates[:, 1]] = artifact.raw_labels
    test_map = build_classification_map(
        artifact.image_shape, artifact.coordinates, test_output.sample_indices, test_output.predictions
    )
    if args.skip_all_labeled_map:
        all_labeled_map = test_map.copy()
    else:
        all_loader = build_all_labeled_loader(artifact, batch_size=batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        all_output = infer_loader(model, all_loader, device, criterion=None, non_blocking=pin_memory)
        all_labeled_map = build_classification_map(
            artifact.image_shape, artifact.coordinates, all_output.sample_indices, all_output.predictions
        )
    np.savez_compressed(
        output_dir / "classification_maps.npz",
        ground_truth=ground_truth,
        test_predictions=test_map,
        all_labeled_predictions=all_labeled_map,
    )
    save_classification_maps(ground_truth, test_map, all_labeled_map, artifact.class_names, output_dir / "classification_map.png")

    # 供汇总脚本使用的紧凑单行结果。
    summary_row = {
        "dataset": artifact.dataset_name,
        "model": model_name,
        "num_classes": artifact.num_classes,
        "output_bands": artifact.output_bands,
        "patch_size": artifact.patch_size,
        "trainable_parameters": trainable_parameters,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs_configured": epochs,
        "epochs_trained": len(history),
        "training_total_seconds": training_wall_seconds,
        "mean_epoch_seconds": performance["training"]["mean_epoch_seconds"],
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "test_samples": metrics["test_samples"],
        "checkpoint": str(selected_checkpoint),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary_row.json", summary_row)

    write_experiment_record(
        output_dir / "实验记录.md",
        effective_config=config_values,
        environment=environment,
        artifact=artifact,
        history=history,
        metrics=metrics,
        performance=performance,
        model_name=model_name,
        selected_checkpoint=selected_checkpoint,
    )
    write_json(output_dir / "run_manifest.json", build_manifest(output_dir))
    LOGGER.info("测试完成：OA=%.4f AA=%.4f Kappa=%.4f", metrics["oa"], metrics["aa"], metrics["kappa"])
    LOGGER.info("结果目录：%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
