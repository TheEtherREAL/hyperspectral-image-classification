"""Train, infer, evaluate and visualize the reproducible HybridSN baseline."""

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

from src.datasets.高光谱预处理 import PreprocessingConfig
from src.evaluation.classification_metrics import (
    build_classification_map,
    classification_summary,
    spatial_overlap_audit,
)
from src.models.HybridSN模型 import HybridSN
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


LOGGER = logging.getLogger("hybridsn_baseline")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行可复现的 Pavia University HybridSN baseline。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/模型训练/HybridSN_Pavia论文复现基线.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="新实验目录；默认按协议在 experiments/ 下创建。",
    )
    parser.add_argument("--epochs", type=int, help="仅用于调试；覆盖 YAML 的 epoch 数。")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="从 checkpoint_last.pt 恢复。")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="只训练，不解封测试集；调试运行推荐使用。",
    )
    parser.add_argument(
        "--skip-all-labeled-map",
        action="store_true",
        help="跳过全部有标签像元推理图。",
    )
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


def relative_or_absolute(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


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
    *,
    device: torch.device,
    command: list[str],
    config_path: Path,
    model_ready_path: Path,
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
        "deterministic_algorithms": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "config": {
            "path": relative_or_absolute(config_path),
            "sha256": sha256_file(config_path),
        },
        "model_ready_artifact": {
            "path": relative_or_absolute(model_ready_path),
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
        lines.extend(
            [
                f"gpu_name={values['gpu']['name']}",
                f"gpu_compute_capability={values['gpu']['compute_capability']}",
                f"gpu_total_memory_bytes={values['gpu']['total_memory_bytes']}",
            ]
        )
    lines.extend(
        [
            f"config_sha256={values['config']['sha256']}",
            f"model_ready_sha256={values['model_ready_artifact']['sha256']}",
            f"git_commit={values['git']['commit']}",
            f"git_branch={values['git']['branch']}",
            f"git_dirty_files={len(values['git']['status_short'])}",
        ]
    )
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = list(history[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_per_class_csv(path: Path, per_class: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)


def write_experiment_record(
    path: Path,
    *,
    effective_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifact: Any,
    history: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    performance: Mapping[str, Any],
    spatial_overlap: Mapping[str, Any],
    selected_checkpoint: Path,
) -> None:
    training = effective_config["training"]
    lines = [
        "# HybridSN baseline 实验记录",
        "",
        "> 本记录由正式运行脚本自动生成；评价数值只来自固定 test split。",
        "",
        "## 1. 实验目的",
        "",
        "复用冻结的 Pavia University 预处理产物，完成 HybridSN 的训练、推理、准确性评价、性能统计与可视化，为后续模型优化和公平对比提供统一基线。",
        "",
        "## 2. 数据与预处理口径",
        "",
        f"- 数据集：{artifact.dataset_name}，影像尺寸 `{artifact.transformed_cube.shape[0]}×{artifact.transformed_cube.shape[1]}×{artifact.transformed_cube.shape[2]}`（PCA 后）。",
        f"- 固定划分：`{artifact.split_protocol}`，split seed={artifact.split_seed}；train={artifact.train_indices.size:,}，validation={artifact.validation_indices.size:,}，test={artifact.test_indices.size:,}。",
        f"- 输入：`N×1×{artifact.output_bands}×{artifact.patch_size}×{artifact.patch_size}`；类别数={artifact.num_classes}。",
        f"- 预处理指纹：`{artifact.config_fingerprint}`；模型就绪文件 SHA-256：`{environment['model_ready_artifact']['sha256']}`。",
        "- 标准化与 PCA 参数只由训练中心像元拟合；测试集未参与预处理拟合或 epoch 选择。",
        "",
        "## 3. 模型与训练设置",
        "",
        "HybridSN 依次采用 3 个 3D 卷积层提取谱空联合特征，经 3D→2D reshape 后使用 2D 卷积和 256→128→9 全连接分类头。模型输出 logits，损失为交叉熵。",
        "",
        f"- 优化器：Adam，learning rate={training['learning_rate']}，weight decay={training['weight_decay']}。",
        f"- batch size={training['batch_size']}，epochs={training['epochs']}，训练 seed={effective_config['experiment']['seed']}。",
        f"- 参数量：{performance['model']['trainable_parameters']:,}；参数内存（float32）：{performance['model']['parameter_bytes'] / 1024**2:.2f} MiB。",
        f"- 设备：{environment.get('gpu', {}).get('name', environment['selected_device'])}；PyTorch {environment['torch']}。",
        f"- checkpoint 选择：`{selected_checkpoint.name}`（{performance['checkpoint_selection']}）。",
        "",
        "## 4. 实验过程记录",
        "",
        f"训练共完成 {len(history)} 个 epoch。首轮 train loss={history[0]['train_loss']:.6f}、train accuracy={history[0]['train_accuracy']:.4%}；末轮 train loss={history[-1]['train_loss']:.6f}、train accuracy={history[-1]['train_accuracy']:.4%}。",
        f"训练总计 {performance['training']['total_seconds']:.2f} s，平均每轮 {performance['training']['mean_epoch_seconds']:.2f} s。训练日志、逐轮 CSV/JSON、checkpoint 与环境记录均保存在本目录。",
        "",
        "## 5. 结果与性能",
        "",
        f"- OA：{metrics['oa']:.4%}",
        f"- AA：{metrics['aa']:.4%}",
        f"- Cohen's Kappa：{metrics['kappa']:.6f}",
        f"- 测试损失：{metrics['test_loss']:.6f}",
        f"- 端到端测试推理：{performance['test_inference']['elapsed_seconds']:.3f} s，{performance['test_inference']['throughput_samples_per_second']:.1f} samples/s。",
        f"- 纯模型批推理：{performance['model_compute_benchmark']['milliseconds_per_batch']:.3f} ms/batch，{performance['model_compute_benchmark']['throughput_samples_per_second']:.1f} samples/s。",
        "",
        "| 类别 | 测试样本 | 准确率 |",
        "|---|---:|---:|",
    ]
    for item in metrics["per_class"]:
        lines.append(
            f"| {item['class_name']} | {item['support']:,} | {item['accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
        "## 6. 可视化与误差分析入口",
            "",
            "- `loss_curve.png`：训练损失与准确率随 epoch 的变化；",
            "- `confusion_matrix.png`：固定测试集混淆矩阵；",
            "- `per_class_accuracy.png`：类别准确率，优先检查低准确率类别；",
            "- `classification_map.png`：标签图、仅测试预测、全部有标签像元预测的空间对照。",
            "",
            "误差分析应结合混淆矩阵与分类图，区分光谱相似类别混淆、地物边界混合像元、类别样本不均衡和空间邻域污染，不使用训练准确率替代测试结论。",
            "",
            "## 7. 结果解释限制：随机像元划分的空间重叠",
            "",
            f"25×25 patch 下，{spatial_overlap['any_training_center_in_query_patch']['fraction_with_at_least_one']:.2%} 的测试 patch 内至少包含一个训练中心像元；{spatial_overlap['same_class_training_center_in_query_patch']['fraction_with_at_least_one']:.2%} 至少包含一个同类训练中心像元。",
            "这不表示标准化、PCA 或训练过程读取了测试标签，但说明随机像元划分不能视为严格的跨区域泛化测试。当前高精度只作为论文/课程兼容口径 baseline；后续应增加空间块划分对照。",
            "",
            "## 8. 可复现与后续对比",
            "",
            "后续改进模型必须复用相同的预处理指纹、split 文件与测试评价函数；调参应切换到带 validation 的公平比较配置，最终只在方案冻结后解封 test。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def main() -> int:
    args = parse_arguments()
    config_path = args.config.resolve()
    config_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config_values["training"]["epochs"] = args.epochs
    preprocessing_config = PreprocessingConfig.from_mapping(config_values)
    epochs = int(config_values["training"]["epochs"])
    batch_size = int(config_values["training"]["batch_size"])
    training_seed = int(config_values["experiment"]["seed"])
    if training_seed != preprocessing_config.split_seed:
        raise ValueError(
            "experiment.seed must equal dataset.split_seed under the unified-seed protocol"
        )
    checkpoint_every = int(config_values["training"].get("checkpoint_every", 10))
    if checkpoint_every < 1:
        raise ValueError("training.checkpoint_every must be positive")

    route_dir = (
        PROJECT_ROOT
        / "data/processed"
        / preprocessing_config.dataset_name
        / preprocessing_config.route_name()
    )
    model_ready_path = route_dir / "model_ready_dataset.npz"
    artifact = load_model_ready_artifact(model_ready_path, preprocessing_config)
    if artifact.num_classes != int(config_values["model"]["num_classes"]):
        raise ValueError("model class count does not match the model-ready artifact")
    if (artifact.output_bands, artifact.patch_size) != (15, 25):
        raise ValueError("the fixed HybridSN baseline requires PCA15 and patch25")
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
    default_name = f"{artifact.dataset_name}__hybridsn__seed{training_seed}__{timestamp}"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / config_values["experiment"]["output_root"] / default_name)
    )
    if args.resume is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    configure_logging(output_dir)

    effective_config_path = output_dir / "config.yaml"
    effective_config_path.write_text(
        yaml.safe_dump(config_values, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
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
        model_ready_path=model_ready_path,
    )
    write_environment_files(output_dir, environment)

    loaders, loader_generator = build_loaders(
        artifact,
        batch_size=batch_size,
        loader_seed=training_seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    if loaders["train"] is None or loaders["test"] is None:
        raise RuntimeError("the baseline requires non-empty train and test splits")
    model = HybridSN(
        num_classes=artifact.num_classes,
        dropout=float(config_values["model"]["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config_values["training"]["learning_rate"]),
        weight_decay=float(config_values["training"]["weight_decay"]),
    )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if trainable_parameters != 4_844_793:
        raise AssertionError(f"unexpected HybridSN parameter count: {trainable_parameters}")

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation_accuracy = float("-inf")
    selected_checkpoint = output_dir / "checkpoint_final.pt"
    selection_reason = "fixed final epoch (paper30 has no validation split)"
    if args.resume is not None:
        resumed = load_checkpoint(
            args.resume.resolve(),
            model=model,
            optimizer=optimizer,
            data_loader_generator=loader_generator,
            expected_preprocessing_fingerprint=artifact.config_fingerprint,
            device=device,
        )
        history = list(resumed["history"])
        start_epoch = int(resumed["epoch"]) + 1
        LOGGER.info("恢复 checkpoint：%s，从 epoch %d 继续", args.resume, start_epoch)

    LOGGER.info("输出目录：%s", output_dir)
    LOGGER.info("设备：%s", environment.get("gpu", {}).get("name", str(device)))
    LOGGER.info(
        "数据：train=%d validation=%d test=%d，输入=1x%dx%dx%d",
        artifact.train_indices.size,
        artifact.validation_indices.size,
        artifact.test_indices.size,
        artifact.output_bands,
        artifact.patch_size,
        artifact.patch_size,
    )
    LOGGER.info(
        "模型参数：%d；Adam lr=%s；batch=%d；epochs=%d",
        trainable_parameters,
        config_values["training"]["learning_rate"],
        batch_size,
        epochs,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_wall_started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        stats = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            non_blocking=pin_memory,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(stats["loss"]),
            "train_accuracy": float(stats["accuracy"]),
            "epoch_seconds": float(stats["seconds"]),
            "train_samples_per_second": float(stats["samples_per_second"]),
        }
        if loaders["validation"] is not None:
            validation = infer_loader(
                model,
                loaders["validation"],
                device,
                criterion=criterion,
                non_blocking=pin_memory,
            )
            record.update(
                {
                    "validation_loss": validation.loss,
                    "validation_accuracy": validation.accuracy,
                }
            )
            if validation.accuracy is not None and validation.accuracy > best_validation_accuracy:
                best_validation_accuracy = validation.accuracy
                selected_checkpoint = output_dir / "checkpoint_best.pt"
                selection_reason = "highest validation accuracy"
                save_checkpoint(
                    selected_checkpoint,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    effective_config=config_values,
                    preprocessing_fingerprint=artifact.config_fingerprint,
                    history=[*history, record],
                    data_loader_generator=loader_generator,
                    selected_by=selection_reason,
                )
        history.append(record)
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)
        message = (
            f"epoch={epoch:03d}/{epochs} train_loss={record['train_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"seconds={record['epoch_seconds']:.2f}"
        )
        if "validation_accuracy" in record:
            message += (
                f" val_loss={record['validation_loss']:.6f} "
                f"val_acc={record['validation_accuracy']:.4f}"
            )
        LOGGER.info(message)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                effective_config=config_values,
                preprocessing_fingerprint=artifact.config_fingerprint,
                history=history,
                data_loader_generator=loader_generator,
                selected_by="recovery checkpoint",
            )
    training_wall_seconds = time.perf_counter() - training_wall_started

    if not history:
        raise RuntimeError("no training history is available")
    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        effective_config=config_values,
        preprocessing_fingerprint=artifact.config_fingerprint,
        history=history,
        data_loader_generator=loader_generator,
        selected_by="fixed final epoch" if loaders["validation"] is None else "final epoch",
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
            "total_seconds": training_wall_seconds,
            "sum_epoch_seconds": float(sum(row["epoch_seconds"] for row in history)),
            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
            "mean_samples_per_second": float(
                np.mean([row["train_samples_per_second"] for row in history])
            ),
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
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
        selected_checkpoint,
        model=model,
        optimizer=None,
        data_loader_generator=None,
        expected_preprocessing_fingerprint=artifact.config_fingerprint,
        device=device,
    )
    compute_benchmark = benchmark_model_compute(
        model,
        loaders["test"],
        device,
        non_blocking=pin_memory,
        warmup_iterations=args.benchmark_warmup,
        measured_iterations=args.benchmark_iterations,
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
                "milliseconds_per_sample": (
                    test_output.elapsed_seconds * 1000.0 / test_output.labels.size
                ),
                "peak_memory_allocated_bytes": test_output.peak_memory_allocated_bytes,
                "peak_memory_reserved_bytes": test_output.peak_memory_reserved_bytes,
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
    save_confusion_matrix(
        summary.confusion_matrix,
        artifact.class_names,
        output_dir / "confusion_matrix.png",
    )
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
    if args.skip_all_labeled_map:
        all_labeled_map = test_map.copy()
        LOGGER.info("按 --skip-all-labeled-map 要求，分类图第三幅复用测试预测。")
    else:
        all_loader = build_all_labeled_loader(
            artifact,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        all_output = infer_loader(
            model,
            all_loader,
            device,
            criterion=None,
            non_blocking=pin_memory,
        )
        all_labeled_map = build_classification_map(
            artifact.image_shape,
            artifact.coordinates,
            all_output.sample_indices,
            all_output.predictions,
        )
        performance["all_labeled_inference"] = {
            "samples": int(all_output.predictions.size),
            "elapsed_seconds": all_output.elapsed_seconds,
            "throughput_samples_per_second": all_output.throughput_samples_per_second,
        }
        write_json(output_dir / "performance.json", performance)
    np.savez_compressed(
        output_dir / "classification_maps.npz",
        ground_truth=ground_truth,
        test_predictions=test_map,
        all_labeled_predictions=all_labeled_map,
    )
    save_classification_maps(
        ground_truth,
        test_map,
        all_labeled_map,
        artifact.class_names,
        output_dir / "classification_map.png",
    )
    write_experiment_record(
        output_dir / "实验记录.md",
        effective_config=config_values,
        environment=environment,
        artifact=artifact,
        history=history,
        metrics=metrics,
        performance=performance,
        spatial_overlap=spatial_overlap,
        selected_checkpoint=selected_checkpoint,
    )
    write_json(output_dir / "run_manifest.json", build_manifest(output_dir))
    LOGGER.info(
        "测试完成：OA=%.4f AA=%.4f Kappa=%.4f，吞吐率=%.1f samples/s",
        metrics["oa"],
        metrics["aa"],
        metrics["kappa"],
        performance["test_inference"]["throughput_samples_per_second"],
    )
    LOGGER.info("结果目录：%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
