"""高光谱数据预处理主程序 / HSI preprocessing workflow.

本文件是面向使用者的只读运行入口。它把已有数据接口按七个步骤组织起来，
但不重新生成划分、不重新拟合或覆盖预处理参数，也不进行模型训练或测试集评价。

This is the user-facing, read-only entry point. It organizes existing data APIs
into seven steps without regenerating splits, refitting/overwriting preprocessing
states, training a model, or evaluating the test set.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

# 允许从项目根目录的上一层直接运行本文件。
# Allow direct execution even when the current directory is above the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.高光谱预处理 import (
    HSIDataBundle,
    HSIPreprocessingPipeline,
    HSITensorDataset,
    PreprocessingConfig,
    load_hsi_data,
)


DEFAULT_CONFIG = Path(
    "configs/数据预处理/Pavia数据预处理.yaml"
)
SPLIT_NAMES = ("train", "validation", "test")
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


@dataclass(frozen=True)
class WorkflowSettings:
    """运行参数 / Runtime settings shared by Python and Notebook entry points."""

    project_root: Path
    config_path: Path
    config_values: dict[str, Any]
    config: PreprocessingConfig
    state_dir: Path
    batch_size: int
    loader_seed: int
    num_workers: int
    pin_memory: bool


def announce_step(
    number: int,
    title_zh: str,
    title_en: str,
    *,
    purpose: str,
    inputs: str,
    outputs: str,
    acceptance: str,
) -> None:
    """在执行前说明目的、输入、输出和验收 / Explain a step before execution."""

    print("\n" + "=" * 78)
    print(f"[步骤 {number}/7 | Step {number}/7] {title_zh} / {title_en}")
    print(f"目的 / Purpose: {purpose}")
    print(f"输入 / Input: {inputs}")
    print(f"输出 / Output: {outputs}")
    print(f"验收 / Acceptance: {acceptance}")


def _resolve_path(project_root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _pin_memory_from_value(value: str | bool | None) -> bool:
    if value is None or value == "auto":
        return torch.cuda.is_available()
    if isinstance(value, bool):
        return value
    return value.lower() == "true"


def load_workflow_settings(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    project_root: Path = PROJECT_ROOT,
    state_dir: Path | str | None = None,
    batch_size: int | None = None,
    loader_seed: int | None = None,
    num_workers: int | None = None,
    pin_memory: str | bool | None = "auto",
    announce: bool = True,
) -> WorkflowSettings:
    """步骤1：加载并校验配置 / Step 1: load and validate configuration."""

    if announce:
        announce_step(
            1,
            "加载配置",
            "Load configuration",
            purpose="确定唯一的数据集、固定划分、光谱处理、空间表示和 DataLoader 参数。",
            inputs=f"YAML 配置文件 {config_path}",
            outputs="经过类型和取值校验的 WorkflowSettings。",
            acceptance="协议 seed=1442，参数有效，且能定位唯一的冻结状态目录。",
        )
    root = Path(project_root).resolve()
    resolved_config = _resolve_path(root, config_path)
    values = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("preprocessing config must contain a mapping")
    config = PreprocessingConfig.from_mapping(values)
    loader_values = dict(values.get("dataloader", {}))

    resolved_batch_size = int(
        batch_size if batch_size is not None else loader_values.get("batch_size", 256)
    )
    resolved_loader_seed = int(
        loader_seed if loader_seed is not None else loader_values.get("loader_seed", 1442)
    )
    resolved_num_workers = int(
        num_workers if num_workers is not None else loader_values.get("num_workers", 0)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if resolved_num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if resolved_loader_seed != config.split_seed:
        raise ValueError(
            "loader_seed must equal split_seed under the unified-seed experiment protocol"
        )

    if pin_memory == "auto" and "pin_memory" in loader_values:
        resolved_pin_memory = bool(loader_values["pin_memory"]) and torch.cuda.is_available()
    else:
        resolved_pin_memory = _pin_memory_from_value(pin_memory)

    if state_dir is None:
        resolved_state_dir = (
            root / "data" / "processed" / config.dataset_name / config.route_name()
        )
    else:
        resolved_state_dir = _resolve_path(root, state_dir)

    settings = WorkflowSettings(
        project_root=root,
        config_path=resolved_config,
        config_values=dict(values),
        config=config,
        state_dir=resolved_state_dir,
        batch_size=resolved_batch_size,
        loader_seed=resolved_loader_seed,
        num_workers=resolved_num_workers,
        pin_memory=resolved_pin_memory,
    )
    print(f"配置 / Config: {settings.config_path}")
    print(f"路线 / Route: {config.route_name()}")
    print(
        "光谱与空间 / Spectral & spatial: "
        f"{config.standardization} + {config.reducer}{config.n_components or ''} + "
        f"{config.representation}{config.patch_size if config.representation == 'patch' else ''}"
    )
    print(f"冻结状态目录 / Frozen state: {settings.state_dir}")
    print("步骤 1 验收通过 / Step 1 accepted")
    return settings


def load_fixed_dataset(
    settings: WorkflowSettings,
    *,
    announce: bool = True,
) -> HSIDataBundle:
    """步骤2：读取原始立方体与固定划分 / Step 2: load cube and frozen split."""

    if announce:
        announce_step(
            2,
            "读取数据与固定划分",
            "Load data and frozen split",
            purpose="建立每个有标签像元的光谱、二维坐标、标签和 split 身份。",
            inputs="原始 .mat 文件，以及 seed=1442 的既有 .npz/.json 划分。",
            outputs="HSIDataBundle；不生成新划分。",
            acceptance="坐标与标签图一致，三组覆盖全部有标签像元且互不重叠。",
        )
    data = load_hsi_data(settings.project_root, settings.config)
    counts = {name: int(values.size) for name, values in data.indices_by_split.items()}
    print(f"数据立方体 / Cube: {tuple(data.cube.shape)} dtype={data.cube.dtype}")
    print(f"标签图 / Label map: {tuple(data.label_map.shape)}")
    print(
        "样本数 / Samples: "
        f"train={counts['train']}, validation={counts['validation']}, test={counts['test']}"
    )
    print("\n类别构成 / Class composition")
    print("ID | English class | 中文类别 | train | val | test")
    for class_id, class_name in enumerate(data.spec.class_names, start=1):
        chinese = (
            PAVIA_CLASS_NAMES_ZH[class_id - 1]
            if data.spec.name == "pavia_university"
            else "待补充"
        )
        class_counts = []
        for split_name in SPLIT_NAMES:
            indices = data.indices_by_split[split_name]
            class_counts.append(int(np.count_nonzero(data.labels[indices] == class_id)))
        print(
            f"{class_id:>2} | {class_name:<22} | {chinese:<8} | "
            f"{class_counts[0]:>5} | {class_counts[1]:>3} | {class_counts[2]:>5}"
        )
    print("步骤 2 验收通过 / Step 2 accepted")
    return data


def verify_frozen_split_relationships(
    settings: WorkflowSettings,
    data: HSIDataBundle,
    *,
    announce: bool = True,
) -> dict[str, Any]:
    """核验当前 split 与两协议关系 / Verify frozen split and protocol relations."""

    if announce:
        print("\n" + "-" * 78)
        print("[检查 2C | Check 2C] 固定划分约束 / Frozen split contract")
        print("目的 / Purpose: 显式确认无交集、全覆盖、类别覆盖和两协议共享关系。")
        print("输入 / Input: 当前 HSIDataBundle 与两套既有 seed=1442 split 文件。")
        print("输出 / Output: 当前协议和跨协议的布尔验收结果；不生成新划分。")
        print("验收 / Acceptance: 所有约束均为 True。")

    assigned = np.concatenate(
        [data.indices_by_split[split_name] for split_name in SPLIT_NAMES]
    )
    current_disjoint = np.unique(assigned).size == assigned.size
    current_full_coverage = np.array_equal(
        np.sort(assigned), np.arange(data.labels.size, dtype=np.int64)
    )
    expected_classes = np.arange(1, len(data.spec.class_names) + 1)
    current_class_coverage = all(
        indices.size == 0
        or np.array_equal(np.unique(data.labels[indices]), expected_classes)
        for indices in data.indices_by_split.values()
    )
    rows, columns = data.coordinates.T
    current_coordinate_label_alignment = np.array_equal(
        data.labels, data.label_map[rows, columns]
    )

    protocol_arrays: dict[str, dict[str, np.ndarray]] = {}
    for protocol_name in ("paper30", "fair24_6_70"):
        split_path = (
            settings.project_root
            / "data"
            / "splits"
            / (
                f"{settings.config.dataset_name}__{protocol_name}__"
                f"seed{settings.config.split_seed}.npz"
            )
        )
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        with np.load(split_path, allow_pickle=False) as artifact:
            protocol_arrays[protocol_name] = {
                name: artifact[name].copy()
                for name in (
                    "coordinates",
                    "labels",
                    "train_indices",
                    "validation_indices",
                    "test_indices",
                )
            }

    paper = protocol_arrays["paper30"]
    fair = protocol_arrays["fair24_6_70"]
    same_sample_identity = np.array_equal(
        paper["coordinates"], fair["coordinates"]
    ) and np.array_equal(paper["labels"], fair["labels"])
    shared_test_set = np.array_equal(paper["test_indices"], fair["test_indices"])
    shared_training_pool = np.array_equal(
        paper["train_indices"],
        np.sort(
            np.concatenate((fair["train_indices"], fair["validation_indices"]))
        ),
    )

    result = {
        "current_protocol": settings.config.split_protocol,
        "current_disjoint": bool(current_disjoint),
        "current_full_coverage": bool(current_full_coverage),
        "current_class_coverage": bool(current_class_coverage),
        "current_coordinate_label_alignment": bool(current_coordinate_label_alignment),
        "same_sample_identity": bool(same_sample_identity),
        "shared_test_set": bool(shared_test_set),
        "fair_train_plus_validation_equals_paper_train": bool(shared_training_pool),
    }
    failed = [name for name, value in result.items() if name != "current_protocol" and not value]
    if failed:
        raise AssertionError(f"frozen split relationship checks failed: {failed}")
    for name, value in result.items():
        print(f"{name}={value}")
    print("固定划分关系验收通过 / Frozen split relationships accepted")
    return result


def load_frozen_preprocessing_state(
    settings: WorkflowSettings,
    data: HSIDataBundle,
    *,
    announce: bool = True,
) -> HSIPreprocessingPipeline:
    """步骤3：加载冻结参数 / Step 3: load the frozen preprocessing state."""

    if announce:
        announce_step(
            3,
            "加载冻结预处理参数",
            "Load frozen preprocessing state",
            purpose="复用只在训练中心像元上拟合的标准化与光谱降维参数，防止数据泄漏。",
            inputs="preprocessing_state.npz、metadata.json 和当前 YAML 指纹。",
            outputs="已拟合但尚未附加完整变换立方体的 HSIPreprocessingPipeline。",
            acceptance="配置指纹相同、训练样本数相同，且元数据声明验证/测试未参与拟合。",
        )
    state_path = settings.state_dir / "preprocessing_state.npz"
    metadata_path = settings.state_dir / "metadata.json"
    if not state_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "frozen preprocessing files are missing; do not refit automatically: "
            f"{settings.state_dir}"
        )
    pipeline = HSIPreprocessingPipeline.load_state(state_path, metadata_path)
    if pipeline.config.fingerprint() != settings.config.fingerprint():
        raise AssertionError("YAML config and frozen preprocessing state do not match")
    metadata = pipeline.fit_metadata_ or {}
    fit_scope = metadata.get("fit_scope", {})
    if int(fit_scope.get("training_samples", -1)) != int(data.train_indices.size):
        raise AssertionError("frozen state training sample count does not match the split")
    if fit_scope.get("validation_and_test_used_for_fit") is not False:
        raise AssertionError("frozen state does not prove train-only fitting")
    print(f"状态文件 / State file: {state_path}")
    print(f"拟合样本 / Fit samples: {fit_scope['training_samples']} (train only)")
    print("验证/测试参与拟合 / Validation/test used for fit: False")
    print("步骤 3 验收通过 / Step 3 accepted")
    return pipeline


def transform_full_cube(
    settings: WorkflowSettings,
    data: HSIDataBundle,
    pipeline: HSIPreprocessingPipeline,
    *,
    announce: bool = True,
) -> np.ndarray:
    """步骤4：应用冻结变换 / Step 4: apply the frozen transformations."""

    if announce:
        announce_step(
            4,
            "变换完整数据立方体",
            "Transform the full data cube",
            purpose="用冻结的训练集参数把 103 波段映射为统一的模型输入特征。",
            inputs=f"原始立方体 {tuple(data.cube.shape)} 与冻结标准化/降维参数。",
            outputs="内存中的变换后立方体；不会把全部 patch 写入磁盘。",
            acceptance="空间尺寸不变，输出波段数符合配置，dtype 正确且无 NaN/Inf。",
        )
    transformed = pipeline.attach_transformed_cube(data.cube)
    expected_shape = (
        data.cube.shape[0],
        data.cube.shape[1],
        pipeline.output_bands,
    )
    if transformed.shape != expected_shape:
        raise AssertionError(f"expected transformed shape {expected_shape}, got {transformed.shape}")
    if transformed.dtype != np.dtype(settings.config.output_dtype):
        raise AssertionError("transformed cube dtype does not match the config")
    if not np.isfinite(transformed).all():
        raise AssertionError("transformed cube contains NaN or infinite values")
    print(f"变换后立方体 / Transformed cube: {transformed.shape} dtype={transformed.dtype}")
    print("全部数值有限 / All values finite: True")
    print("步骤 4 验收通过 / Step 4 accepted")
    return transformed


def build_torch_datasets(
    data: HSIDataBundle,
    pipeline: HSIPreprocessingPipeline,
    *,
    announce: bool = True,
) -> dict[str, HSITensorDataset | None]:
    """步骤5：构造 Dataset / Step 5: build PyTorch Dataset objects."""

    if announce:
        announce_step(
            5,
            "构造 PyTorch Dataset",
            "Build PyTorch datasets",
            purpose="按固定中心坐标动态提取 pixel 或 patch，并保持样本身份可追溯。",
            inputs="变换后立方体、固定 split 坐标、原始标签和 sample_index。",
            outputs="train/validation/test 三个 Dataset（空 split 返回 None）。",
            acceptance="长度与固定划分一致；input、label、raw_label、coordinate、sample_index 对齐。",
        )
    datasets = pipeline.build_torch_datasets(data)
    for split_name in SPLIT_NAMES:
        dataset = datasets[split_name]
        expected = int(data.indices_by_split[split_name].size)
        actual = 0 if dataset is None else len(dataset)
        if actual != expected:
            raise AssertionError(f"{split_name} Dataset length mismatch: {actual} != {expected}")
        print(f"{split_name:<10} Dataset: {actual} samples")

    train_sample = datasets["train"][0]
    if int(train_sample["raw_label"]) != int(train_sample["label"]) + 1:
        raise AssertionError("raw/model label mapping is invalid")
    print(f"首个训练输入 / First train input: {tuple(train_sample['input'].shape)}")
    print(
        "身份字段 / Identity: "
        f"sample_index={int(train_sample['sample_index'])}, "
        f"coordinate={tuple(int(v) for v in train_sample['coordinate'])}, "
        f"raw_label={int(train_sample['raw_label'])}, model_label={int(train_sample['label'])}"
    )
    print("步骤 5 验收通过 / Step 5 accepted")
    return datasets


def build_torch_dataloaders(
    settings: WorkflowSettings,
    data: HSIDataBundle,
    pipeline: HSIPreprocessingPipeline,
    *,
    announce: bool = True,
):
    """步骤6：构造 DataLoader / Step 6: build reproducible DataLoaders."""

    if announce:
        announce_step(
            6,
            "构造 PyTorch DataLoader",
            "Build PyTorch DataLoaders",
            purpose="按 batch 输出张量；训练集可复现打乱，验证/测试保持固定顺序。",
            inputs="三个 Dataset，以及 batch_size、loader_seed、num_workers、pin_memory。",
            outputs="可直接交给后续模型训练/验证代码的 DataLoader 字典。",
            acceptance="批次数计算正确，drop_last=False，所有样本均能被遍历。",
        )
    loaders = pipeline.build_torch_loaders(
        data,
        batch_size=settings.batch_size,
        loader_seed=settings.loader_seed,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )
    print(
        "加载参数 / Loader settings: "
        f"batch_size={settings.batch_size}, seed={settings.loader_seed}, "
        f"num_workers={settings.num_workers}, pin_memory={settings.pin_memory}"
    )
    for split_name in SPLIT_NAMES:
        loader = loaders[split_name]
        samples = int(data.indices_by_split[split_name].size)
        expected_batches = math.ceil(samples / settings.batch_size) if samples else 0
        actual_batches = 0 if loader is None else len(loader)
        if actual_batches != expected_batches:
            raise AssertionError(f"{split_name} DataLoader batch count mismatch")
        print(f"{split_name:<10} DataLoader: {actual_batches} batches")
    print("步骤 6 验收通过 / Step 6 accepted")
    return loaders


def inspect_first_training_batch(
    settings: WorkflowSettings,
    loaders,
    *,
    announce: bool = True,
) -> dict[str, torch.Tensor]:
    """步骤7：检查一个训练批次 / Step 7: inspect one training batch."""

    if announce:
        announce_step(
            7,
            "检查训练批次接口",
            "Inspect the training batch contract",
            purpose="在连接正式模型前确认张量维度、dtype、标签映射和身份字段。",
            inputs="train DataLoader 的第一个 batch。",
            outputs="包含 input/label/raw_label/coordinate/sample_index 的批次字典。",
            acceptance="patch 路线为 N×1×B×H×W；标签为 int64 且 raw_label=model_label+1。",
        )
    train_loader = loaders["train"]
    if train_loader is None:
        raise AssertionError("training DataLoader is missing")
    batch = next(iter(train_loader))
    required = {"input", "label", "raw_label", "coordinate", "sample_index"}
    if set(batch) != required:
        raise AssertionError(f"batch keys must be {sorted(required)}")
    inputs = batch["input"]
    if settings.config.representation == "patch":
        expected_tail = (
            1,
            settings.config.n_components
            if settings.config.reducer in {"pca", "lda"}
            else inputs.shape[2],
            settings.config.patch_size,
            settings.config.patch_size,
        )
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != expected_tail:
            raise AssertionError(
                f"expected patch batch N×{expected_tail}, got {tuple(inputs.shape)}"
            )
    elif inputs.ndim != 2:
        raise AssertionError("pixel batch must have shape N×features")
    if inputs.dtype != torch.float32:
        raise AssertionError(f"expected float32 input, got {inputs.dtype}")
    if batch["label"].dtype != torch.int64:
        raise AssertionError("model labels must be int64")
    if not torch.equal(batch["raw_label"], batch["label"] + 1):
        raise AssertionError("raw/model labels are not aligned")
    if not torch.isfinite(inputs).all():
        raise AssertionError("batch contains NaN or infinite values")
    print(f"输入 / input: shape={tuple(inputs.shape)}, dtype={inputs.dtype}")
    print(f"模型标签 / label: shape={tuple(batch['label'].shape)}, dtype={batch['label'].dtype}")
    print(f"原始标签 / raw_label: shape={tuple(batch['raw_label'].shape)}")
    print(f"坐标 / coordinate: shape={tuple(batch['coordinate'].shape)}")
    print(f"样本编号 / sample_index: shape={tuple(batch['sample_index'].shape)}")
    print("标签映射正确 / Label mapping valid: True")
    print("步骤 7 验收通过 / Step 7 accepted")
    return batch


def run_readonly_workflow(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    stop_after: int = 7,
    state_dir: Path | str | None = None,
    batch_size: int | None = None,
    loader_seed: int | None = None,
    num_workers: int | None = None,
    pin_memory: str | bool | None = "auto",
) -> dict[str, Any]:
    """顺序运行至指定步骤 / Run the read-only workflow through a selected step."""

    if stop_after not in range(1, 8):
        raise ValueError("stop_after must be between 1 and 7")
    result: dict[str, Any] = {}
    settings = load_workflow_settings(
        config_path,
        state_dir=state_dir,
        batch_size=batch_size,
        loader_seed=loader_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    result["settings"] = settings
    if stop_after == 1:
        return result
    data = load_fixed_dataset(settings)
    result["data"] = data
    if stop_after == 2:
        return result
    pipeline = load_frozen_preprocessing_state(settings, data)
    result["pipeline"] = pipeline
    if stop_after == 3:
        return result
    result["transformed_cube"] = transform_full_cube(settings, data, pipeline)
    if stop_after == 4:
        return result
    result["datasets"] = build_torch_datasets(data, pipeline)
    if stop_after == 5:
        return result
    loaders = build_torch_dataloaders(settings, data, pipeline)
    result["loaders"] = loaders
    if stop_after == 6:
        return result
    result["first_train_batch"] = inspect_first_training_batch(settings, loaders)
    print("\n数据处理主流程完成 / Data preprocessing workflow completed")
    print("安全边界 / Safety: no split generation, no refit, no overwrite, no training")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="高光谱数据预处理统一主程序 / Unified HSI preprocessing workflow"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--stop-after", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--loader-seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", choices=("auto", "true", "false"), default="auto")
    args = parser.parse_args()
    run_readonly_workflow(
        args.config,
        stop_after=args.stop_after,
        state_dir=args.state_dir,
        batch_size=args.batch_size,
        loader_seed=args.loader_seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


if __name__ == "__main__":
    main()
