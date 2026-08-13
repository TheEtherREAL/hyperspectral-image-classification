"""只读检查数据或模型 YAML 配置 / Read-only YAML configuration checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.高光谱预处理 import PreprocessingConfig


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _positive_number(values: Mapping[str, Any], key: str) -> float | None:
    if key not in values:
        return None
    value = float(values[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def validate_experiment_sections(values: Mapping[str, Any]) -> list[str]:
    """校验未来模型配置中的通用字段，不执行训练。"""

    messages: list[str] = []
    model = values.get("model")
    training = values.get("training")
    if model is None and training is None:
        return messages
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise ValueError("model and training must both be YAML mappings")
    num_classes = int(model.get("num_classes", 0))
    if num_classes < 2:
        raise ValueError("model.num_classes must be at least 2")
    dropout = float(model.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    _positive_number(training, "learning_rate")
    _positive_number(training, "batch_size")
    _positive_number(training, "epochs")
    if float(training.get("weight_decay", 0.0)) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if "early_stopping_patience" in training:
        _positive_number(training, "early_stopping_patience")
    loader_batch_size = values.get("dataloader", {}).get("batch_size")
    training_batch_size = training.get("batch_size")
    if loader_batch_size is not None and training_batch_size is not None:
        if int(loader_batch_size) != int(training_batch_size):
            raise ValueError(
                "dataloader.batch_size and training.batch_size must be identical"
            )
    messages.append("模型/训练字段结构有效（仅配置检查，未开始训练）")
    return messages


def inspect_config(config_path: Path) -> dict[str, Any]:
    """解析配置并返回只读检查摘要。"""

    resolved = _resolve(config_path)
    values = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("YAML root must be a mapping")
    config = PreprocessingConfig.from_mapping(values)
    state_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / config.dataset_name
        / config.route_name()
    )
    state_files = (
        state_dir / "preprocessing_state.npz",
        state_dir / "metadata.json",
    )
    runtime = values.get("dataloader", {})
    if runtime and not isinstance(runtime, Mapping):
        raise ValueError("dataloader must be a YAML mapping")
    if runtime:
        if int(runtime.get("batch_size", 256)) < 1:
            raise ValueError("dataloader.batch_size must be positive")
        if int(runtime.get("num_workers", 0)) < 0:
            raise ValueError("dataloader.num_workers must be non-negative")
    messages = validate_experiment_sections(values)
    return {
        "path": resolved,
        "values": values,
        "config": config,
        "route": config.route_name(),
        "state_dir": state_dir,
        "state_exists": all(path.is_file() for path in state_files),
        "messages": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查高光谱 YAML 配置")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-state",
        action="store_true",
        help="若对应冻结预处理状态不存在，则返回错误",
    )
    args = parser.parse_args()
    result = inspect_config(args.config)
    config = result["config"]
    print(f"配置文件={result['path']}")
    print(f"数据集={config.dataset_name}")
    print(f"划分协议={config.split_protocol} seed={config.split_seed}")
    print(f"数据路线={result['route']}")
    print(
        "预处理="
        f"{config.standardization}+{config.reducer}{config.n_components or ''}+"
        f"{config.representation}{config.patch_size if config.representation == 'patch' else ''}"
    )
    print(f"冻结状态目录={result['state_dir']}")
    print(f"冻结状态存在={result['state_exists']}")
    for message in result["messages"]:
        print(message)
    if args.require_state and not result["state_exists"]:
        raise FileNotFoundError(
            "配置有效，但对应预处理状态尚不存在；请先运行 scripts/生成预处理状态.py"
        )
    print("配置检查=通过")


if __name__ == "__main__":
    main()
