"""生成冻结预处理状态与 HybridSN 可直接复用的模型就绪数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.datasets.高光谱预处理 import (
    HSIPreprocessingPipeline,
    PreprocessingConfig,
    load_hsi_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拟合/复用预处理状态并保存模型就绪 NPZ。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PreprocessingConfig.from_mapping(values)
    data = load_hsi_data(PROJECT_ROOT, config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data/processed" / config.dataset_name / config.route_name()
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "preprocessing_state.npz"
    metadata_path = output_dir / "metadata.json"
    model_ready_path = output_dir / "model_ready_dataset.npz"
    existing_state = state_path.is_file() and metadata_path.is_file()
    if existing_state and not args.overwrite:
        pipeline = HSIPreprocessingPipeline.load_state(state_path, metadata_path)
        if pipeline.config != config:
            raise ValueError("existing preprocessing state does not match the requested config")
        pipeline.attach_transformed_cube(data.cube)
        print(f"preprocessing_state=reused:{state_path.relative_to(PROJECT_ROOT)}")
    else:
        if (state_path.exists() or metadata_path.exists()) and not args.overwrite:
            raise FileExistsError("preprocessing state is incomplete; use --overwrite after inspection")
        pipeline = HSIPreprocessingPipeline(config).fit(data)
        pipeline.save_state(output_dir, overwrite=args.overwrite)
        print(f"preprocessing_state=created:{state_path.relative_to(PROJECT_ROOT)}")

    if model_ready_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite model-ready artifact: {model_ready_path}")
    temporary = output_dir / "model_ready_dataset.tmp.npz"
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
    temporary.replace(model_ready_path)
    print(f"model_ready={model_ready_path.relative_to(PROJECT_ROOT)}")
    print(
        f"route={config.route_name()} shape={pipeline.transformed_cube_.shape} "
        f"train={data.indices_by_split['train'].size} "
        f"validation={data.indices_by_split['validation'].size} "
        f"test={data.indices_by_split['test'].size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
