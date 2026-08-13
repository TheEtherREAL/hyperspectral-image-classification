"""拟合并保存无泄漏预处理状态 / Fit and save preprocessing state."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.datasets.高光谱预处理 import (
    HSIPreprocessingPipeline,
    PreprocessingConfig,
    load_hsi_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PreprocessingConfig.from_mapping(values)
    data = load_hsi_data(PROJECT_ROOT, config)
    pipeline = HSIPreprocessingPipeline(config).fit(data)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / config.dataset_name
            / config.route_name()
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    paths = pipeline.save_state(output_dir, overwrite=args.overwrite)

    print(f"dataset={config.dataset_name}")
    print(f"route={config.route_name()}")
    print(f"train_samples={data.train_indices.size}")
    print(f"transformed_cube_shape={pipeline.transformed_cube_.shape}")
    if config.reducer in {"pca", "lda"}:
        print(
            "cumulative_explained_variance_ratio="
            f"{pipeline.reducer.explained_variance_ratio_.sum():.10f}"
        )
    for name, path in paths.items():
        print(f"{name}={path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
