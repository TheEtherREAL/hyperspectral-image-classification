"""为注册的三套高光谱数据生成可追溯、不可变的固定划分。"""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import sklearn

from src.datasets.固定划分 import create_fixed_protocol_splits, write_split_artifacts
from src.datasets.数据读取 import load_dataset, public_mat_keys
from src.datasets.数据集注册 import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 1442


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_metadata(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "md5": file_hash(path, "md5"),
        "sha256": file_hash(path, "sha256"),
        "mat_variables": public_mat_keys(path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成两套分层随机固定划分。")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="pavia_university")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    spec = DATASETS[args.dataset]
    raw_dir = PROJECT_ROOT / "data/raw"
    cube, label_map = load_dataset(raw_dir, spec)
    splits = create_fixed_protocol_splits(label_map, seed=args.seed)

    source_files = {
        spec.data_file: source_file_metadata(raw_dir / spec.data_file),
        spec.label_file: source_file_metadata(raw_dir / spec.label_file),
    }
    shared_metadata = {
        "dataset": {
            "name": spec.name,
            "cube_shape": list(cube.shape),
            "label_shape": list(label_map.shape),
            "data_key": spec.data_key,
            "label_key": spec.label_key,
            "background_label": 0,
            "class_ids": list(range(1, len(spec.class_names) + 1)),
            "class_names": list(spec.class_names),
            "labeled_pixels": int(np.count_nonzero(label_map)),
            "source_files": source_files,
        },
        "sample_representation": {
            "coordinates": "zero-based (row, column) in the original label map",
            "labels": "original dataset class IDs 1..C; background 0 is excluded",
            "split_ids": {"0": "train", "1": "validation", "2": "test"},
            "indices": "positions into the aligned coordinates and labels arrays",
        },
        "partition_algorithm": {
            "implementation": "sklearn.model_selection.train_test_split",
            "shuffle": True,
            "stratify": "original class label",
            "steps": [
                {
                    "name": "shared paper train / test split",
                    "train_size": 0.30,
                    "test_size": 0.70,
                    "random_state": args.seed,
                },
                {
                    "name": "fair train / validation split within the 30% pool",
                    "train_size": 0.80,
                    "test_size": 0.20,
                    "random_state": args.seed,
                },
            ],
            "relationship": (
                "Both protocols share the same test indices; fair train plus validation "
                "equals the paper30 training pool."
            ),
            "rounding": "Actual integer counts are authoritative and recorded in statistics.",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    for protocol_name, split in splits.items():
        paths = write_split_artifacts(
            split,
            output_dir=output_dir,
            dataset_name=spec.name,
            class_names=spec.class_names,
            metadata=shared_metadata,
        )
        counts = {name: int(indices.size) for name, indices in split.indices_by_split().items()}
        print(f"protocol={protocol_name} seed={args.seed} counts={counts}")
        for artifact_name, path in paths.items():
            print(f"{artifact_name}={path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
