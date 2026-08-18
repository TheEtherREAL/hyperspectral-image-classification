"""把 Indian Pines / Salinas 的 fair24_6_70 固定划分从旧序列化格式升级到当前格式。

背景：旧版 ``生成固定划分.py`` 生成的 NPZ 只含
``coordinates / labels / train_indices / validation_indices / test_indices``，
JSON 的 ``protocol`` 是纯字符串。当前 ``load_hsi_data`` 期望 NPZ 含
``dataset_name / protocol_name / seed`` 等元数据键，且 JSON 的 ``protocol``
为对象。Pavia 已在 2026-08-18 00:15 重写为新格式，Indian Pines / Salinas 仍为旧格式，
导致 ``运行论文复现.py`` 在这两个数据集上报
``KeyError: 'dataset_name is not a file in the archive'``。

本脚本**只做格式迁移，不改变任何划分身份**：先校验旧文件冻结的索引与当前
``create_fixed_protocol_splits(seed=1442)`` 逐元素一致，再以同一索引重写为新格式。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.数据读取 import load_dataset  # noqa: E402
from src.datasets.数据集注册 import DATASETS  # noqa: E402
from src.datasets.固定划分 import SPLIT_NAMES, create_fixed_protocol_splits  # noqa: E402
SEED = 1442
PROTOCOL = "fair24_6_70"
TARGET_FRACTIONS = {"train": 0.24, "validation": 0.06, "test": 0.70}


def upgrade_npz(path: Path, dataset_name: str) -> None:
    old = np.load(path, allow_pickle=False)
    coordinates = old["coordinates"].astype(np.int32, copy=True)
    labels = old["labels"].astype(np.int16, copy=True)
    train = old["train_indices"].astype(np.int64, copy=True)
    validation = old["validation_indices"].astype(np.int64, copy=True)
    test = old["test_indices"].astype(np.int64, copy=True)

    split_ids = np.full(labels.shape[0], fill_value=255, dtype=np.uint8)
    split_ids[train] = 0
    split_ids[validation] = 1
    split_ids[test] = 2

    np.savez_compressed(
        path,
        schema_version=np.asarray("1.0"),
        dataset_name=np.asarray(dataset_name),
        protocol_name=np.asarray(PROTOCOL),
        seed=np.asarray(SEED, dtype=np.int64),
        coordinates=coordinates,
        labels=labels,
        split_ids=split_ids,
        split_id_names=np.asarray(SPLIT_NAMES),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        train_coordinates=coordinates[train],
        train_labels=labels[train],
        validation_coordinates=coordinates[validation],
        validation_labels=labels[validation],
        test_coordinates=coordinates[test],
        test_labels=labels[test],
    )


def upgrade_json(path: Path, dataset_name: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("protocol"), dict):
        return  # 已经是新格式，跳过
    data["schema_version"] = "1.0"
    data["dataset_name"] = dataset_name
    data["protocol"] = {
        "name": PROTOCOL,
        "purpose": "24% train / 6% validation / 70% test fair model comparison",
        "seed": SEED,
        "target_fractions": dict(TARGET_FRACTIONS),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for dataset_name in ("indian_pines", "salinas"):
        split_dir = PROJECT_ROOT / "data" / "splits"
        stem = f"{dataset_name}__{PROTOCOL}__seed{SEED}"
        npz_path = split_dir / f"{stem}.npz"
        json_path = split_dir / f"{stem}.json"

        # 校验旧文件冻结的索引与当前算法完全一致（防误改划分身份）。
        _, label_map = load_dataset(PROJECT_ROOT / "data" / "raw", DATASETS[dataset_name])
        current = create_fixed_protocol_splits(label_map, seed=SEED)[PROTOCOL].indices_by_split()
        old = np.load(npz_path, allow_pickle=False)
        for split_name, indices in current.items():
            frozen = np.sort(old[f"{split_name}_indices"].astype(np.int64))
            assert np.array_equal(frozen, np.sort(indices.astype(np.int64))), (
                f"{dataset_name}/{split_name} 冻结索引与当前算法不一致，拒绝迁移"
            )

        upgrade_npz(npz_path, dataset_name)
        upgrade_json(json_path, dataset_name)
        print(f"已迁移 {dataset_name} -> 新格式（索引逐元素一致，仅补元数据键）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
