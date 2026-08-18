"""按顺序运行论文复现 + 多模型对比的全部实验（跳过已完成者）。

运行清单（15 项 = 5 模型 × 3 数据集）：
  - 论文支路（原始波段 + patch11）：paper3d1dcnn / paper3dcnn / improvedpaper3d1dcnn
  - HybridSN 支路（PCA15 + patch25）：hybridsn / improved_hybridsn

每项通过 ``运行论文复现.py`` 单独跑一次；若 experiments 下已存在对应
(dataset, model) 的 summary_row.json 则跳过，便于断点续跑。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("cannot locate 实验交付/pyproject.toml")


PROJECT_ROOT = find_project_root(Path(__file__))


PAPER_CONFIGS = {
    "pavia_university": PROJECT_ROOT / "configs/模型训练/Paper复现_Pavia.yaml",
    "indian_pines": PROJECT_ROOT / "configs/模型训练/Paper复现_IndianPines.yaml",
    "salinas": PROJECT_ROOT / "configs/模型训练/Paper复现_Salinas.yaml",
}
HYBRIDSN_CONFIGS = {
    "pavia_university": PROJECT_ROOT / "configs/模型训练/HybridSN对比_Pavia.yaml",
    "indian_pines": PROJECT_ROOT / "configs/模型训练/HybridSN对比_IndianPines.yaml",
    "salinas": PROJECT_ROOT / "configs/模型训练/HybridSN对比_Salinas.yaml",
}

DATASETS = ["pavia_university", "indian_pines", "salinas"]


def build_run_list() -> list[tuple[Path, str, str]]:
    """返回 (config_path, model_name, dataset_name) 的完整运行清单。"""
    runs: list[tuple[Path, str, str]] = []
    for dataset in DATASETS:
        runs.append((PAPER_CONFIGS[dataset], "paper3d1dcnn", dataset))
    for dataset in DATASETS:
        runs.append((PAPER_CONFIGS[dataset], "paper3dcnn", dataset))
    for dataset in DATASETS:
        runs.append((PAPER_CONFIGS[dataset], "improvedpaper3d1dcnn", dataset))
    for dataset in DATASETS:
        runs.append((HYBRIDSN_CONFIGS[dataset], "hybridsn", dataset))
    for dataset in DATASETS:
        runs.append((HYBRIDSN_CONFIGS[dataset], "improved_hybridsn", dataset))
    return runs


def is_done(dataset: str, model: str, experiments_root: Path) -> bool:
    for path in experiments_root.rglob("summary_row.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # 只有完整运行（max_epochs_configured > 1）才算完成，1-epoch 冒烟测试不算。
        if row.get("dataset") == dataset and row.get("model") == model and int(row.get("max_epochs_configured", 0)) > 1:
            return True
    return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序运行全部对比实验。")
    parser.add_argument("--dry-run", action="store_true", help="只打印清单，不执行。")
    parser.add_argument("--only-dataset", type=str, help="只跑指定数据集。")
    parser.add_argument("--only-model", type=str, help="只跑指定模型。")
    parser.add_argument("--epochs", type=int, help="覆盖所有运行的 epoch 数（调试）。")
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "experiments")
    return parser.parse_args()


def run_one(config: Path, model: str, dataset: str, epochs: int | None) -> bool:
    command = [sys.executable, str(PROJECT_ROOT / "scripts/运行论文复现.py"), "--config", str(config), "--model", model]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    print(f"\n===== 开始运行 dataset={dataset} model={model} =====", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode == 0:
        print(f"===== 完成 dataset={dataset} model={model} =====\n", flush=True)
        return True
    print(f"===== 失败(exit={completed.returncode}) dataset={dataset} model={model} =====\n", flush=True)
    return False


def main() -> int:
    args = parse_arguments()
    runs = build_run_list()
    filtered = [
        (config, model, dataset)
        for (config, model, dataset) in runs
        if (args.only_dataset is None or dataset == args.only_dataset)
        and (args.only_model is None or model == args.only_model)
    ]
    if args.dry_run:
        for config, model, dataset in filtered:
            done = is_done(dataset, model, args.experiments_root)
            print(f"{'[done]' if done else '[todo]'} {dataset:>18s} {model:>22s} <- {config.name}")
        print(f"共 {len(filtered)} 项（已完成 {sum(1 for _, m, d in filtered if is_done(d, m, args.experiments_root))} 项）")
        return 0

    results: list[tuple[str, str, bool]] = []
    for config, model, dataset in filtered:
        if is_done(dataset, model, args.experiments_root):
            print(f"[skip] {dataset} {model}（已存在 summary_row.json）", flush=True)
            results.append((dataset, model, True))
            continue
        success = run_one(config, model, dataset, args.epochs)
        results.append((dataset, model, success))

    failed = [(d, m) for (d, m, ok) in results if not ok]
    print("\n===== 汇总 =====")
    for dataset, model, ok in results:
        print(f"  {'OK ' if ok else 'FAIL'} {dataset:>18s} {model:>22s}")
    if failed:
        print(f"\n共 {len(failed)} 项失败：{failed}")
        return 1
    print(f"\n全部 {len(results)} 项完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
