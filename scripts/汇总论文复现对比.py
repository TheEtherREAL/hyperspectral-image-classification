"""汇总论文复现与多模型对比结果：扫描 summary_row.json，生成对比表与对比图。

每个实验目录由 ``运行论文复现.py`` 生成一个 ``summary_row.json``。本脚本：
1. 扫描 experiments 目录下所有 summary_row.json；
2. 按 (dataset, model) 去重，保留时间戳最新的那次运行；
3. 输出对比 CSV、Markdown 表格，以及 OA/AA/Kappa、参数量-精度、训练耗时三张图。

只读取真实运行产出的数值，不进行任何推断。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("cannot locate 实验交付/pyproject.toml")


PROJECT_ROOT = find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ORDER = [
    "paper3dcnn",
    "paper3d1dcnn",
    "improvedpaper3d1dcnn",
    "hybridsn",
    "improved_hybridsn",
]

MODEL_LABELS = {
    "paper3dcnn": "Paper3DCNN",
    "paper3d1dcnn": "Paper3D1DCNN",
    "improvedpaper3d1dcnn": "改进Paper3D1DCNN",
    "hybridsn": "HybridSN",
    "improved_hybridsn": "改进HybridSN",
}

DATASET_ORDER = ["pavia_university", "indian_pines", "salinas"]
DATASET_LABELS = {
    "pavia_university": "Pavia University",
    "indian_pines": "Indian Pines",
    "salinas": "Salinas",
}


def _model_sort_key(name: str) -> int:
    key = name.lower().replace("-", "").replace("_", "")
    for index, canonical in enumerate(MODEL_ORDER):
        if key == canonical or key == canonical.replace("_", ""):
            return index
    return len(MODEL_ORDER)


def _dataset_sort_key(name: str) -> int:
    try:
        return DATASET_ORDER.index(name)
    except ValueError:
        return len(DATASET_ORDER)


def collect_summary_rows(experiments_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(experiments_root.rglob("summary_row.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"跳过无法解析的 {path}: {error}", file=sys.stderr)
            continue
        row["_source"] = str(path)
        row["_dirname"] = path.parent.name
        # 跳过 1-epoch 冒烟测试，只汇总完整运行。
        if int(row.get("max_epochs_configured", 0)) <= 1:
            continue
        rows.append(row)
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个 (dataset, model) 保留时间戳最新（目录名最大）的一次运行。"""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        if not key[0] or not key[1]:
            continue
        current = best.get(key)
        if current is None or row["_dirname"] > current["_dirname"]:
            best[key] = row
    return list(best.values())


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (_dataset_sort_key(str(row["dataset"])), _model_sort_key(str(row["model"]))),
    )


def as_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        table.append(
            {
                "dataset": DATASET_LABELS.get(row["dataset"], row["dataset"]),
                "dataset_key": row["dataset"],
                "model": MODEL_LABELS.get(row["model"], row["model"]),
                "model_key": row["model"],
                "params": int(row.get("trainable_parameters", 0)),
                "epochs_trained": int(row.get("epochs_trained", 0)),
                "max_epochs": int(row.get("max_epochs_configured", 0)),
                "train_seconds": float(row.get("training_total_seconds", 0.0)),
                "mean_epoch_seconds": float(row.get("mean_epoch_seconds", 0.0)),
                "oa": float(row.get("oa", float("nan"))),
                "aa": float(row.get("aa", float("nan"))),
                "kappa": float(row.get("kappa", float("nan"))),
                "test_samples": int(row.get("test_samples", 0)),
                "output_bands": int(row.get("output_bands", 0)),
                "patch_size": int(row.get("patch_size", 0)),
                "output_dir": row.get("output_dir", ""),
            }
        )
    return table


def write_csv(table: list[dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "dataset", "model", "params", "output_bands", "patch_size",
        "epochs_trained", "max_epochs", "train_seconds", "mean_epoch_seconds",
        "oa", "aa", "kappa", "test_samples", "output_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


def write_markdown(table: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 论文复现与多模型对比结果汇总",
        "",
        "> 所有数值来自 `运行论文复现.py` 产出的 `summary_row.json`，同一固定划分 "
        "`fair24_6_70`、seed=1442，测试集仅评测一次。",
        "",
        "| 数据集 | 模型 | 参数 | 波段/patch | OA | AA | Kappa | 训练轮数 | 训练耗时(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['params']:,} | "
            f"{row['output_bands']}/{row['patch_size']} | {row['oa']:.4f} | "
            f"{row['aa']:.4f} | {row['kappa']:.4f} | "
            f"{row['epochs_trained']}/{row['max_epochs']} | {row['train_seconds']:.1f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metrics_by_dataset(table: list[dict[str, Any]], path: Path) -> None:
    datasets = DATASET_ORDER
    models = [m for m in MODEL_ORDER]
    metric_keys = [("oa", "OA"), ("aa", "AA"), ("kappa", "Kappa")]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=180)
    for axis, (metric, title) in zip(axes, metric_keys, strict=True):
        width = 0.72
        for dataset_index, dataset in enumerate(datasets):
            subset = [r for r in table if r["dataset_key"] == dataset]
            by_model = {r["model_key"]: r[metric] for r in subset}
            values = [by_model.get(m, float("nan")) for m in models]
            x = np.arange(len(models)) + dataset_index * (width + 0.04)
            axis.bar(x, values, width=width, label=DATASET_LABELS[dataset], alpha=0.85)
        axis.set_xticks(np.arange(len(models)) + (len(datasets) - 1) * (width + 0.04) / 2)
        axis.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title, fontsize=13)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("统一协议（fair24_6_70 / seed1442）四模型 × 三数据集对比", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_params_vs_oa(table: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), dpi=180)
    model_colors = {
        "paper3dcnn": "#e41a1c",
        "paper3d1dcnn": "#ff7f00",
        "improvedpaper3d1dcnn": "#377eb8",
        "hybridsn": "#4daf4a",
        "improved_hybridsn": "#984ea3",
    }
    dataset_markers = {"pavia_university": "o", "indian_pines": "s", "salinas": "^"}
    plotted_models: set[str] = set()
    for row in table:
        model = row["model_key"]
        axis.scatter(
            row["params"],
            row["oa"],
            marker=dataset_markers.get(row["dataset_key"], "x"),
            s=110,
            color=model_colors.get(model, "#333333"),
            edgecolor="black",
            linewidths=0.6,
            label=MODEL_LABELS[model] if model not in plotted_models else None,
        )
        plotted_models.add(model)
        axis.annotate(
            f"{row['dataset_key'][:6]}",
            (row["params"], row["oa"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="#333333",
        )
    axis.set_xscale("log")
    axis.set_xlabel("可训练参数量（对数坐标）")
    axis.set_ylabel("总体精度 OA")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    axis.set_title("参数量与测试精度权衡（OA）")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_training_time(table: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5), dpi=180)
    labels = [f"{row['dataset'][:8]}·{row['model']}" for row in table]
    seconds = [row["train_seconds"] for row in table]
    positions = np.arange(len(labels))
    axis.barh(positions, seconds, color="#2878B5", alpha=0.85)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("训练总耗时（秒）")
    axis.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, seconds, strict=True):
        axis.text(value + 3, position, f"{value:.0f}s", va="center", fontsize=7)
    axis.set_title("各实验训练耗时")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总论文复现与多模型对比结果。")
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "report" / "对比结果")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    rows = collect_summary_rows(args.experiments_root)
    if not rows:
        print("未在 experiments 目录找到任何 summary_row.json。", file=sys.stderr)
        return 1
    rows = sort_rows(deduplicate(rows))
    table = as_table(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table, args.output_dir / "对比结果.csv")
    write_markdown(table, args.output_dir / "对比结果.md")
    plot_metrics_by_dataset(table, args.output_dir / "对比指标.png")
    plot_params_vs_oa(table, args.output_dir / "参数量精度权衡.png")
    plot_training_time(table, args.output_dir / "训练耗时.png")

    print(f"汇总 {len(table)} 条运行结果，输出到 {args.output_dir}")
    for row in table:
        print(
            f"  {row['dataset']:>16s} | {row['model']:>20s} | OA={row['oa']:.4f} "
            f"AA={row['aa']:.4f} Kappa={row['kappa']:.4f} | params={row['params']:>10,} "
            f"| epochs={row['epochs_trained']:>3}/{row['max_epochs']} "
            f"| {row['train_seconds']:>7.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
