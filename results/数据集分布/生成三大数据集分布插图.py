"""生成三大数据集分布插图 / Dataset-distribution figures for the three HSI datasets.

直观展现 Pavia University、Indian Pines、Salinas 三大数据集的空间结构、
类别样本分布与光谱分布，产出三张可直接插入实验报告的 PNG 插图。

用法：
    python 生成三大数据集分布插图.py

输出目录：本脚本同级的 figures/ 子目录，包含
    fig1_三大数据集_空间结构.png      —— 假彩色合成图 + 地物真值图（3 行 × 2 列）
    fig2_三大数据集_类别分布.png      —— 各类别有标签像元数（3 面板水平条形图）
    fig3_三大数据集_光谱分布.png      —— 有标签像元平均光谱（均值 ± 标准差）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 保证可导入本工程 src 包（本脚本位于 实验交付/数据集分布插图/ 下）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.数据读取 import load_dataset
from src.datasets.数据集注册 import DATASETS

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = Path(__file__).resolve().parent / "figures"

# 三大数据集顺序与显示元信息（空间尺寸 / 波段数 / 场景 / 传感器光谱范围）
ORDER = [
    ("pavia_university", "Pavia University", "城市", "610×340", 103, "0.43–0.86 μm"),
    ("indian_pines", "Indian Pines", "农业", "145×145", 200, "0.4–2.45 μm"),
    ("salinas", "Salinas", "农业", "512×217", 204, "0.4–2.45 μm"),
]
# 每套数据集统一的强调色，便于三幅图对照
ACCENT = {
    "pavia_university": "#2563EB",
    "indian_pines": "#16A34A",
    "salinas": "#EA580C",
}
# 假彩色合成所取的红/绿/蓝波段索引（沿用工程既有口径）
RGB_BANDS = (60, 30, 10)


def _stretch_band(band: np.ndarray) -> np.ndarray:
    """按 2%~98% 百分位做线性拉伸，抑制噪声、提升显示对比度。"""
    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _counts(label: np.ndarray, n_classes: int) -> np.ndarray:
    """返回类别 1..n_classes 各自的有标签像元数。"""
    return np.array([np.count_nonzero(label == c) for c in range(1, n_classes + 1)])


def _load(key: str):
    """读取数据集，返回 (cube, label, class_names)。"""
    cube, label = load_dataset(RAW_DIR, DATASETS[key])
    return cube, label, DATASETS[key].class_names


# --------------------------------------------------------------------------- #
# 图 1：空间结构（假彩色合成 + 地物真值）
# --------------------------------------------------------------------------- #
def plot_spatial_structure() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.5))

    for i, (key, name, _scene, _size, _n_bands, _wav) in enumerate(ORDER):
        cube, label, class_names = _load(key)
        rgb = np.stack([_stretch_band(cube[:, :, b]) for b in RGB_BANDS], axis=-1)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{name}　假彩色合成", fontsize=11)
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(
            label, cmap="nipy_spectral", vmin=0, vmax=len(class_names)
        )
        axes[i, 1].set_title(f"{name}　地物真值", fontsize=11)
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.suptitle("三大高光谱数据集空间结构", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "fig1_三大数据集_空间结构.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig1_三大数据集_空间结构.png")


# --------------------------------------------------------------------------- #
# 图 2：类别分布（有标签像元数）
# --------------------------------------------------------------------------- #
def plot_class_distribution() -> None:
    ratios = [9, 16, 16]  # 各数据集类别数，用于按行分配高度
    fig, axes = plt.subplots(
        3, 1, figsize=(10.5, 13.5), gridspec_kw={"height_ratios": ratios}
    )

    for i, (key, name, scene, size, n_bands, _wav) in enumerate(ORDER):
        cube, label, class_names = _load(key)
        counts = _counts(label, len(class_names))
        total = int((label > 0).sum())
        imbalance = float(counts.max() / counts.min())

        # 类别 1 在顶部：y 自上而下为 1..C
        y = np.arange(len(class_names), 0, -1)
        axes[i].barh(y, counts, color=ACCENT[key], alpha=0.9, edgecolor="white", lw=0.3)
        axes[i].set_yticks(y)
        axes[i].set_yticklabels(
            [f"{j}. {n}" for j, n in enumerate(class_names, start=1)], fontsize=7
        )
        axes[i].invert_yaxis()
        axes[i].set_xlabel("有标签像元数", fontsize=9)
        axes[i].set_title(
            f"{name}（{len(class_names)} 类，{total:,} 像元，"
            f"最/最少类比 {imbalance:.1f}:1）",
            fontsize=10,
            loc="left",
            color=ACCENT[key],
        )
        axes[i].grid(axis="x", linestyle="--", alpha=0.3)

        # 标注数值
        for yy, c in zip(y, counts):
            axes[i].text(c, yy, f" {c:,}", va="center", fontsize=6.5)

    fig.suptitle("三大高光谱数据集类别样本分布（可见类别不平衡）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "fig2_三大数据集_类别分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig2_三大数据集_类别分布.png")


# --------------------------------------------------------------------------- #
# 图 3：光谱分布（有标签像元平均光谱 ± 标准差）
# --------------------------------------------------------------------------- #
def plot_spectral_distribution() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11))

    for i, (key, name, scene, size, n_bands, wav) in enumerate(ORDER):
        cube, label, class_names = _load(key)
        mask = label > 0
        spectra = cube[mask]  # (N, B)
        mean = spectra.mean(axis=0)
        std = spectra.std(axis=0)
        bands = np.arange(1, n_bands + 1)

        axes[i].plot(bands, mean, color=ACCENT[key], lw=1.5, label="均值")
        axes[i].fill_between(
            bands, mean - std, mean + std, color=ACCENT[key], alpha=0.18, label="±1 标准差"
        )
        axes[i].set_xlabel("波段序号", fontsize=9)
        axes[i].set_ylabel("光谱值", fontsize=9)
        axes[i].set_title(
            f"{name}（{n_bands} 波段，{wav}，{scene}）", fontsize=10, loc="left", color=ACCENT[key]
        )
        axes[i].legend(loc="upper right", fontsize=7)
        axes[i].grid(linestyle="--", alpha=0.3)

    fig.suptitle("三大高光谱数据集有标签像元平均光谱", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "fig3_三大数据集_光谱分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig3_三大数据集_光谱分布.png")


# --------------------------------------------------------------------------- #
# 控制台汇总
# --------------------------------------------------------------------------- #
def print_summary() -> None:
    print("\n===== 三大数据集分布汇总 =====")
    for key, name, scene, size, n_bands, wav in ORDER:
        cube, label, class_names = _load(key)
        counts = _counts(label, len(class_names))
        total = int((label > 0).sum())
        print(
            f"\n【{name}】{scene} · 空间 {cube.shape[0]}×{cube.shape[1]} · "
            f"{n_bands} 波段 · {len(class_names)} 类 · 有标签像元 {total:,}"
        )
        for j, (cname, c) in enumerate(zip(class_names, counts), start=1):
            print(f"  {j:2d} {cname:32s} {c:7,}")
    print()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print_summary()
    plot_spatial_structure()
    plot_class_distribution()
    plot_spectral_distribution()
    print(f"\n插图已输出至：{FIG_DIR}")


if __name__ == "__main__":
    main()
