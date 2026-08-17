"""高光谱预处理统计与可视化 / HSI preprocessing visual analysis.

划分数量与空间位置可以展示全部 split，因为它们属于已冻结的实验设计。
光谱、降维特征和 patch 内容只允许使用训练样本，避免在模型选择前窥视测试特征。

Split counts and membership locations may include all frozen partitions. Spectral,
reducer-feature and patch content visualizations are deliberately restricted to training
samples to avoid test-feature inspection during model development.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.datasets.高光谱预处理 import (
    HSIDataBundle,
    HSIPreprocessingPipeline,
    HSITensorDataset,
    LDASpectralReducer,
    PCASpectralReducer,
)


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_LABELS = ("训练 / Train", "验证 / Validation", "测试 / Test")
SPLIT_COLORS = ("#4472C4", "#ED7D31", "#A5A5A5")
CLASS_COLORS = tuple(plt.get_cmap("tab10")(index) for index in range(9))


def configure_bilingual_plots() -> None:
    """配置 Windows 中文字体 / Configure fonts for bilingual chart text."""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _class_title(class_id: int, english: str, chinese_names: Sequence[str] | None) -> str:
    chinese = (
        chinese_names[class_id - 1]
        if chinese_names is not None and class_id <= len(chinese_names)
        else ""
    )
    return f"{class_id}. {chinese} / {english}" if chinese else f"{class_id}. {english}"


def compute_split_class_counts(data: HSIDataBundle) -> np.ndarray:
    """返回 C×3 的逐类 split 数量 / Return per-class split counts (C×3)."""

    class_count = len(data.spec.class_names)
    counts = np.zeros((class_count, len(SPLIT_NAMES)), dtype=np.int64)
    for split_column, split_name in enumerate(SPLIT_NAMES):
        indices = data.indices_by_split[split_name]
        split_labels = data.labels[indices]
        for class_id in range(1, class_count + 1):
            counts[class_id - 1, split_column] = np.count_nonzero(
                split_labels == class_id
            )
    if int(counts.sum()) != int(data.labels.size):
        raise AssertionError("split class counts do not cover every labeled sample")
    return counts


def plot_split_statistics(data: HSIDataBundle) -> tuple[Figure, np.ndarray]:
    """绘制总体与逐类 split 数量 / Plot overall and per-class split counts."""

    configure_bilingual_plots()
    counts = compute_split_class_counts(data)
    totals = counts.sum(axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    bars = axes[0].bar(SPLIT_LABELS, totals, color=SPLIT_COLORS)
    axes[0].set_title("总体划分数量 / Overall split counts")
    axes[0].set_ylabel("有标签像元数 / Labeled pixels")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, totals, strict=True):
        ratio = value / totals.sum() * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({ratio:.1f}%)",
            ha="center",
            va="bottom",
        )

    class_ids = np.arange(1, len(data.spec.class_names) + 1)
    width = 0.25
    for offset, (label, color) in enumerate(zip(SPLIT_LABELS, SPLIT_COLORS, strict=True)):
        axes[1].bar(
            class_ids + (offset - 1) * width,
            counts[:, offset],
            width=width,
            label=label,
            color=color,
        )
    axes[1].set_title("逐类别划分数量 / Per-class split counts")
    axes[1].set_xlabel("类别编号 / Class ID")
    axes[1].set_ylabel("像元数 / Pixels")
    axes[1].set_xticks(class_ids)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure, axes


def plot_split_spatial_map(data: HSIDataBundle) -> tuple[Figure, Axes]:
    """绘制冻结 split 的空间位置 / Plot spatial membership of frozen splits."""

    configure_bilingual_plots()
    membership = np.full(data.label_map.shape, np.nan, dtype=np.float32)
    for split_id, split_name in enumerate(SPLIT_NAMES):
        indices = data.indices_by_split[split_name]
        rows, columns = data.coordinates[indices].T
        membership[rows, columns] = split_id
    masked = np.ma.masked_invalid(membership)
    color_map = ListedColormap(SPLIT_COLORS).with_extremes(bad="white")

    figure, axis = plt.subplots(figsize=(8.5, 7.2))
    axis.imshow(masked, cmap=color_map, vmin=-0.5, vmax=2.5, interpolation="nearest")
    axis.set_title("固定划分空间分布 / Spatial map of frozen splits")
    axis.set_xlabel("列 / Column")
    axis.set_ylabel("行 / Row")
    axis.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(SPLIT_COLORS, SPLIT_LABELS, strict=True)
        ],
        loc="upper right",
    )
    figure.tight_layout()
    return figure, axis


def plot_training_mean_spectra(
    data: HSIDataBundle,
    *,
    class_names_zh: Sequence[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """仅绘制训练集逐类均值±标准差 / Plot train-only class spectra."""

    configure_bilingual_plots()
    rows, columns = data.train_coordinates.T
    train_spectra = data.cube[rows, columns, :].astype(np.float64, copy=False)
    train_labels = data.train_labels
    class_count = len(data.spec.class_names)
    grid_columns = 3
    grid_rows = int(np.ceil(class_count / grid_columns))
    figure, axes = plt.subplots(
        grid_rows,
        grid_columns,
        figsize=(15, 3.5 * grid_rows),
        sharex=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    bands = np.arange(1, data.cube.shape[2] + 1)
    for class_id, (english, axis) in enumerate(
        zip(data.spec.class_names, axes_array, strict=False),
        start=1,
    ):
        values = train_spectra[train_labels == class_id]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        color = CLASS_COLORS[(class_id - 1) % len(CLASS_COLORS)]
        axis.plot(bands, mean, color=color, linewidth=1.4, label="均值 / Mean")
        axis.fill_between(
            bands,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            label="±1 标准差 / ±1 SD",
        )
        axis.set_title(_class_title(class_id, english, class_names_zh), fontsize=10)
        axis.grid(alpha=0.2)
    for axis in axes_array[class_count:]:
        axis.axis("off")
    for axis in axes_array[-grid_columns:]:
        axis.set_xlabel("波段编号 / Band index")
    for axis in axes_array[::grid_columns]:
        axis.set_ylabel("原始数值 / Raw value")
    axes_array[0].legend(fontsize=8)
    figure.suptitle(
        "训练集逐类光谱统计（不使用验证/测试特征）\n"
        "Train-only class spectral statistics (no validation/test features)",
        y=1.01,
    )
    figure.tight_layout()
    return figure, axes_array


def plot_pca_explained_variance(
    pipeline: HSIPreprocessingPipeline,
) -> tuple[Figure, Axes, Axes]:
    """绘制冻结 PCA 的解释方差 / Plot explained variance of frozen PCA."""

    configure_bilingual_plots()
    if not isinstance(pipeline.reducer, PCASpectralReducer):
        raise ValueError("PCA explained variance requires a PCA preprocessing route")
    ratios = pipeline.reducer.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)
    components = np.arange(1, ratios.size + 1)
    figure, left_axis = plt.subplots(figsize=(10.5, 5.2))
    right_axis = left_axis.twinx()
    left_axis.bar(components, ratios, color="#4472C4", alpha=0.8)
    right_axis.plot(components, cumulative, color="#C00000", marker="o", linewidth=2)
    left_axis.set_title("冻结 PCA 的解释方差 / Explained variance of frozen PCA")
    left_axis.set_xlabel("主成分编号 / Component")
    left_axis.set_ylabel("单分量解释方差（%）/ Individual variance (%)", color="#4472C4")
    right_axis.set_ylabel("累计解释方差（%）/ Cumulative variance (%)", color="#C00000")
    left_axis.set_xticks(components)
    left_axis.grid(axis="y", alpha=0.25)
    right_axis.set_ylim(0, 101)
    right_axis.annotate(
        f"PCA-{ratios.size}: {cumulative[-1]:.4f}%",
        xy=(components[-1], cumulative[-1]),
        xytext=(-125, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C00000"},
    )
    figure.tight_layout()
    return figure, left_axis, right_axis


def plot_reducer_explained_variance(
    pipeline: HSIPreprocessingPipeline,
) -> tuple[Figure, Axes, Axes]:
    """绘制 PCA 或 LDA 的冻结解释比 / Plot frozen PCA/LDA ratios."""

    if isinstance(pipeline.reducer, PCASpectralReducer):
        return plot_pca_explained_variance(pipeline)
    configure_bilingual_plots()
    if not isinstance(pipeline.reducer, LDASpectralReducer):
        raise ValueError("explained ratios require a PCA or LDA preprocessing route")
    ratios = pipeline.reducer.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)
    components = np.arange(1, ratios.size + 1)
    figure, left_axis = plt.subplots(figsize=(10.5, 5.2))
    right_axis = left_axis.twinx()
    left_axis.bar(components, ratios, color="#70AD47", alpha=0.8)
    right_axis.plot(components, cumulative, color="#C00000", marker="o", linewidth=2)
    left_axis.set_title("冻结 LDA 的判别解释比 / Explained ratio of frozen LDA")
    left_axis.set_xlabel("判别分量编号 / Discriminant component")
    left_axis.set_ylabel("单分量解释比（%）/ Individual ratio (%)", color="#548235")
    right_axis.set_ylabel("累计解释比（%）/ Cumulative ratio (%)", color="#C00000")
    left_axis.set_xticks(components)
    left_axis.grid(axis="y", alpha=0.25)
    right_axis.set_ylim(0, 101)
    right_axis.annotate(
        f"LDA-{ratios.size}: {cumulative[-1]:.4f}%",
        xy=(components[-1], cumulative[-1]),
        xytext=(-125, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C00000"},
    )
    figure.tight_layout()
    return figure, left_axis, right_axis


def plot_training_pca_scatter(
    data: HSIDataBundle,
    pipeline: HSIPreprocessingPipeline,
    *,
    class_names_zh: Sequence[str] | None = None,
    max_samples_per_class: int = 500,
    seed: int = 1442,
) -> tuple[Figure, Axes]:
    """绘制训练集 PCA1/PCA2 散点 / Plot train-only PCA1/PCA2 scatter."""

    configure_bilingual_plots()
    if pipeline.transformed_cube_ is None or pipeline.output_bands < 2:
        raise ValueError("attach a transformed cube with at least two components first")
    if max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive")
    rows, columns = data.train_coordinates.T
    features = pipeline.transformed_cube_[rows, columns, :2]
    labels = data.train_labels
    generator = np.random.default_rng(seed)

    figure, axis = plt.subplots(figsize=(9, 7))
    for class_id, english in enumerate(data.spec.class_names, start=1):
        positions = np.flatnonzero(labels == class_id)
        if positions.size > max_samples_per_class:
            positions = generator.choice(
                positions,
                size=max_samples_per_class,
                replace=False,
            )
        axis.scatter(
            features[positions, 0],
            features[positions, 1],
            s=9,
            alpha=0.45,
            color=CLASS_COLORS[(class_id - 1) % len(CLASS_COLORS)],
            label=_class_title(class_id, english, class_names_zh),
        )
    axis.set_title(
        "训练集 PCA-1/PCA-2 分布 / Train-only PCA-1/PCA-2 distribution"
    )
    axis.set_xlabel("第一主成分 / PCA component 1")
    axis.set_ylabel("第二主成分 / PCA component 2")
    axis.grid(alpha=0.2)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    figure.tight_layout()
    return figure, axis


def plot_training_reducer_scatter(
    data: HSIDataBundle,
    pipeline: HSIPreprocessingPipeline,
    *,
    class_names_zh: Sequence[str] | None = None,
    max_samples_per_class: int = 500,
    seed: int = 1442,
) -> tuple[Figure, Axes]:
    """绘制训练集 PCA/LDA 前两分量 / Plot first two train-only components."""

    figure, axis = plot_training_pca_scatter(
        data,
        pipeline,
        class_names_zh=class_names_zh,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    if isinstance(pipeline.reducer, PCASpectralReducer):
        return figure, axis
    if not isinstance(pipeline.reducer, LDASpectralReducer):
        plt.close(figure)
        raise ValueError("component scatter requires a PCA or LDA preprocessing route")
    axis.set_title("训练集 LDA-1/LDA-2 分布 / Train-only LDA-1/LDA-2 distribution")
    axis.set_xlabel("第一判别分量 / LDA component 1")
    axis.set_ylabel("第二判别分量 / LDA component 2")
    return figure, axis


def plot_training_patch_gallery(
    data: HSIDataBundle,
    datasets: dict[str, HSITensorDataset | None],
    *,
    class_names_zh: Sequence[str] | None = None,
    component_index: int = 0,
) -> tuple[Figure, np.ndarray]:
    """每类展示一个训练 patch / Show one deterministic train patch per class."""

    configure_bilingual_plots()
    train_dataset = datasets.get("train")
    if train_dataset is None or train_dataset.representation != "patch":
        raise ValueError("patch gallery requires a non-empty training patch Dataset")
    if component_index < 0 or component_index >= train_dataset.transformed_cube.shape[2]:
        raise ValueError("component_index is out of range")
    class_count = len(data.spec.class_names)
    grid_columns = 3
    grid_rows = int(np.ceil(class_count / grid_columns))
    figure, axes = plt.subplots(grid_rows, grid_columns, figsize=(12, 4 * grid_rows))
    axes_array = np.asarray(axes).reshape(-1)

    train_indices = train_dataset.sample_indices
    train_labels = data.labels[train_indices]
    train_coordinates = data.coordinates[train_indices]
    for class_id, (english, axis) in enumerate(
        zip(data.spec.class_names, axes_array, strict=False),
        start=1,
    ):
        positions = np.flatnonzero(train_labels == class_id)
        coordinates = train_coordinates[positions]
        coordinate_median = np.median(coordinates, axis=0)
        representative = positions[
            np.argmin(np.square(coordinates - coordinate_median).sum(axis=1))
        ]
        sample = train_dataset[int(representative)]
        patch = sample["input"][0, component_index].numpy()
        center = train_dataset.patch_size // 2
        axis.imshow(patch, cmap="viridis")
        axis.scatter([center], [center], c="red", s=18, marker="+")
        coordinate = tuple(int(value) for value in sample["coordinate"])
        axis.set_title(
            _class_title(class_id, english, class_names_zh)
            + f"\n坐标 / Coordinate: {coordinate}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[class_count:]:
        axis.axis("off")
    figure.suptitle(
        f"训练集代表性邻域（PCA-{component_index + 1}）/ "
        f"Representative train patches (PCA-{component_index + 1})",
        y=1.01,
    )
    figure.tight_layout()
    return figure, axes_array


def plot_batch_class_distribution(
    batch: dict[str, torch.Tensor],
    *,
    class_names_en: Sequence[str],
    class_names_zh: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    """绘制一个训练 batch 的类别构成 / Plot class counts in one train batch."""

    configure_bilingual_plots()
    labels = batch["label"].detach().cpu().numpy()
    counts = np.bincount(labels, minlength=len(class_names_en))
    class_ids = np.arange(1, len(class_names_en) + 1)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    bars = axis.bar(class_ids, counts, color=CLASS_COLORS[: len(class_names_en)])
    axis.set_title("首个训练批次类别构成 / Class composition of first train batch")
    axis.set_xlabel("类别编号 / Class ID")
    axis.set_ylabel("样本数 / Samples")
    axis.set_xticks(class_ids)
    axis.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            str(int(count)),
            ha="center",
            va="bottom",
        )
    legend_labels = [
        _class_title(class_id, english, class_names_zh)
        for class_id, english in enumerate(class_names_en, start=1)
    ]
    axis.legend(
        handles=[
            Patch(facecolor=CLASS_COLORS[index], label=label)
            for index, label in enumerate(legend_labels)
        ],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    figure.tight_layout()
    return figure, axis
