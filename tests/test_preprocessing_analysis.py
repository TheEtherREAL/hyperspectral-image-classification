from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.高光谱预处理 import (
    HSIDataBundle,
    HSIPreprocessingPipeline,
    LDASpectralReducer,
    PreprocessingConfig,
)
from src.datasets.数据集注册 import DatasetSpec
from src.visualization.预处理分析 import (
    compute_split_class_counts,
    plot_reducer_explained_variance,
    plot_split_spatial_map,
    plot_split_statistics,
    plot_training_mean_spectra,
    plot_training_reducer_scatter,
)


def _tiny_bundle() -> HSIDataBundle:
    label_map = np.array([[1, 1, 2], [2, 1, 2]], dtype=np.int16)
    coordinates = np.argwhere(label_map > 0).astype(np.int32)
    labels = label_map[coordinates[:, 0], coordinates[:, 1]]
    cube = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    return HSIDataBundle(
        spec=DatasetSpec(
            name="tiny",
            data_file="tiny.mat",
            label_file="tiny_gt.mat",
            data_key="tiny",
            label_key="tiny_gt",
            class_names=("Class A", "Class B"),
        ),
        cube=cube,
        label_map=label_map,
        coordinates=coordinates,
        labels=labels,
        indices_by_split={
            "train": np.array([0, 2], dtype=np.int64),
            "validation": np.array([1, 3], dtype=np.int64),
            "test": np.array([4, 5], dtype=np.int64),
        },
        split_path=Path("tiny.npz"),
        split_metadata_path=Path("tiny.json"),
        split_metadata={"protocol": {"name": "tiny", "seed": 1}},
    )


def test_split_class_counts_cover_all_samples() -> None:
    counts = compute_split_class_counts(_tiny_bundle())

    assert counts.shape == (2, 3)
    np.testing.assert_array_equal(counts.sum(axis=0), np.array([2, 2, 2]))
    assert int(counts.sum()) == 6


def test_split_count_and_spatial_figures_are_created() -> None:
    bundle = _tiny_bundle()
    count_figure, count_axes = plot_split_statistics(bundle)
    spatial_figure, spatial_axis = plot_split_spatial_map(bundle)

    assert len(count_axes) == 2
    assert spatial_axis.images[0].get_array().shape == bundle.label_map.shape
    plt.close(count_figure)
    plt.close(spatial_figure)


def test_spectral_figure_uses_one_panel_per_class() -> None:
    figure, axes = plot_training_mean_spectra(
        _tiny_bundle(),
        class_names_zh=("类别甲", "类别乙"),
    )

    assert sum(bool(axis.lines) for axis in axes) == 2
    plt.close(figure)


def test_lda_ratio_and_scatter_figures_are_route_aware() -> None:
    pipeline = HSIPreprocessingPipeline(
        PreprocessingConfig(
            reducer="lda",
            n_components=2,
            whiten=False,
            representation="pixel",
            patch_size=1,
        )
    )
    reducer = LDASpectralReducer(n_components=2)
    reducer.n_features_in_ = 4
    reducer.n_samples_seen_ = 6
    reducer.max_components_ = 2
    reducer.classes_ = np.array([1, 2, 3])
    reducer.xbar_ = np.zeros(4)
    reducer.scalings_ = np.eye(4, 2)
    reducer.explained_variance_ratio_ = np.array([0.75, 0.25])
    pipeline.reducer = reducer
    pipeline.transformed_cube_ = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    ratio_figure, ratio_axis, _ = plot_reducer_explained_variance(pipeline)
    scatter_figure, scatter_axis = plot_training_reducer_scatter(
        _tiny_bundle(), pipeline
    )

    assert "LDA" in ratio_axis.get_title()
    assert "LDA" in scatter_axis.get_title()
    plt.close(ratio_figure)
    plt.close(scatter_figure)
