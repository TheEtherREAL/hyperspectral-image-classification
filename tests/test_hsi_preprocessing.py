from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.datasets.高光谱预处理 import (
    BandSelectionReducer,
    HSIDataBundle,
    HSIPreprocessingPipeline,
    HSITensorDataset,
    LDASpectralReducer,
    PreprocessingConfig,
)
from src.datasets.数据集注册 import DATASETS


def make_bundle(tmp_path: Path, cube: np.ndarray | None = None) -> HSIDataBundle:
    if cube is None:
        grid = np.arange(6 * 6 * 4, dtype=np.float64).reshape(6, 6, 4)
        cube = grid + np.array([0.0, 11.0, 31.0, 71.0])
    label_map = np.tile(np.array([1, 2, 1, 2, 1, 2], dtype=np.uint8), (6, 1))
    coordinates = np.argwhere(label_map > 0).astype(np.int32)
    labels = label_map[coordinates[:, 0], coordinates[:, 1]].astype(np.int16)
    indices = {
        "train": np.arange(0, 12, dtype=np.int64),
        "validation": np.arange(12, 18, dtype=np.int64),
        "test": np.arange(18, 36, dtype=np.int64),
    }

    raw_dir = tmp_path / "data" / "raw"
    split_dir = tmp_path / "data" / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "PaviaU.mat").write_bytes(b"synthetic cube")
    (raw_dir / "PaviaU_gt.mat").write_bytes(b"synthetic labels")
    split_path = split_dir / "pavia_university__fair24_6_70__seed1442.npz"
    split_path.write_bytes(b"synthetic fixed split")
    split_metadata_path = split_path.with_suffix(".json")
    split_metadata = {"protocol": {"name": "fair24_6_70", "seed": 1442}}
    split_metadata_path.write_text(json.dumps(split_metadata), encoding="utf-8")

    return HSIDataBundle(
        spec=DATASETS["pavia_university"],
        cube=np.asarray(cube),
        label_map=label_map,
        coordinates=coordinates,
        labels=labels,
        indices_by_split=indices,
        split_path=split_path,
        split_metadata_path=split_metadata_path,
        split_metadata=split_metadata,
    )


def config(**overrides) -> PreprocessingConfig:
    values = {
        "dataset_name": "pavia_university",
        "split_protocol": "fair24_6_70",
        "split_seed": 1442,
        "standardization": "standard",
        "reducer": "pca",
        "n_components": 3,
        "whiten": False,
        "representation": "patch",
        "patch_size": 3,
        "padding_mode": "constant",
        "padding_value": 0.0,
        "output_dtype": "float32",
    }
    values.update(overrides)
    return PreprocessingConfig(**values)


def test_standardization_and_pca_are_fitted_on_training_centers_only(tmp_path: Path) -> None:
    first_data = make_bundle(tmp_path / "first")
    changed_cube = first_data.cube.copy()
    nontraining = np.concatenate(
        (first_data.indices_by_split["validation"], first_data.indices_by_split["test"])
    )
    changed_coordinates = first_data.coordinates[nontraining]
    changed_cube[changed_coordinates[:, 0], changed_coordinates[:, 1], :] += 1_000_000
    second_data = make_bundle(tmp_path / "second", changed_cube)

    first = HSIPreprocessingPipeline(config()).fit(first_data)
    second = HSIPreprocessingPipeline(config()).fit(second_data)

    np.testing.assert_allclose(first.standardizer.mean_, second.standardizer.mean_)
    np.testing.assert_allclose(first.standardizer.scale_, second.standardizer.scale_)
    np.testing.assert_allclose(first.reducer.components_, second.reducer.components_)
    np.testing.assert_allclose(
        first.reducer.explained_variance_ratio_,
        second.reducer.explained_variance_ratio_,
    )
    assert first.standardizer.n_samples_seen_ == first_data.train_indices.size
    assert first.reducer.n_samples_seen_ == first_data.train_indices.size


def test_training_spectra_are_standardized_per_band(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    pipeline = HSIPreprocessingPipeline(config()).fit(data)
    rows, columns = data.train_coordinates.T
    standardized = pipeline.standardizer.transform(data.cube[rows, columns, :])

    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-12)


def test_pca_output_shape_and_finiteness(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    pipeline = HSIPreprocessingPipeline(config()).fit(data)

    assert pipeline.transformed_cube_.shape == (6, 6, 3)
    assert pipeline.transformed_cube_.dtype == np.float32
    assert np.isfinite(pipeline.transformed_cube_).all()
    assert pipeline.reducer.components_.shape == (3, 4)
    assert 0.0 < pipeline.reducer.explained_variance_ratio_.sum() <= 1.0 + 1e-12


def test_corner_patch_has_fixed_shape_center_and_constant_padding() -> None:
    transformed = np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2)
    coordinates = np.array([[0, 0]], dtype=np.int32)
    labels = np.array([2], dtype=np.int16)
    dataset = HSITensorDataset(
        transformed,
        coordinates,
        labels,
        np.array([0], dtype=np.int64),
        representation="patch",
        patch_size=3,
        padding_mode="constant",
        padding_value=-5.0,
    )

    sample = dataset[0]
    patch = sample["input"].numpy()
    assert patch.shape == (1, 2, 3, 3)
    np.testing.assert_array_equal(patch[0, :, 1, 1], transformed[0, 0, :])
    np.testing.assert_array_equal(patch[0, :, 0, 0], np.array([-5.0, -5.0]))
    assert sample["label"].item() == 1
    assert sample["raw_label"].item() == 2
    assert sample["coordinate"].tolist() == [0, 0]


def test_pixel_feature_splits_preserve_sample_identity(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    pipeline = HSIPreprocessingPipeline(
        config(representation="pixel", patch_size=1)
    ).fit(data)
    features = pipeline.build_feature_splits(data)

    train = features["train"]
    assert train.x.shape == (12, 3)
    assert train.y.shape == (12,)
    np.testing.assert_array_equal(train.y, train.raw_labels - 1)
    np.testing.assert_array_equal(train.coordinates, data.coordinates[train.sample_indices])


def test_torch_loader_shapes_labels_and_order_are_reproducible(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    pipeline = HSIPreprocessingPipeline(config()).fit(data)
    first = pipeline.build_torch_loaders(
        data, batch_size=5, loader_seed=77, num_workers=0, pin_memory=False
    )
    second = pipeline.build_torch_loaders(
        data, batch_size=5, loader_seed=77, num_workers=0, pin_memory=False
    )

    first_batch = next(iter(first["train"]))
    second_batch = next(iter(second["train"]))
    assert first_batch["input"].shape == (5, 1, 3, 3, 3)
    assert first_batch["input"].dtype == torch.float32
    assert first_batch["label"].dtype == torch.int64
    np.testing.assert_array_equal(first_batch["sample_index"], second_batch["sample_index"])
    np.testing.assert_array_equal(first_batch["label"], first_batch["raw_label"] - 1)


def test_paper_protocol_returns_no_validation_loader(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    paper_metadata = {"protocol": {"name": "paper30", "seed": 1442}}
    paper_data = HSIDataBundle(
        spec=data.spec,
        cube=data.cube,
        label_map=data.label_map,
        coordinates=data.coordinates,
        labels=data.labels,
        indices_by_split={
            "train": np.arange(0, 18, dtype=np.int64),
            "validation": np.empty(0, dtype=np.int64),
            "test": np.arange(18, 36, dtype=np.int64),
        },
        split_path=data.split_path,
        split_metadata_path=data.split_metadata_path,
        split_metadata=paper_metadata,
    )
    pipeline = HSIPreprocessingPipeline(
        config(split_protocol="paper30", representation="pixel", patch_size=1)
    ).fit(paper_data)
    loaders = pipeline.build_torch_loaders(
        paper_data, batch_size=4, loader_seed=1, num_workers=0, pin_memory=False
    )

    assert loaders["validation"] is None
    assert loaders["train"] is not None
    assert loaders["test"] is not None


def test_saved_state_round_trip_reproduces_transformation(tmp_path: Path) -> None:
    data = make_bundle(tmp_path / "data_root")
    fitted = HSIPreprocessingPipeline(config()).fit(data)
    paths = fitted.save_state(tmp_path / "state")
    loaded = HSIPreprocessingPipeline.load_state(paths["state"], paths["metadata"])
    loaded.attach_transformed_cube(data.cube)

    np.testing.assert_allclose(loaded.standardizer.mean_, fitted.standardizer.mean_)
    np.testing.assert_allclose(loaded.standardizer.scale_, fitted.standardizer.scale_)
    np.testing.assert_allclose(loaded.reducer.components_, fitted.reducer.components_)
    np.testing.assert_allclose(loaded.transformed_cube_, fitted.transformed_cube_)
    assert loaded.fit_metadata_["fit_scope"]["validation_and_test_used_for_fit"] is False


def test_lda_matches_sklearn_svd_transform(tmp_path: Path) -> None:
    data = make_bundle(tmp_path)
    pipeline = HSIPreprocessingPipeline(
        config(reducer="lda", n_components=1)
    ).fit(data)
    rows, columns = data.train_coordinates.T
    standardized_train = pipeline.standardizer.transform(data.cube[rows, columns, :])
    reference = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    reference.fit(standardized_train, data.train_labels)

    all_spectra = data.cube.reshape(-1, data.cube.shape[2])
    standardized_all = pipeline.standardizer.transform(all_spectra)
    expected = reference.transform(standardized_all)
    actual = pipeline.reducer.transform(standardized_all)

    assert isinstance(pipeline.reducer, LDASpectralReducer)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert pipeline.transformed_cube_.shape == (6, 6, 1)
    assert np.isfinite(pipeline.transformed_cube_).all()


def test_lda_is_fitted_on_training_spectra_and_labels_only(tmp_path: Path) -> None:
    first_data = make_bundle(tmp_path / "first")
    changed_cube = first_data.cube.copy()
    nontraining = np.concatenate(
        (first_data.indices_by_split["validation"], first_data.indices_by_split["test"])
    )
    changed_coordinates = first_data.coordinates[nontraining]
    changed_cube[changed_coordinates[:, 0], changed_coordinates[:, 1], :] += 1_000_000
    second_data = make_bundle(tmp_path / "second", changed_cube)

    first = HSIPreprocessingPipeline(
        config(reducer="lda", n_components=1)
    ).fit(first_data)
    second = HSIPreprocessingPipeline(
        config(reducer="lda", n_components=1)
    ).fit(second_data)

    np.testing.assert_allclose(first.standardizer.mean_, second.standardizer.mean_)
    np.testing.assert_allclose(first.reducer.xbar_, second.reducer.xbar_)
    np.testing.assert_allclose(first.reducer.scalings_, second.reducer.scalings_)
    np.testing.assert_array_equal(first.reducer.classes_, second.reducer.classes_)
    assert first.reducer.n_samples_seen_ == first_data.train_indices.size


def test_lda_saved_state_round_trip_reproduces_transformation(tmp_path: Path) -> None:
    data = make_bundle(tmp_path / "data_root")
    fitted = HSIPreprocessingPipeline(
        config(reducer="lda", n_components=1)
    ).fit(data)
    paths = fitted.save_state(tmp_path / "state")
    loaded = HSIPreprocessingPipeline.load_state(paths["state"], paths["metadata"])
    loaded.attach_transformed_cube(data.cube)

    assert isinstance(loaded.reducer, LDASpectralReducer)
    np.testing.assert_allclose(loaded.reducer.xbar_, fitted.reducer.xbar_)
    np.testing.assert_allclose(loaded.reducer.scalings_, fitted.reducer.scalings_)
    np.testing.assert_array_equal(loaded.reducer.classes_, fitted.reducer.classes_)
    np.testing.assert_allclose(loaded.transformed_cube_, fitted.transformed_cube_)
    assert loaded.fit_metadata_["spectral_reducer"]["supervised"] is True


def test_one_line_yaml_reducer_switch_selects_method_specific_parameters() -> None:
    values = {
        "dataset": {
            "name": "pavia_university",
            "split_protocol": "fair24_6_70",
            "split_seed": 1442,
        },
        "spectral_preprocessing": {
            "standardization": "standard",
            "reducer": "pca",
            "pca": {"n_components": 15, "whiten": False},
            "lda": {"n_components": 8},
        },
        "spatial_preprocessing": {"representation": "patch", "patch_size": 25},
    }
    pca_config = PreprocessingConfig.from_mapping(values)
    values["spectral_preprocessing"]["reducer"] = "LDA"
    lda_config = PreprocessingConfig.from_mapping(values)

    assert pca_config.n_components == 15
    assert pca_config.route_name().endswith("standard_pca15_patch25")
    assert lda_config.n_components == 8
    assert lda_config.whiten is False
    assert lda_config.route_name().endswith("standard_lda8_patch25")


def test_lda_rejects_more_than_classes_minus_one_components() -> None:
    with pytest.raises(ValueError, match="classes-1=8"):
        config(reducer="lda", n_components=9).validate()


@pytest.mark.parametrize("method", ["uniform", "fisher"])
def test_band_selection_routes_are_fitted_and_round_trip(
    tmp_path: Path,
    method: str,
) -> None:
    data = make_bundle(tmp_path / method)
    fitted = HSIPreprocessingPipeline(
        config(
            reducer="band_selection",
            n_components=3,
            band_selection_method=method,
        )
    ).fit(data)

    assert isinstance(fitted.reducer, BandSelectionReducer)
    assert fitted.reducer.method == method
    assert fitted.reducer.n_samples_seen_ == data.train_indices.size
    assert fitted.reducer.selected_indices_.shape == (3,)
    assert np.all(np.diff(fitted.reducer.selected_indices_) > 0)
    assert fitted.transformed_cube_.shape == (*data.cube.shape[:2], 3)

    paths = fitted.save_state(tmp_path / f"state_{method}")
    loaded = HSIPreprocessingPipeline.load_state(paths["state"], paths["metadata"])
    loaded.attach_transformed_cube(data.cube)
    np.testing.assert_array_equal(
        loaded.reducer.selected_indices_, fitted.reducer.selected_indices_
    )
    np.testing.assert_allclose(loaded.transformed_cube_, fitted.transformed_cube_)


@pytest.mark.parametrize("representation", ["lbp", "gabor"])
def test_planned_representations_fail_explicitly_instead_of_silently(
    representation: str,
) -> None:
    with pytest.raises(NotImplementedError, match="later route"):
        config(representation=representation).validate()


def test_even_patch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be odd"):
        config(patch_size=4).validate()
