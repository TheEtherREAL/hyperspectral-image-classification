import numpy as np
import pytest

from src.features.traditional_spatial import (
    SpatialFeatureConfig,
    build_traditional_features,
    gabor_local_statistics,
    lbp_local_histograms,
    local_binary_pattern_codes,
)


def test_lbp_codes_and_histograms_are_deterministic_and_normalized() -> None:
    image = np.arange(81, dtype=np.float32).reshape(9, 9)
    coordinates = np.asarray([[0, 0], [4, 4], [8, 8]], dtype=np.int64)
    first = local_binary_pattern_codes(image)
    second = local_binary_pattern_codes(image)
    np.testing.assert_array_equal(first, second)
    histograms = lbp_local_histograms(image, coordinates, bins=16, window_size=5)
    assert histograms.shape == (3, 16)
    np.testing.assert_allclose(histograms.sum(axis=1), 1.0, atol=1e-6)


def test_gabor_statistics_are_finite_and_have_expected_dimension() -> None:
    rows, columns = np.mgrid[:17, :19]
    image = np.sin(rows / 2.0) + np.cos(columns / 3.0)
    coordinates = np.asarray([[1, 1], [8, 9], [15, 17]], dtype=np.int64)
    output = gabor_local_statistics(
        image,
        coordinates,
        orientations=(0.0, 90.0),
        frequencies=(0.1, 0.2),
        radius=3,
        window_size=5,
    )
    assert output.shape == (3, 8)
    assert np.isfinite(output).all()
    assert np.all(output[:, 1::2] >= 0)


def test_feature_fusion_dimensions_and_names_match() -> None:
    rng = np.random.default_rng(1442)
    cube = rng.normal(size=(15, 16, 4)).astype(np.float32)
    coordinates = np.asarray([[2, 3], [7, 8], [12, 13]], dtype=np.int64)
    config = SpatialFeatureConfig(
        component_count=2,
        lbp_bins=8,
        pooling_window=5,
        gabor_orientations=(0.0, 90.0),
        gabor_frequencies=(0.1,),
        gabor_radius=3,
    )
    features, names = build_traditional_features(
        cube,
        coordinates,
        include_spectral=True,
        include_lbp=True,
        include_gabor=True,
        config=config,
    )
    assert features.shape == (3, 4 + 2 * 8 + 2 * 2 * 1 * 2)
    assert features.shape[1] == len(names)
    assert len(set(names)) == len(names)


def test_invalid_feature_request_is_rejected() -> None:
    cube = np.ones((3, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one"):
        build_traditional_features(
            cube,
            np.asarray([[1, 1]]),
            include_spectral=False,
        )
