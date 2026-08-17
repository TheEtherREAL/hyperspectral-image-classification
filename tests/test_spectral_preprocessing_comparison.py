"""Tests for the controlled spectral preprocessing comparison routes."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.spectral_preprocessing import (
    VARIANT_KEYS,
    fisher_band_scores,
    fit_spectral_variant,
    top_fisher_band_indices,
    uniform_band_indices,
)


def test_uniform_indices_preserve_endpoints_and_count() -> None:
    indices = uniform_band_indices(103, 15)
    assert indices[0] == 0
    assert indices[-1] == 102
    assert indices.size == np.unique(indices).size == 15
    assert np.all(np.diff(indices) > 0)


def test_fisher_scores_rank_discriminative_band() -> None:
    spectra = np.array(
        [
            [-5.0, 0.0, 2.0],
            [-4.0, 1.0, 2.0],
            [4.0, 0.0, 2.0],
            [5.0, 1.0, 2.0],
        ]
    )
    labels = np.array([1, 1, 2, 2])
    scores = fisher_band_scores(spectra, labels)
    assert int(np.argmax(scores)) == 0
    np.testing.assert_array_equal(top_fisher_band_indices(scores, 2), [0, 1])


@pytest.mark.parametrize("variant_key", VARIANT_KEYS)
def test_all_variants_fit_on_train_and_return_same_shape(variant_key: str) -> None:
    rng = np.random.default_rng(12)
    cube = rng.normal(size=(8, 7, 20)).astype(np.float32)
    coordinates = np.array([(row, column) for row in range(4) for column in range(5)])
    labels = np.tile(np.arange(1, 5), 5)
    result = fit_spectral_variant(
        cube,
        coordinates,
        labels,
        variant_key,
        output_bands=15,
    )
    assert result.transformed_cube.shape == (8, 7, 15)
    assert result.transformed_cube.dtype == np.float32
    assert np.isfinite(result.transformed_cube).all()
    assert result.metadata["fit_scope"] == {
        "training_samples": 20,
        "validation_and_test_used_for_fit": False,
    }
    assert len(result.fingerprint) == 64
