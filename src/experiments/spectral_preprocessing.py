"""Leakage-safe spectral preprocessing variants for controlled comparisons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.datasets.高光谱预处理 import BandStandardizer, PCASpectralReducer


VARIANT_KEYS = (
    "standard_pca15",
    "raw_pca15",
    "standard_pca15_whiten",
    "standard_uniform15",
    "standard_fisher15",
)


@dataclass(frozen=True)
class SpectralVariantResult:
    """One fitted transform and its full-image output."""

    key: str
    display_name: str
    description: str
    transformed_cube: np.ndarray
    metadata: dict[str, Any]
    state: dict[str, np.ndarray]
    fingerprint: str


def uniform_band_indices(total_bands: int, output_bands: int) -> np.ndarray:
    """Return deterministic, endpoint-preserving uniformly spaced band indices."""

    if total_bands < 1 or output_bands < 1 or output_bands > total_bands:
        raise ValueError("output_bands must be in the range 1..total_bands")
    indices = np.rint(np.linspace(0, total_bands - 1, output_bands)).astype(np.int64)
    if np.unique(indices).size != output_bands:
        raise AssertionError("uniform band selection produced duplicate indices")
    return indices


def fisher_band_scores(spectra: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-band Fisher scores from training samples only.

    The score is between-class scatter divided by within-class scatter.  It is
    invariant to a non-zero per-band affine scale, but the caller still passes
    the same standardized training matrix used by the model input pipeline.
    """

    values = np.asarray(spectra, dtype=np.float64)
    target = np.asarray(labels).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.size or values.shape[0] == 0:
        raise ValueError("spectra and labels must be non-empty and aligned")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("spectra and labels must be finite")
    classes = np.unique(target)
    if classes.size < 2:
        raise ValueError("Fisher scoring requires at least two classes")

    global_mean = values.mean(axis=0)
    between = np.zeros(values.shape[1], dtype=np.float64)
    within = np.zeros(values.shape[1], dtype=np.float64)
    for class_label in classes:
        class_values = values[target == class_label]
        class_mean = class_values.mean(axis=0)
        between += class_values.shape[0] * np.square(class_mean - global_mean)
        within += np.square(class_values - class_mean).sum(axis=0)
    epsilon = np.finfo(np.float64).eps
    scores = between / np.maximum(within, epsilon)
    if not np.isfinite(scores).all():
        raise AssertionError("Fisher scoring produced non-finite values")
    return scores


def top_fisher_band_indices(scores: np.ndarray, output_bands: int) -> np.ndarray:
    """Select top-scoring bands, returning them in original spectral order."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if output_bands < 1 or output_bands > values.size:
        raise ValueError("output_bands must be in the range 1..number of scores")
    # mergesort makes ties deterministic; the final ascending order preserves
    # wavelength adjacency for the 3D convolution.
    ranked = np.argsort(-values, kind="mergesort")[:output_bands]
    return np.sort(ranked.astype(np.int64))


def _hash_state(metadata: dict[str, Any], state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name in sorted(state):
        array = np.ascontiguousarray(state[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _transform_flat_in_chunks(
    flat_cube: np.ndarray,
    standardizer: BandStandardizer,
    *,
    reducer: PCASpectralReducer | None,
    selected_indices: np.ndarray | None,
    output_bands: int,
    chunk_size: int = 65_536,
) -> np.ndarray:
    output = np.empty((flat_cube.shape[0], output_bands), dtype=np.float32)
    for start in range(0, flat_cube.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_cube.shape[0])
        prepared = standardizer.transform(flat_cube[start:stop])
        if reducer is not None:
            transformed = reducer.transform(prepared)
        else:
            transformed = prepared[:, selected_indices]
        output[start:stop] = transformed.astype(np.float32, copy=False)
    return output


def fit_spectral_variant(
    cube: np.ndarray,
    train_coordinates: np.ndarray,
    train_labels: np.ndarray,
    variant_key: str,
    *,
    output_bands: int = 15,
) -> SpectralVariantResult:
    """Fit one comparison route on training centers and transform the full cube."""

    if variant_key not in VARIANT_KEYS:
        raise ValueError(f"unknown spectral variant: {variant_key!r}")
    cube_values = np.asarray(cube)
    coordinates = np.asarray(train_coordinates, dtype=np.int64)
    labels = np.asarray(train_labels).reshape(-1)
    if cube_values.ndim != 3:
        raise ValueError("cube must have shape height x width x bands")
    if coordinates.shape != (labels.size, 2) or labels.size == 0:
        raise ValueError("training coordinates and labels must be non-empty and aligned")
    rows, columns = coordinates.T
    train_spectra = cube_values[rows, columns, :].astype(np.float64)
    total_bands = cube_values.shape[-1]
    if output_bands > total_bands:
        raise ValueError("output_bands exceeds the raw spectral band count")

    standardization_enabled = variant_key != "raw_pca15"
    standardizer = BandStandardizer(enabled=standardization_enabled).fit(train_spectra)
    prepared_train = standardizer.transform(train_spectra)
    reducer: PCASpectralReducer | None = None
    selected_indices: np.ndarray | None = None
    fisher_scores: np.ndarray | None = None

    if variant_key in {"standard_pca15", "raw_pca15", "standard_pca15_whiten"}:
        whiten = variant_key == "standard_pca15_whiten"
        reducer = PCASpectralReducer(output_bands, whiten=whiten).fit(prepared_train)
        method = "pca"
    elif variant_key == "standard_uniform15":
        selected_indices = uniform_band_indices(total_bands, output_bands)
        method = "uniform_band_selection"
    else:
        fisher_scores = fisher_band_scores(prepared_train, labels)
        selected_indices = top_fisher_band_indices(fisher_scores, output_bands)
        method = "fisher_band_selection"

    names = {
        "standard_pca15": "标准化 + PCA15",
        "raw_pca15": "原始数值 + PCA15",
        "standard_pca15_whiten": "标准化 + PCA15 whitening",
        "standard_uniform15": "标准化 + 均匀15波段",
        "standard_fisher15": "标准化 + Fisher 15波段",
    }
    descriptions = {
        "standard_pca15": "逐波段标准化后使用训练集拟合 PCA，保留15个主成分。",
        "raw_pca15": "不做逐波段标准化；PCA仍执行自身的均值中心化。",
        "standard_pca15_whiten": "标准化、PCA15，并将各主成分除以其标准差。",
        "standard_uniform15": "不做线性降维，按波段序号均匀保留15个原始波段。",
        "standard_fisher15": "不做线性降维，只用训练标签的Fisher分数选择15个原始波段。",
    }
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "variant_key": variant_key,
        "display_name": names[variant_key],
        "description": descriptions[variant_key],
        "fit_scope": {
            "training_samples": int(labels.size),
            "validation_and_test_used_for_fit": False,
        },
        "input_bands": int(total_bands),
        "output_bands": int(output_bands),
        "standardization": "standard" if standardization_enabled else "none",
        "method": method,
        "whiten": bool(reducer.whiten) if reducer is not None else False,
        "selected_band_indices_zero_based": (
            selected_indices.tolist() if selected_indices is not None else None
        ),
        "selected_band_numbers_one_based": (
            (selected_indices + 1).tolist() if selected_indices is not None else None
        ),
    }
    if reducer is not None:
        metadata["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
        metadata["cumulative_explained_variance_ratio"] = float(
            reducer.explained_variance_ratio_.sum()
        )
    if fisher_scores is not None:
        metadata["selected_fisher_scores"] = fisher_scores[selected_indices].tolist()

    state: dict[str, np.ndarray] = {
        **standardizer.state(),
        "variant_key": np.asarray(variant_key),
        "output_bands": np.asarray(output_bands, dtype=np.int64),
    }
    if reducer is not None:
        state.update(reducer.state())
    else:
        state.update(
            {
                "reducer_name": np.asarray(method),
                "selected_band_indices": selected_indices,
            }
        )
        if fisher_scores is not None:
            state["fisher_scores"] = fisher_scores

    flat_cube = cube_values.reshape(-1, total_bands)
    transformed_flat = _transform_flat_in_chunks(
        flat_cube,
        standardizer,
        reducer=reducer,
        selected_indices=selected_indices,
        output_bands=output_bands,
    )
    transformed_cube = transformed_flat.reshape(
        cube_values.shape[0], cube_values.shape[1], output_bands
    )
    if not np.isfinite(transformed_cube).all():
        raise AssertionError("transformed cube contains non-finite values")
    fingerprint = _hash_state(metadata, state)
    metadata["fingerprint"] = fingerprint
    return SpectralVariantResult(
        key=variant_key,
        display_name=names[variant_key],
        description=descriptions[variant_key],
        transformed_cube=transformed_cube,
        metadata=metadata,
        state=state,
        fingerprint=fingerprint,
    )


__all__ = [
    "SpectralVariantResult",
    "VARIANT_KEYS",
    "fisher_band_scores",
    "fit_spectral_variant",
    "top_fisher_band_indices",
    "uniform_band_indices",
]
