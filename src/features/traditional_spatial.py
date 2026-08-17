"""Self-contained LBP and Gabor features for HSI center-pixel classification.

The reducers are fitted elsewhere using training centers only.  This module
then extracts label-free spatial context from the reduced full image.  LBP is
represented by a local histogram; Gabor responses are summarized by local
mean and standard deviation.  Neither extractor reads ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SpatialFeatureConfig:
    """Fixed feature settings used by the architecture comparison."""

    component_count: int = 3
    lbp_bins: int = 16
    pooling_window: int = 9
    gabor_orientations: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    gabor_frequencies: tuple[float, ...] = (0.1, 0.2)
    gabor_sigma: float = 2.5
    gabor_gamma: float = 0.5
    gabor_radius: int = 6

    def validate(self) -> None:
        if self.component_count < 1:
            raise ValueError("component_count must be positive")
        if self.lbp_bins < 1 or self.lbp_bins > 256 or 256 % self.lbp_bins:
            raise ValueError("lbp_bins must be a positive divisor of 256")
        if self.pooling_window < 1 or self.pooling_window % 2 == 0:
            raise ValueError("pooling_window must be a positive odd integer")
        if not self.gabor_orientations or not self.gabor_frequencies:
            raise ValueError("at least one Gabor orientation and frequency is required")
        if any(frequency <= 0 or frequency >= 0.5 for frequency in self.gabor_frequencies):
            raise ValueError("Gabor frequencies must lie in (0, 0.5)")
        if self.gabor_sigma <= 0 or self.gabor_gamma <= 0 or self.gabor_radius < 1:
            raise ValueError("Gabor sigma, gamma and radius must be positive")


def _validate_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or infinite values")
    return values


def _validate_coordinates(coordinates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape N x 2")
    if values.size and (
        np.any(values[:, 0] < 0)
        or np.any(values[:, 0] >= shape[0])
        or np.any(values[:, 1] < 0)
        or np.any(values[:, 1] >= shape[1])
    ):
        raise ValueError("coordinates contain values outside the image")
    return values


def local_binary_pattern_codes(image: np.ndarray) -> np.ndarray:
    """Return deterministic 8-neighbour, radius-1 LBP codes."""

    values = _validate_image(image)
    padded = np.pad(values, 1, mode="reflect")
    center = padded[1:-1, 1:-1]
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    )
    codes = np.zeros(values.shape, dtype=np.uint8)
    height, width = values.shape
    for bit, (row_offset, column_offset) in enumerate(offsets):
        neighbour = padded[
            1 + row_offset : 1 + row_offset + height,
            1 + column_offset : 1 + column_offset + width,
        ]
        codes |= ((neighbour >= center).astype(np.uint8) << bit)
    return codes


def lbp_local_histograms(
    image: np.ndarray,
    coordinates: np.ndarray,
    *,
    bins: int = 16,
    window_size: int = 9,
) -> np.ndarray:
    """Pool quantized LBP codes into a normalized local histogram."""

    if bins < 1 or bins > 256 or 256 % bins:
        raise ValueError("bins must be a positive divisor of 256")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    codes = local_binary_pattern_codes(image)
    points = _validate_coordinates(coordinates, codes.shape)
    quantized = codes.astype(np.int16) // (256 // bins)
    features = np.empty((points.shape[0], bins), dtype=np.float32)
    for bin_index in range(bins):
        density = ndimage.uniform_filter(
            (quantized == bin_index).astype(np.float32),
            size=window_size,
            mode="reflect",
        )
        features[:, bin_index] = density[points[:, 0], points[:, 1]]
    row_sums = features.sum(axis=1, keepdims=True)
    features /= np.maximum(row_sums, np.finfo(np.float32).eps)
    return features


def _gabor_kernel(
    frequency: float,
    theta_degrees: float,
    *,
    sigma: float,
    gamma: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)
    theta = np.deg2rad(theta_degrees)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(np.square(x_theta) + gamma**2 * np.square(y_theta)) / (2 * sigma**2))
    phase = 2 * np.pi * frequency * x_theta
    real = envelope * np.cos(phase)
    imaginary = envelope * np.sin(phase)
    real -= real.mean()
    imaginary -= imaginary.mean()
    norm = np.sqrt(np.square(real).sum() + np.square(imaginary).sum())
    if np.isclose(norm, 0.0):
        raise ValueError("degenerate Gabor kernel")
    return (real / norm).astype(np.float32), (imaginary / norm).astype(np.float32)


def gabor_local_statistics(
    image: np.ndarray,
    coordinates: np.ndarray,
    *,
    orientations: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    frequencies: Sequence[float] = (0.1, 0.2),
    sigma: float = 2.5,
    gamma: float = 0.5,
    radius: int = 6,
    window_size: int = 9,
) -> np.ndarray:
    """Return local mean/std of complex Gabor magnitude for each filter."""

    values = _validate_image(image)
    points = _validate_coordinates(coordinates, values.shape)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not orientations or not frequencies:
        raise ValueError("orientations and frequencies cannot be empty")
    output = np.empty(
        (points.shape[0], len(orientations) * len(frequencies) * 2),
        dtype=np.float32,
    )
    feature_index = 0
    for frequency in frequencies:
        for orientation in orientations:
            real_kernel, imaginary_kernel = _gabor_kernel(
                float(frequency),
                float(orientation),
                sigma=sigma,
                gamma=gamma,
                radius=radius,
            )
            real_response = ndimage.convolve(values, real_kernel, mode="reflect")
            imaginary_response = ndimage.convolve(values, imaginary_kernel, mode="reflect")
            magnitude = np.hypot(real_response, imaginary_response).astype(np.float32)
            local_mean = ndimage.uniform_filter(magnitude, size=window_size, mode="reflect")
            local_square_mean = ndimage.uniform_filter(
                np.square(magnitude), size=window_size, mode="reflect"
            )
            local_std = np.sqrt(
                np.maximum(local_square_mean - np.square(local_mean), 0.0)
            )
            rows, columns = points.T
            output[:, feature_index] = local_mean[rows, columns]
            output[:, feature_index + 1] = local_std[rows, columns]
            feature_index += 2
    if not np.isfinite(output).all():
        raise AssertionError("Gabor extraction produced non-finite values")
    return output


def build_traditional_features(
    transformed_cube: np.ndarray,
    coordinates: np.ndarray,
    *,
    include_spectral: bool = True,
    include_lbp: bool = False,
    include_gabor: bool = False,
    config: SpatialFeatureConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a labeled-center feature matrix from a reduced HSI cube."""

    settings = config or SpatialFeatureConfig()
    settings.validate()
    cube = np.asarray(transformed_cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[2] < 1 or not np.isfinite(cube).all():
        raise ValueError("transformed_cube must be a finite H x W x bands array")
    points = _validate_coordinates(coordinates, cube.shape[:2])
    if not any((include_spectral, include_lbp, include_gabor)):
        raise ValueError("at least one feature family must be enabled")

    matrices: list[np.ndarray] = []
    names: list[str] = []
    if include_spectral:
        matrices.append(cube[points[:, 0], points[:, 1], :])
        names.extend(f"spectral_{index + 1:02d}" for index in range(cube.shape[2]))

    component_count = min(settings.component_count, cube.shape[2])
    if include_lbp:
        for component in range(component_count):
            matrices.append(
                lbp_local_histograms(
                    cube[:, :, component],
                    points,
                    bins=settings.lbp_bins,
                    window_size=settings.pooling_window,
                )
            )
            names.extend(
                f"lbp_c{component + 1}_bin{bin_index:02d}"
                for bin_index in range(settings.lbp_bins)
            )

    if include_gabor:
        for component in range(component_count):
            matrices.append(
                gabor_local_statistics(
                    cube[:, :, component],
                    points,
                    orientations=settings.gabor_orientations,
                    frequencies=settings.gabor_frequencies,
                    sigma=settings.gabor_sigma,
                    gamma=settings.gabor_gamma,
                    radius=settings.gabor_radius,
                    window_size=settings.pooling_window,
                )
            )
            for frequency in settings.gabor_frequencies:
                for orientation in settings.gabor_orientations:
                    names.extend(
                        (
                            f"gabor_c{component + 1}_f{frequency:g}_t{orientation:g}_mean",
                            f"gabor_c{component + 1}_f{frequency:g}_t{orientation:g}_std",
                        )
                    )

    features = np.ascontiguousarray(np.concatenate(matrices, axis=1), dtype=np.float32)
    if features.shape != (points.shape[0], len(names)):
        raise AssertionError("feature names and matrix dimensions are inconsistent")
    if not np.isfinite(features).all():
        raise AssertionError("traditional feature matrix contains non-finite values")
    return features, names


__all__ = [
    "SpatialFeatureConfig",
    "build_traditional_features",
    "gabor_local_statistics",
    "lbp_local_histograms",
    "local_binary_pattern_codes",
]
