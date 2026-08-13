"""无数据泄漏、配置驱动的高光谱预处理核心 / HSI preprocessing core.

This module owns data preparation only. Model definitions, optimizers, losses,
training loops and evaluation metrics deliberately live elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.data import DataLoader, Dataset

from .数据读取 import load_dataset
from .数据集注册 import DATASETS, DatasetSpec


PREPROCESSING_SCHEMA_VERSION = "1.0"
SUPPORTED_STANDARDIZATION = {"none", "standard"}
SUPPORTED_REDUCERS = {"none", "pca", "lda"}
PLANNED_REDUCERS = {"band_selection"}
SUPPORTED_REPRESENTATIONS = {"pixel", "patch"}
PLANNED_REPRESENTATIONS = {"lbp", "gabor"}
SUPPORTED_PADDING_MODES = {"constant", "reflect", "edge"}
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for one reproducible preprocessing route."""

    dataset_name: str = "pavia_university"
    split_protocol: str = "fair24_6_70"
    split_seed: int = 345
    standardization: str = "standard"
    reducer: str = "pca"
    n_components: int | None = 15
    whiten: bool = False
    representation: str = "patch"
    patch_size: int = 25
    padding_mode: str = "constant"
    padding_value: float = 0.0
    output_dtype: str = "float32"

    def validate(self) -> None:
        if self.dataset_name not in DATASETS:
            raise ValueError(f"unknown dataset: {self.dataset_name!r}")
        if self.split_protocol not in {"paper30", "fair24_6_70"}:
            raise ValueError(f"unsupported split protocol: {self.split_protocol!r}")
        if not isinstance(self.split_seed, int) or self.split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        if self.standardization not in SUPPORTED_STANDARDIZATION:
            raise ValueError(
                f"standardization must be one of {sorted(SUPPORTED_STANDARDIZATION)}"
            )
        if self.reducer in PLANNED_REDUCERS:
            raise NotImplementedError(
                f"reducer {self.reducer!r} is reserved for a later comparison route"
            )
        if self.reducer not in SUPPORTED_REDUCERS:
            raise ValueError(f"unsupported reducer: {self.reducer!r}")
        if self.reducer in {"pca", "lda"} and (
            self.n_components is None or self.n_components < 1
        ):
            raise ValueError(
                f"{self.reducer.upper()} requires a positive n_components"
            )
        if self.reducer == "lda":
            maximum = len(DATASETS[self.dataset_name].class_names) - 1
            if self.n_components > maximum:
                raise ValueError(
                    f"LDA n_components={self.n_components} exceeds "
                    f"classes-1={maximum} for {self.dataset_name}"
                )
            if self.whiten:
                raise ValueError("whiten is a PCA-only option and must be false for LDA")
        if self.reducer == "none" and self.n_components is not None:
            raise ValueError("n_components must be null when reducer='none'")
        if self.representation in PLANNED_REPRESENTATIONS:
            raise NotImplementedError(
                f"representation {self.representation!r} is reserved for a later route"
            )
        if self.representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"unsupported representation: {self.representation!r}")
        if not isinstance(self.patch_size, int) or self.patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        if self.representation == "patch" and self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd so that it has one center pixel")
        if self.padding_mode not in SUPPORTED_PADDING_MODES:
            raise ValueError(f"unsupported padding mode: {self.padding_mode!r}")
        if self.output_dtype not in {"float32", "float64"}:
            raise ValueError("output_dtype must be 'float32' or 'float64'")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PreprocessingConfig":
        """Build a config from the project's nested YAML-style mapping."""
        dataset = values.get("dataset", {})
        spectral = values.get("spectral_preprocessing", values.get("preprocessing", {}))
        spatial = values.get("spatial_preprocessing", values.get("preprocessing", {}))
        reducer = str(spectral.get("reducer", "pca")).strip().lower()
        if reducer == "pca":
            reducer_values = spectral.get("pca", {})
            n_components = reducer_values.get(
                "n_components", spectral.get("n_components", 15)
            )
            whiten = bool(reducer_values.get("whiten", spectral.get("whiten", False)))
        elif reducer == "lda":
            reducer_values = spectral.get("lda", {})
            n_components = reducer_values.get(
                "n_components", spectral.get("n_components", 8)
            )
            whiten = False
        else:
            n_components = None
            whiten = False
        config = cls(
            dataset_name=dataset.get("name", "pavia_university"),
            split_protocol=dataset.get("split_protocol", "fair24_6_70"),
            split_seed=int(dataset.get("split_seed", 345)),
            standardization=spectral.get(
                "standardization", dataset.get("normalization", "standard")
            ),
            reducer=reducer,
            n_components=n_components,
            whiten=whiten,
            representation=spatial.get("representation", "patch"),
            patch_size=int(spatial.get("patch_size", 25)),
            padding_mode=spatial.get("padding_mode", "constant"),
            padding_value=float(spatial.get("padding_value", 0.0)),
            output_dtype=values.get("output", {}).get("dtype", "float32"),
        )
        config.validate()
        return config

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def route_name(self) -> str:
        reducer = self.reducer if self.reducer == "none" else f"{self.reducer}{self.n_components}"
        representation = (
            "pixel" if self.representation == "pixel" else f"patch{self.patch_size}"
        )
        return (
            f"{self.split_protocol}__seed{self.split_seed}__"
            f"{self.standardization}_{reducer}_{representation}"
        )


@dataclass(frozen=True)
class HSIDataBundle:
    """Raw cube plus one already-frozen labeled-pixel split."""

    spec: DatasetSpec
    cube: np.ndarray
    label_map: np.ndarray
    coordinates: np.ndarray
    labels: np.ndarray
    indices_by_split: dict[str, np.ndarray]
    split_path: Path
    split_metadata_path: Path
    split_metadata: dict[str, Any]

    @property
    def train_indices(self) -> np.ndarray:
        return self.indices_by_split["train"]

    @property
    def train_coordinates(self) -> np.ndarray:
        return self.coordinates[self.train_indices]

    @property
    def train_labels(self) -> np.ndarray:
        return self.labels[self.train_indices]


@dataclass(frozen=True)
class FeatureSplit:
    """Traditional-ML feature matrix with traceable sample identity."""

    x: np.ndarray
    y: np.ndarray
    raw_labels: np.ndarray
    coordinates: np.ndarray
    sample_indices: np.ndarray


def _scalar(array: np.ndarray) -> Any:
    return np.asarray(array).item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hsi_data(project_root: Path, config: PreprocessingConfig) -> HSIDataBundle:
    """Load raw data and an existing fixed split without regenerating it."""
    config.validate()
    project_root = Path(project_root).resolve()
    raw_dir = project_root / "data" / "raw"
    split_dir = project_root / "data" / "splits"
    spec = DATASETS[config.dataset_name]
    cube, label_map = load_dataset(raw_dir, spec)

    stem = f"{config.dataset_name}__{config.split_protocol}__seed{config.split_seed}"
    split_path = split_dir / f"{stem}.npz"
    split_metadata_path = split_dir / f"{stem}.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not split_metadata_path.is_file():
        raise FileNotFoundError(split_metadata_path)

    with np.load(split_path, allow_pickle=False) as artifact:
        if str(_scalar(artifact["dataset_name"])) != config.dataset_name:
            raise ValueError("split dataset does not match preprocessing config")
        if str(_scalar(artifact["protocol_name"])) != config.split_protocol:
            raise ValueError("split protocol does not match preprocessing config")
        if int(_scalar(artifact["seed"])) != config.split_seed:
            raise ValueError("split seed does not match preprocessing config")
        coordinates = artifact["coordinates"].astype(np.int32, copy=True)
        labels = artifact["labels"].astype(np.int16, copy=True)
        indices_by_split = {
            name: artifact[f"{name}_indices"].astype(np.int64, copy=True)
            for name in SPLIT_NAMES
        }

    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    if split_metadata["protocol"]["name"] != config.split_protocol:
        raise ValueError("split JSON protocol does not match preprocessing config")
    if int(split_metadata["protocol"]["seed"]) != config.split_seed:
        raise ValueError("split JSON seed does not match preprocessing config")

    if coordinates.shape != (labels.size, 2):
        raise ValueError("fixed split coordinates and labels are not aligned")
    rows, columns = coordinates.T
    if np.any(rows < 0) or np.any(rows >= label_map.shape[0]):
        raise ValueError("fixed split contains an out-of-range row")
    if np.any(columns < 0) or np.any(columns >= label_map.shape[1]):
        raise ValueError("fixed split contains an out-of-range column")
    np.testing.assert_array_equal(labels, label_map[rows, columns])
    if np.any(labels == 0):
        raise ValueError("background pixels must not enter preprocessing")

    assigned = np.concatenate(list(indices_by_split.values()))
    if assigned.size != labels.size or not np.array_equal(
        np.sort(assigned), np.arange(labels.size)
    ):
        raise ValueError("fixed split must cover every labeled sample exactly once")
    if np.unique(assigned).size != labels.size:
        raise ValueError("fixed split partitions overlap")

    return HSIDataBundle(
        spec=spec,
        cube=cube,
        label_map=label_map,
        coordinates=coordinates,
        labels=labels,
        indices_by_split=indices_by_split,
        split_path=split_path,
        split_metadata_path=split_metadata_path,
        split_metadata=split_metadata,
    )


class BandStandardizer:
    """Per-band standardization fitted exclusively on training spectra."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples_seen_: int | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None and self.scale_ is not None

    def fit(self, spectra: np.ndarray) -> "BandStandardizer":
        values = _validate_spectral_matrix(spectra)
        if self.enabled:
            mean = values.mean(axis=0, dtype=np.float64)
            variance = np.mean(np.square(values - mean), axis=0, dtype=np.float64)
            scale = np.sqrt(variance)
            scale[scale == 0] = 1.0
        else:
            mean = np.zeros(values.shape[1], dtype=np.float64)
            scale = np.ones(values.shape[1], dtype=np.float64)
        self.mean_ = mean
        self.scale_ = scale
        self.n_samples_seen_ = int(values.shape[0])
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("standardizer must be fitted before transform")
        values = _validate_spectral_matrix(spectra)
        if values.shape[1] != self.mean_.size:
            raise ValueError("spectral band count does not match fitted standardizer")
        return (values - self.mean_) / self.scale_

    def state(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("standardizer has no fitted state")
        return {
            "standardizer_enabled": np.asarray(self.enabled),
            "standardizer_mean": self.mean_,
            "standardizer_scale": self.scale_,
            "standardizer_n_samples_seen": np.asarray(self.n_samples_seen_, dtype=np.int64),
        }

    @classmethod
    def from_state(cls, artifact: Mapping[str, np.ndarray]) -> "BandStandardizer":
        instance = cls(enabled=bool(_scalar(artifact["standardizer_enabled"])))
        instance.mean_ = np.asarray(artifact["standardizer_mean"], dtype=np.float64).copy()
        instance.scale_ = np.asarray(artifact["standardizer_scale"], dtype=np.float64).copy()
        instance.n_samples_seen_ = int(_scalar(artifact["standardizer_n_samples_seen"]))
        return instance


class IdentityReducer:
    """No spectral reduction; preserves the standardized band vector."""

    name = "none"

    def __init__(self) -> None:
        self.n_features_in_: int | None = None

    @property
    def fitted(self) -> bool:
        return self.n_features_in_ is not None

    @property
    def n_output_features(self) -> int:
        if not self.fitted:
            raise RuntimeError("reducer has not been fitted")
        return int(self.n_features_in_)

    def fit(self, spectra: np.ndarray, labels: np.ndarray | None = None) -> "IdentityReducer":
        values = _validate_spectral_matrix(spectra)
        self.n_features_in_ = int(values.shape[1])
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        values = _validate_spectral_matrix(spectra)
        if not self.fitted or values.shape[1] != self.n_features_in_:
            raise ValueError("spectral band count does not match fitted identity reducer")
        return values

    def state(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("reducer has no fitted state")
        return {
            "reducer_name": np.asarray(self.name),
            "reducer_n_features_in": np.asarray(self.n_features_in_, dtype=np.int64),
        }

    @classmethod
    def from_state(cls, artifact: Mapping[str, np.ndarray]) -> "IdentityReducer":
        instance = cls()
        instance.n_features_in_ = int(_scalar(artifact["reducer_n_features_in"]))
        return instance


class PCASpectralReducer:
    """PCA fitted on standardized training-center spectra only."""

    name = "pca"

    def __init__(self, n_components: int, *, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int | None = None
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.components_ is not None

    @property
    def n_output_features(self) -> int:
        if not self.fitted:
            raise RuntimeError("reducer has not been fitted")
        return int(self.components_.shape[0])

    def fit(
        self,
        spectra: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> "PCASpectralReducer":
        values = _validate_spectral_matrix(spectra)
        maximum = min(values.shape)
        if self.n_components > maximum:
            raise ValueError(
                f"PCA n_components={self.n_components} exceeds min(samples, bands)={maximum}"
            )
        estimator = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver="full",
        )
        estimator.fit(values)
        self.n_features_in_ = int(estimator.n_features_in_)
        self.n_samples_seen_ = int(values.shape[0])
        self.components_ = estimator.components_.astype(np.float64, copy=True)
        self.mean_ = estimator.mean_.astype(np.float64, copy=True)
        self.explained_variance_ = estimator.explained_variance_.astype(np.float64, copy=True)
        self.explained_variance_ratio_ = estimator.explained_variance_ratio_.astype(
            np.float64, copy=True
        )
        self.singular_values_ = estimator.singular_values_.astype(np.float64, copy=True)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PCA reducer must be fitted before transform")
        values = _validate_spectral_matrix(spectra)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("spectral band count does not match fitted PCA")
        transformed = (values - self.mean_) @ self.components_.T
        if self.whiten:
            transformed = transformed / np.sqrt(self.explained_variance_)
        return transformed

    def state(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("reducer has no fitted state")
        return {
            "reducer_name": np.asarray(self.name),
            "reducer_n_features_in": np.asarray(self.n_features_in_, dtype=np.int64),
            "reducer_n_samples_seen": np.asarray(self.n_samples_seen_, dtype=np.int64),
            "reducer_n_components": np.asarray(self.n_components, dtype=np.int64),
            "reducer_whiten": np.asarray(self.whiten),
            "reducer_components": self.components_,
            "reducer_mean": self.mean_,
            "reducer_explained_variance": self.explained_variance_,
            "reducer_explained_variance_ratio": self.explained_variance_ratio_,
            "reducer_singular_values": self.singular_values_,
        }

    @classmethod
    def from_state(cls, artifact: Mapping[str, np.ndarray]) -> "PCASpectralReducer":
        instance = cls(
            n_components=int(_scalar(artifact["reducer_n_components"])),
            whiten=bool(_scalar(artifact["reducer_whiten"])),
        )
        instance.n_features_in_ = int(_scalar(artifact["reducer_n_features_in"]))
        instance.n_samples_seen_ = int(_scalar(artifact["reducer_n_samples_seen"]))
        instance.components_ = np.asarray(artifact["reducer_components"], dtype=np.float64).copy()
        instance.mean_ = np.asarray(artifact["reducer_mean"], dtype=np.float64).copy()
        instance.explained_variance_ = np.asarray(
            artifact["reducer_explained_variance"], dtype=np.float64
        ).copy()
        instance.explained_variance_ratio_ = np.asarray(
            artifact["reducer_explained_variance_ratio"], dtype=np.float64
        ).copy()
        instance.singular_values_ = np.asarray(
            artifact["reducer_singular_values"], dtype=np.float64
        ).copy()
        return instance


class LDASpectralReducer:
    """Supervised LDA fitted on standardized training spectra and labels only.

    The implementation follows scikit-learn's recommended ``solver='svd'``
    route.  Only the fitted projection state required by ``transform`` is
    persisted, so a loaded pipeline does not need to refit or see labels.
    """

    name = "lda"
    solver = "svd"

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int | None = None
        self.max_components_: int | None = None
        self.classes_: np.ndarray | None = None
        self.xbar_: np.ndarray | None = None
        self.scalings_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.xbar_ is not None and self.scalings_ is not None

    @property
    def n_output_features(self) -> int:
        if not self.fitted:
            raise RuntimeError("reducer has not been fitted")
        return int(self.max_components_)

    def fit(
        self,
        spectra: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> "LDASpectralReducer":
        values = _validate_spectral_matrix(spectra)
        if labels is None:
            raise ValueError("LDA requires training labels")
        target = np.asarray(labels)
        if target.ndim != 1 or target.size != values.shape[0]:
            raise ValueError("LDA labels must be a vector aligned with spectra")
        if not np.isfinite(target).all():
            raise ValueError("LDA labels contain NaN or infinite values")
        classes = np.unique(target)
        if classes.size < 2:
            raise ValueError("LDA requires at least two training classes")
        maximum = min(classes.size - 1, values.shape[1])
        if self.n_components > maximum:
            raise ValueError(
                f"LDA n_components={self.n_components} exceeds "
                f"min(classes-1, bands)={maximum}"
            )

        estimator = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver,
        )
        estimator.fit(values, target)
        self.n_features_in_ = int(estimator.n_features_in_)
        self.n_samples_seen_ = int(values.shape[0])
        self.max_components_ = int(estimator._max_components)
        self.classes_ = np.asarray(estimator.classes_).copy()
        self.xbar_ = estimator.xbar_.astype(np.float64, copy=True)
        self.scalings_ = estimator.scalings_.astype(np.float64, copy=True)
        self.explained_variance_ratio_ = estimator.explained_variance_ratio_.astype(
            np.float64, copy=True
        )
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LDA reducer must be fitted before transform")
        values = _validate_spectral_matrix(spectra)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("spectral band count does not match fitted LDA")
        transformed = (values - self.xbar_) @ self.scalings_
        return transformed[:, : self.max_components_]

    def state(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("reducer has no fitted state")
        return {
            "reducer_name": np.asarray(self.name),
            "reducer_solver": np.asarray(self.solver),
            "reducer_n_features_in": np.asarray(self.n_features_in_, dtype=np.int64),
            "reducer_n_samples_seen": np.asarray(self.n_samples_seen_, dtype=np.int64),
            "reducer_n_components": np.asarray(self.n_components, dtype=np.int64),
            "reducer_max_components": np.asarray(self.max_components_, dtype=np.int64),
            "reducer_classes": self.classes_,
            "reducer_xbar": self.xbar_,
            "reducer_scalings": self.scalings_,
            "reducer_explained_variance_ratio": self.explained_variance_ratio_,
        }

    @classmethod
    def from_state(cls, artifact: Mapping[str, np.ndarray]) -> "LDASpectralReducer":
        solver = str(_scalar(artifact["reducer_solver"]))
        if solver != cls.solver:
            raise ValueError(f"unsupported saved LDA solver: {solver!r}")
        instance = cls(n_components=int(_scalar(artifact["reducer_n_components"])))
        instance.n_features_in_ = int(_scalar(artifact["reducer_n_features_in"]))
        instance.n_samples_seen_ = int(_scalar(artifact["reducer_n_samples_seen"]))
        instance.max_components_ = int(_scalar(artifact["reducer_max_components"]))
        instance.classes_ = np.asarray(artifact["reducer_classes"]).copy()
        instance.xbar_ = np.asarray(artifact["reducer_xbar"], dtype=np.float64).copy()
        instance.scalings_ = np.asarray(
            artifact["reducer_scalings"], dtype=np.float64
        ).copy()
        instance.explained_variance_ratio_ = np.asarray(
            artifact["reducer_explained_variance_ratio"], dtype=np.float64
        ).copy()
        return instance


def _validate_spectral_matrix(spectra: np.ndarray) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected samples×bands matrix, got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("spectral matrix must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("spectral matrix contains NaN or infinite values")
    return values


def _build_reducer(
    config: PreprocessingConfig,
) -> IdentityReducer | PCASpectralReducer | LDASpectralReducer:
    if config.reducer == "none":
        return IdentityReducer()
    if config.reducer == "pca":
        return PCASpectralReducer(int(config.n_components), whiten=config.whiten)
    if config.reducer == "lda":
        return LDASpectralReducer(int(config.n_components))
    raise AssertionError(f"validated reducer is not implemented: {config.reducer}")


class HSITensorDataset(Dataset):
    """On-demand pixel or patch tensors for one frozen split partition."""

    def __init__(
        self,
        transformed_cube: np.ndarray,
        coordinates: np.ndarray,
        raw_labels: np.ndarray,
        sample_indices: np.ndarray,
        *,
        representation: str,
        patch_size: int,
        padding_mode: str,
        padding_value: float,
    ) -> None:
        self.transformed_cube = np.asarray(transformed_cube)
        self.coordinates = np.asarray(coordinates, dtype=np.int32)
        self.raw_labels = np.asarray(raw_labels, dtype=np.int16)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.representation = representation
        self.patch_size = patch_size
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.radius = patch_size // 2
        self._padded_cube: np.ndarray | None = None

        if self.transformed_cube.ndim != 3:
            raise ValueError("transformed_cube must have shape H×W×features")
        if self.coordinates.shape != (self.raw_labels.size, 2):
            raise ValueError("coordinates and raw_labels must be aligned")
        if representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"unsupported representation: {representation!r}")
        if representation == "patch":
            padding = ((self.radius, self.radius), (self.radius, self.radius), (0, 0))
            kwargs = {"constant_values": padding_value} if padding_mode == "constant" else {}
            self._padded_cube = np.pad(
                self.transformed_cube,
                padding,
                mode=padding_mode,
                **kwargs,
            )

    def __len__(self) -> int:
        return int(self.sample_indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample_index = int(self.sample_indices[item])
        row, column = self.coordinates[sample_index]
        raw_label = int(self.raw_labels[sample_index])

        if self.representation == "pixel":
            model_input = np.ascontiguousarray(self.transformed_cube[row, column, :])
        else:
            patch = self._padded_cube[
                row : row + self.patch_size,
                column : column + self.patch_size,
                :,
            ]
            if patch.shape[:2] != (self.patch_size, self.patch_size):
                raise AssertionError("patch extraction produced an unexpected spatial shape")
            model_input = np.ascontiguousarray(patch.transpose(2, 0, 1)[None, ...])

        return {
            "input": torch.from_numpy(model_input),
            "label": torch.tensor(raw_label - 1, dtype=torch.long),
            "raw_label": torch.tensor(raw_label, dtype=torch.long),
            "coordinate": torch.tensor((int(row), int(column)), dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }


class HSIPreprocessingPipeline:
    """Fit-on-train pipeline shared by traditional and PyTorch routes."""

    def __init__(self, config: PreprocessingConfig) -> None:
        config.validate()
        self.config = config
        self.standardizer = BandStandardizer(enabled=config.standardization == "standard")
        self.reducer = _build_reducer(config)
        self.transformed_cube_: np.ndarray | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    @property
    def fitted(self) -> bool:
        return self.standardizer.fitted and self.reducer.fitted

    @property
    def output_bands(self) -> int:
        return self.reducer.n_output_features

    def fit(self, data: HSIDataBundle) -> "HSIPreprocessingPipeline":
        """Fit all statistical transforms on training-center spectra only."""
        if data.spec.name != self.config.dataset_name:
            raise ValueError("data bundle dataset does not match pipeline config")
        if data.split_metadata["protocol"]["name"] != self.config.split_protocol:
            raise ValueError("data bundle protocol does not match pipeline config")
        if int(data.split_metadata["protocol"]["seed"]) != self.config.split_seed:
            raise ValueError("data bundle seed does not match pipeline config")

        train_rows, train_columns = data.train_coordinates.T
        train_spectra = data.cube[train_rows, train_columns, :]
        self.standardizer.fit(train_spectra)
        standardized_train = self.standardizer.transform(train_spectra)
        self.reducer.fit(standardized_train, data.train_labels)
        self.transformed_cube_ = self.transform_cube(data.cube)
        self.fit_metadata_ = self._build_fit_metadata(data)
        return self

    def transform_spectra(self, spectra: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("pipeline must be fitted before transform")
        return self.reducer.transform(self.standardizer.transform(spectra))

    def transform_cube(self, cube: np.ndarray, *, chunk_size: int = 32768) -> np.ndarray:
        """Apply fitted transforms to a full cube in bounded-memory chunks."""
        if not self.fitted:
            raise RuntimeError("pipeline must be fitted before transform_cube")
        cube = np.asarray(cube)
        if cube.ndim != 3:
            raise ValueError("cube must have shape H×W×bands")
        flat = cube.reshape(-1, cube.shape[2])
        output = np.empty(
            (flat.shape[0], self.output_bands),
            dtype=np.dtype(self.config.output_dtype),
        )
        for start in range(0, flat.shape[0], chunk_size):
            stop = min(start + chunk_size, flat.shape[0])
            output[start:stop] = self.transform_spectra(flat[start:stop]).astype(
                self.config.output_dtype, copy=False
            )
        if not np.isfinite(output).all():
            raise ValueError("transformed cube contains NaN or infinite values")
        return output.reshape(cube.shape[0], cube.shape[1], self.output_bands)

    def build_feature_splits(self, data: HSIDataBundle) -> dict[str, FeatureSplit]:
        """Return center-pixel matrices for SVM/XGBoost-style models."""
        self._require_transformed_cube()
        if self.config.representation != "pixel":
            raise ValueError(
                "traditional feature matrices require representation='pixel'; "
                "LBP/Gabor will be added through dedicated feature extractors later"
            )
        features = self.transformed_cube_[
            data.coordinates[:, 0], data.coordinates[:, 1], :
        ]
        result: dict[str, FeatureSplit] = {}
        for split_name, indices in data.indices_by_split.items():
            result[split_name] = FeatureSplit(
                x=np.ascontiguousarray(features[indices]),
                y=(data.labels[indices] - 1).astype(np.int64, copy=False),
                raw_labels=data.labels[indices].copy(),
                coordinates=data.coordinates[indices].copy(),
                sample_indices=indices.copy(),
            )
        return result

    def build_torch_datasets(self, data: HSIDataBundle) -> dict[str, HSITensorDataset | None]:
        self._require_transformed_cube()
        datasets: dict[str, HSITensorDataset | None] = {}
        for split_name, indices in data.indices_by_split.items():
            datasets[split_name] = None if indices.size == 0 else HSITensorDataset(
                self.transformed_cube_,
                data.coordinates,
                data.labels,
                indices,
                representation=self.config.representation,
                patch_size=self.config.patch_size,
                padding_mode=self.config.padding_mode,
                padding_value=self.config.padding_value,
            )
        return datasets

    def build_torch_loaders(
        self,
        data: HSIDataBundle,
        *,
        batch_size: int,
        loader_seed: int,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ) -> dict[str, DataLoader | None]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        datasets = self.build_torch_datasets(data)
        generator = torch.Generator().manual_seed(loader_seed)
        loaders: dict[str, DataLoader | None] = {}
        for split_name, dataset in datasets.items():
            loaders[split_name] = None if dataset is None else DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split_name == "train",
                num_workers=num_workers,
                pin_memory=pin_memory,
                generator=generator if split_name == "train" else None,
                drop_last=False,
            )
        return loaders

    def save_state(self, output_dir: Path, *, overwrite: bool = False) -> dict[str, Path]:
        """Save fitted parameters and metadata, but not duplicated full cubes/patches."""
        if not self.fitted or self.fit_metadata_ is None:
            raise RuntimeError("pipeline must be fitted before saving")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "state": output_dir / "preprocessing_state.npz",
            "metadata": output_dir / "metadata.json",
        }
        if not overwrite:
            existing = [str(path) for path in paths.values() if path.exists()]
            if existing:
                raise FileExistsError(f"refusing to overwrite preprocessing artifacts: {existing}")

        state = {
            "schema_version": np.asarray(PREPROCESSING_SCHEMA_VERSION),
            "config_fingerprint": np.asarray(self.config.fingerprint()),
            **self.standardizer.state(),
            **self.reducer.state(),
        }
        np.savez_compressed(paths["state"], **state)
        metadata = dict(self.fit_metadata_)
        metadata["artifacts"] = {
            "state": paths["state"].name,
            "metadata": paths["metadata"].name,
        }
        paths["metadata"].write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return paths

    @classmethod
    def load_state(
        cls,
        state_path: Path,
        metadata_path: Path,
    ) -> "HSIPreprocessingPipeline":
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        config = PreprocessingConfig(**metadata["config"])
        config.validate()
        with np.load(state_path, allow_pickle=False) as saved:
            artifact = {name: saved[name].copy() for name in saved.files}
        if str(_scalar(artifact["schema_version"])) != PREPROCESSING_SCHEMA_VERSION:
            raise ValueError("unsupported preprocessing state schema")
        if str(_scalar(artifact["config_fingerprint"])) != config.fingerprint():
            raise ValueError("preprocessing state and metadata config fingerprints differ")

        instance = cls(config)
        instance.standardizer = BandStandardizer.from_state(artifact)
        reducer_name = str(_scalar(artifact["reducer_name"]))
        if reducer_name == "none":
            instance.reducer = IdentityReducer.from_state(artifact)
        elif reducer_name == "pca":
            instance.reducer = PCASpectralReducer.from_state(artifact)
        elif reducer_name == "lda":
            instance.reducer = LDASpectralReducer.from_state(artifact)
        else:
            raise ValueError(f"unsupported reducer in saved state: {reducer_name!r}")
        instance.fit_metadata_ = metadata
        return instance

    def attach_transformed_cube(self, cube: np.ndarray) -> np.ndarray:
        """Transform and attach a cube after loading a saved state."""
        self.transformed_cube_ = self.transform_cube(cube)
        return self.transformed_cube_

    def _require_transformed_cube(self) -> None:
        if self.transformed_cube_ is None:
            raise RuntimeError("fit the pipeline or attach a transformed cube first")

    def _build_fit_metadata(self, data: HSIDataBundle) -> dict[str, Any]:
        reducer_details: dict[str, Any] = {
            "name": self.config.reducer,
            "input_bands": int(data.cube.shape[2]),
            "output_bands": self.output_bands,
        }
        if isinstance(self.reducer, PCASpectralReducer):
            reducer_details.update(
                {
                    "whiten": self.reducer.whiten,
                    "explained_variance_ratio": self.reducer.explained_variance_ratio_.tolist(),
                    "cumulative_explained_variance_ratio": float(
                        self.reducer.explained_variance_ratio_.sum()
                    ),
                }
            )
        elif isinstance(self.reducer, LDASpectralReducer):
            reducer_details.update(
                {
                    "solver": self.reducer.solver,
                    "supervised": True,
                    "classes_seen": self.reducer.classes_.tolist(),
                    "explained_variance_ratio": self.reducer.explained_variance_ratio_.tolist(),
                    "cumulative_explained_variance_ratio": float(
                        self.reducer.explained_variance_ratio_.sum()
                    ),
                }
            )

        source_files = {}
        for path in (
            data.split_path.parents[1] / "raw" / data.spec.data_file,
            data.split_path.parents[1] / "raw" / data.spec.label_file,
        ):
            source_files[path.name] = {"sha256": _sha256_file(path)}
        return {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "config": asdict(self.config),
            "config_fingerprint": self.config.fingerprint(),
            "dataset": {
                "name": data.spec.name,
                "cube_shape": list(data.cube.shape),
                "label_shape": list(data.label_map.shape),
                "class_names": list(data.spec.class_names),
                "source_files": source_files,
            },
            "split": {
                "protocol": self.config.split_protocol,
                "seed": self.config.split_seed,
                "split_file": data.split_path.name,
                "split_file_sha256": _sha256_file(data.split_path),
                "split_metadata_file": data.split_metadata_path.name,
                "sample_counts": {
                    name: int(indices.size) for name, indices in data.indices_by_split.items()
                },
            },
            "fit_scope": {
                "samples": "training center pixels only",
                "training_samples": int(data.train_indices.size),
                "validation_and_test_used_for_fit": False,
            },
            "standardization": {
                "name": self.config.standardization,
                "per_band": True,
                "samples_seen": self.standardizer.n_samples_seen_,
            },
            "spectral_reducer": reducer_details,
            "spatial_representation": {
                "name": self.config.representation,
                "patch_size": self.config.patch_size,
                "padding_mode": self.config.padding_mode,
                "padding_value": self.config.padding_value,
                "patches_materialized_on_disk": False,
            },
            "label_contract": {
                "stored_raw_labels": "1..C",
                "model_labels": "0..C-1",
                "background_label": 0,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        }
