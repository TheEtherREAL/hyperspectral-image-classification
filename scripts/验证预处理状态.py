"""验证冻结预处理状态 / Verify state against raw data and frozen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.datasets.高光谱预处理 import (
    HSIPreprocessingPipeline,
    LDASpectralReducer,
    PCASpectralReducer,
    PreprocessingConfig,
    load_hsi_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = PreprocessingConfig.from_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    state_dir = args.state_dir
    if state_dir is None:
        state_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / config.dataset_name
            / config.route_name()
        )
    elif not state_dir.is_absolute():
        state_dir = PROJECT_ROOT / state_dir

    state_path = state_dir / "preprocessing_state.npz"
    metadata_path = state_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["config_fingerprint"] != config.fingerprint():
        raise AssertionError("saved metadata does not match the selected config")

    data = load_hsi_data(PROJECT_ROOT, config)
    for file_name, record in metadata["dataset"]["source_files"].items():
        if sha256(PROJECT_ROOT / "data" / "raw" / file_name) != record["sha256"]:
            raise AssertionError(f"raw source hash mismatch: {file_name}")
    if sha256(data.split_path) != metadata["split"]["split_file_sha256"]:
        raise AssertionError("fixed split hash mismatch")
    if metadata["fit_scope"]["training_samples"] != data.train_indices.size:
        raise AssertionError("saved fit sample count does not match the fixed split")
    if metadata["fit_scope"]["validation_and_test_used_for_fit"] is not False:
        raise AssertionError("metadata does not assert train-only fitting")

    loaded = HSIPreprocessingPipeline.load_state(state_path, metadata_path)
    loaded.attach_transformed_cube(data.cube)
    rebuilt = HSIPreprocessingPipeline(config).fit(data)
    np.testing.assert_allclose(loaded.standardizer.mean_, rebuilt.standardizer.mean_)
    np.testing.assert_allclose(loaded.standardizer.scale_, rebuilt.standardizer.scale_)
    if isinstance(loaded.reducer, PCASpectralReducer):
        np.testing.assert_allclose(loaded.reducer.components_, rebuilt.reducer.components_)
        np.testing.assert_allclose(
            loaded.reducer.explained_variance_ratio_,
            rebuilt.reducer.explained_variance_ratio_,
        )
    elif isinstance(loaded.reducer, LDASpectralReducer):
        np.testing.assert_allclose(loaded.reducer.xbar_, rebuilt.reducer.xbar_)
        np.testing.assert_allclose(loaded.reducer.scalings_, rebuilt.reducer.scalings_)
        np.testing.assert_array_equal(loaded.reducer.classes_, rebuilt.reducer.classes_)
        np.testing.assert_allclose(
            loaded.reducer.explained_variance_ratio_,
            rebuilt.reducer.explained_variance_ratio_,
        )
    np.testing.assert_allclose(loaded.transformed_cube_, rebuilt.transformed_cube_)

    datasets = loaded.build_torch_datasets(data)
    sample = datasets["train"][0]
    if sample["raw_label"].item() != sample["label"].item() + 1:
        raise AssertionError("raw/model label mapping is invalid")
    print(f"route={config.route_name()}")
    print(f"train_samples={data.train_indices.size}")
    print(f"transformed_cube_shape={loaded.transformed_cube_.shape}")
    print(f"first_input_shape={tuple(sample['input'].shape)}")
    if isinstance(loaded.reducer, (PCASpectralReducer, LDASpectralReducer)):
        print(
            "cumulative_explained_variance_ratio="
            f"{loaded.reducer.explained_variance_ratio_.sum():.10f}"
        )
    print("source_hashes=passed")
    print("saved_state_rebuild=passed")
    print("preprocessing_verification=passed")


if __name__ == "__main__":
    main()
