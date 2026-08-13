from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.检查配置 import PROJECT_ROOT, inspect_config
from src.datasets.高光谱预处理 import PreprocessingConfig


UNIFIED_CONFIG = Path("configs/数据预处理/Pavia数据预处理.yaml")


@pytest.mark.parametrize(
    ("relative_path", "route", "state_exists"),
    [
        (
            str(UNIFIED_CONFIG),
            "fair24_6_70__seed345__standard_pca15_patch25",
            True,
        ),
        (
            "configs/模型训练/HybridSN_Pavia论文复现基线.yaml",
            "paper30__seed345__standard_pca15_patch25",
            True,
        ),
        (
            "configs/模型训练/HybridSN_Pavia公平调参模板.yaml",
            "fair24_6_70__seed345__standard_pca15_patch25",
            True,
        ),
    ],
)
def test_all_public_yaml_configs(
    relative_path: str,
    route: str,
    state_exists: bool,
) -> None:
    result = inspect_config(Path(relative_path))

    assert result["path"] == (PROJECT_ROOT / relative_path).resolve()
    assert result["route"] == route
    assert result["state_exists"] is state_exists


@pytest.mark.parametrize(
    ("protocol", "reducer", "representation", "route", "state_exists"),
    [
        (
            "fair24_6_70",
            "pca",
            "patch",
            "fair24_6_70__seed345__standard_pca15_patch25",
            True,
        ),
        (
            "fair24_6_70",
            "lda",
            "patch",
            "fair24_6_70__seed345__standard_lda8_patch25",
            True,
        ),
        (
            "fair24_6_70",
            "pca",
            "pixel",
            "fair24_6_70__seed345__standard_pca15_pixel",
            False,
        ),
        (
            "fair24_6_70",
            "lda",
            "pixel",
            "fair24_6_70__seed345__standard_lda8_pixel",
            True,
        ),
        (
            "fair24_6_70",
            "none",
            "pixel",
            "fair24_6_70__seed345__standard_none_pixel",
            False,
        ),
        (
            "paper30",
            "pca",
            "patch",
            "paper30__seed345__standard_pca15_patch25",
            True,
        ),
        (
            "paper30",
            "lda",
            "patch",
            "paper30__seed345__standard_lda8_patch25",
            True,
        ),
    ],
)
def test_unified_yaml_selectors_reproduce_historical_routes(
    protocol: str,
    reducer: str,
    representation: str,
    route: str,
    state_exists: bool,
) -> None:
    values = yaml.safe_load((PROJECT_ROOT / UNIFIED_CONFIG).read_text(encoding="utf-8"))
    selected = copy.deepcopy(values)
    selected["dataset"]["split_protocol"] = protocol
    selected["spectral_preprocessing"]["reducer"] = reducer
    selected["spatial_preprocessing"]["representation"] = representation
    config = PreprocessingConfig.from_mapping(selected)
    state_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / config.dataset_name
        / config.route_name()
    )

    assert config.route_name() == route
    assert all(
        (state_dir / name).is_file()
        for name in ("preprocessing_state.npz", "metadata.json")
    ) is state_exists


def test_seven_replaced_yaml_presets_are_preserved_in_archive() -> None:
    archive = PROJECT_ROOT / "归档" / "阶段2历史配置" / "数据预处理多文件预设"
    yaml_names = {path.name for path in archive.glob("*.yaml")}

    assert yaml_names == {
        "Pavia公平比较_LDA8_单像元.yaml",
        "Pavia公平比较_LDA8_邻域25.yaml",
        "Pavia公平比较_PCA15_单像元.yaml",
        "Pavia公平比较_PCA15_邻域25.yaml",
        "Pavia公平比较_原始103波段_单像元.yaml",
        "Pavia论文复现_LDA8_邻域25.yaml",
        "Pavia论文复现_PCA15_邻域25.yaml",
    }
