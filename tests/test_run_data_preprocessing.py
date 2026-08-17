from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.运行数据预处理 import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    inspect_first_training_batch,
    load_fixed_dataset,
    load_workflow_settings,
    run_readonly_workflow,
    verify_frozen_split_relationships,
)


def test_default_settings_match_the_frozen_fair_route() -> None:
    settings = load_workflow_settings(DEFAULT_CONFIG, announce=False)

    assert settings.project_root == PROJECT_ROOT
    assert settings.config.split_protocol == "fair24_6_70"
    assert settings.config.split_seed == 1442
    assert settings.config.route_name() == (
        "fair24_6_70__seed1442__standard_pca15_patch25"
    )
    assert (settings.state_dir / "preprocessing_state.npz").is_file()
    assert (settings.state_dir / "metadata.json").is_file()


def test_batch_inspection_accepts_the_public_patch_contract() -> None:
    settings = load_workflow_settings(
        DEFAULT_CONFIG,
        batch_size=2,
        pin_memory=False,
        announce=False,
    )
    batch = {
        "input": torch.zeros((2, 1, 15, 25, 25), dtype=torch.float32),
        "label": torch.tensor([0, 8], dtype=torch.int64),
        "raw_label": torch.tensor([1, 9], dtype=torch.int64),
        "coordinate": torch.tensor([[0, 0], [1, 1]], dtype=torch.int64),
        "sample_index": torch.tensor([0, 1], dtype=torch.int64),
    }

    inspected = inspect_first_training_batch(
        settings,
        {"train": [batch]},
        announce=False,
    )

    assert inspected is batch


def test_readonly_workflow_rejects_an_unknown_step_before_reading_data() -> None:
    with pytest.raises(ValueError, match="between 1 and 7"):
        run_readonly_workflow(stop_after=8)


def test_default_config_path_stays_inside_the_project() -> None:
    settings = load_workflow_settings(DEFAULT_CONFIG, announce=False)
    assert settings.config_path == (
        PROJECT_ROOT / Path(DEFAULT_CONFIG)
    ).resolve()


def test_frozen_pavia_protocol_relationships_are_preserved() -> None:
    settings = load_workflow_settings(DEFAULT_CONFIG, announce=False)
    data = load_fixed_dataset(settings, announce=False)

    checks = verify_frozen_split_relationships(settings, data, announce=False)

    assert checks["current_disjoint"] is True
    assert checks["current_full_coverage"] is True
    assert checks["current_class_coverage"] is True
    assert checks["shared_test_set"] is True
    assert checks["fair_train_plus_validation_equals_paper_train"] is True
