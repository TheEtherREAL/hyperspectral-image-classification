"""高光谱数据集元数据注册 / HSI dataset registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_file: str
    label_file: str
    data_key: str
    label_key: str
    class_names: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "pavia_university": DatasetSpec(
        name="pavia_university",
        data_file="PaviaU.mat",
        label_file="PaviaU_gt.mat",
        data_key="paviaU",
        label_key="paviaU_gt",
        class_names=(
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ),
    ),
    "indian_pines": DatasetSpec(
        name="indian_pines",
        data_file="Indian_pines_corrected.mat",
        label_file="Indian_pines_gt.mat",
        data_key="indian_pines_corrected",
        label_key="indian_pines_gt",
        class_names=(
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ),
    ),
    "salinas": DatasetSpec(
        name="salinas",
        data_file="Salinas_corrected.mat",
        label_file="Salinas_gt.mat",
        data_key="salinas_corrected",
        label_key="salinas_gt",
        class_names=(
            "Broccoli_green_weeds_1",
            "Broccoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ),
    ),
}
