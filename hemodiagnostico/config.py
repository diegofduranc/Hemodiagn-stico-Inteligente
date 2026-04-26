import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class DataGenerationConfig:
    output_csv: str = "hemograma_data.csv"
    random_seed: int = 42


@dataclass
class TrainingConfig:
    data_path: str = "hemograma_data.csv"
    output_dir: str = "outputs"
    tune: bool = False
    cv_folds: int = 5
    random_seed: int = 42


@dataclass
class ProjectConfig:
    data_generation: DataGenerationConfig
    training: TrainingConfig


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(config_path: str = "configs/project_config.json") -> ProjectConfig:
    default_cfg = {
        "data_generation": {
            "output_csv": "hemograma_data.csv",
            "random_seed": 42,
        },
        "training": {
            "data_path": "hemograma_data.csv",
            "output_dir": "outputs",
            "tune": False,
            "cv_folds": 5,
            "random_seed": 42,
        },
    }

    cfg_path = Path(config_path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        merged = _merge_dict(default_cfg, user_cfg)
    else:
        merged = default_cfg

    return ProjectConfig(
        data_generation=DataGenerationConfig(**merged["data_generation"]),
        training=TrainingConfig(**merged["training"]),
    )
