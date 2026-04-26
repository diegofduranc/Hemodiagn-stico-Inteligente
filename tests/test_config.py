from pathlib import Path

from hemodiagnostico.config import load_project_config


def test_load_project_config_defaults_when_missing(tmp_path: Path) -> None:
    missing_cfg = tmp_path / "missing.json"
    cfg = load_project_config(str(missing_cfg))

    assert cfg.data_generation.output_csv == "hemograma_data.csv"
    assert cfg.training.output_dir == "outputs"
    assert cfg.training.cv_folds == 5


def test_load_project_config_override_values(tmp_path: Path) -> None:
    cfg_file = tmp_path / "project_config.json"
    cfg_file.write_text(
        """
{
  "data_generation": {
    "output_csv": "custom_data.csv",
    "random_seed": 777
  },
  "training": {
    "data_path": "custom_data.csv",
    "output_dir": "custom_outputs",
    "tune": true,
    "cv_folds": 3,
    "random_seed": 777
  }
}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_project_config(str(cfg_file))

    assert cfg.data_generation.output_csv == "custom_data.csv"
    assert cfg.data_generation.random_seed == 777
    assert cfg.training.data_path == "custom_data.csv"
    assert cfg.training.output_dir == "custom_outputs"
    assert cfg.training.tune is True
    assert cfg.training.cv_folds == 3
    assert cfg.training.random_seed == 777
