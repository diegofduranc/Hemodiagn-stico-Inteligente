from pathlib import Path

from hemodiagnostico.config import DataGenerationConfig, TrainingConfig
from hemodiagnostico.data_generation import generate_dataset
import hemodiagnostico.model_training as model_training


def test_training_smoke_without_xgboost(tmp_path: Path, monkeypatch) -> None:
    data_csv = tmp_path / "smoke_data.csv"
    full_df = generate_dataset(DataGenerationConfig(output_csv=str(data_csv), random_seed=42))

    # Use a smaller balanced subset to keep smoke test runtime practical.
    sampled = full_df.groupby("condition_id", group_keys=False).sample(
        n=150,
        random_state=42,
    )
    sampled.to_csv(data_csv, index=False)

    monkeypatch.setattr(model_training, "XGBOOST_AVAILABLE", False)

    output_dir = tmp_path / "smoke_outputs"
    cfg = TrainingConfig(
        data_path=str(data_csv),
        output_dir=str(output_dir),
        tune=False,
        cv_folds=0,
        random_seed=42,
    )

    model_training.run_training(cfg)

    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "random_forest" / "classification_report.txt").exists()
    assert (output_dir / "random_forest_esp32" / "classification_report.txt").exists()
    assert (output_dir / "esp32_export" / "esp32_preprocess.h").exists()
    assert (output_dir / "esp32_export" / "esp32_rf_model.h").exists()
