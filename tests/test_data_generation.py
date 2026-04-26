from pathlib import Path

import pandas as pd

from hemodiagnostico.config import DataGenerationConfig
from hemodiagnostico.data_generation import generate_dataset


def test_generate_dataset_shape_and_classes(tmp_path: Path) -> None:
    out_csv = tmp_path / "hemograma_test.csv"
    cfg = DataGenerationConfig(output_csv=str(out_csv), random_seed=42)

    df = generate_dataset(cfg)

    assert out_csv.exists()
    assert len(df) == 50000
    assert df["condition_name"].nunique() == 10

    expected_cols = {
        "RBC_Mil_uL",
        "HGB_g_dL",
        "HCT_pct",
        "MCV_fL",
        "RDW_pct",
        "WBC_K_uL",
        "Neutrophils_pct",
        "Lymphocytes_pct",
        "Eosinophils_pct",
        "Monocytes_pct",
        "Platelets_K_uL",
        "Blasts_pct",
        "condition_id",
        "condition_name",
    }
    assert set(df.columns) == expected_cols

    loaded = pd.read_csv(out_csv)
    assert len(loaded) == 50000
