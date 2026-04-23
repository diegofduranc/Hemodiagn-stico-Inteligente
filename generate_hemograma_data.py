import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


RANDOM_SEED = 42


@dataclass
class ClassProfile:
    condition_id: int
    condition_name: str
    n_samples: int
    rbc_mean: float
    rbc_std: float
    mcv_mean: float
    mcv_std: float
    wbc_mean: float
    wbc_std: float
    platelets_mean: float
    platelets_std: float
    neut_mean: float
    neut_std: float
    lymph_mean: float
    lymph_std: float
    eos_mean: float
    eos_std: float
    blasts_mean: float
    blasts_std: float
    rdw_mean: float
    rdw_std: float
    hct_multiplier: float = 1.0
    hgb_multiplier: float = 1.0


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _generate_profile_rows(profile: ClassProfile, rng: np.random.Generator) -> pd.DataFrame:
    n = profile.n_samples

    rbc = _clip(rng.normal(profile.rbc_mean, profile.rbc_std, n), 2.0, 8.0)
    mcv = _clip(rng.normal(profile.mcv_mean, profile.mcv_std, n), 50.0, 130.0)

    # HCT and HGB are derived from RBC and MCV to preserve physiological consistency.
    hct = (rbc * mcv / 10.0) * profile.hct_multiplier + rng.normal(0.0, 1.2, n)
    hct = _clip(hct, 18.0, 70.0)

    hgb = (hct / 2.95) * profile.hgb_multiplier + rng.normal(0.0, 0.5, n)
    hgb = _clip(hgb, 5.0, 22.0)

    wbc = _clip(rng.normal(profile.wbc_mean, profile.wbc_std, n), 1.0, 250.0)
    platelets = _clip(rng.normal(profile.platelets_mean, profile.platelets_std, n), 5.0, 900.0)
    rdw = _clip(rng.normal(profile.rdw_mean, profile.rdw_std, n), 10.0, 30.0)

    neut = _clip(rng.normal(profile.neut_mean, profile.neut_std, n), 1.0, 95.0)
    lymph = _clip(rng.normal(profile.lymph_mean, profile.lymph_std, n), 1.0, 95.0)
    eos = _clip(rng.normal(profile.eos_mean, profile.eos_std, n), 0.0, 40.0)

    # Normalize differential to make percentages coherent with each other.
    mono = _clip(rng.normal(6.0, 2.0, n), 1.0, 18.0)
    diff_sum = neut + lymph + eos + mono
    neut = (neut / diff_sum) * 100.0
    lymph = (lymph / diff_sum) * 100.0
    eos = (eos / diff_sum) * 100.0
    mono = (mono / diff_sum) * 100.0

    blasts = _clip(rng.normal(profile.blasts_mean, profile.blasts_std, n), 0.0, 85.0)

    return pd.DataFrame(
        {
            "RBC_Mil_uL": rbc,
            "HGB_g_dL": hgb,
            "HCT_pct": hct,
            "MCV_fL": mcv,
            "RDW_pct": rdw,
            "WBC_K_uL": wbc,
            "Neutrophils_pct": neut,
            "Lymphocytes_pct": lymph,
            "Eosinophils_pct": eos,
            "Monocytes_pct": mono,
            "Platelets_K_uL": platelets,
            "Blasts_pct": blasts,
            "condition_id": profile.condition_id,
            "condition_name": profile.condition_name,
        }
    )


def build_profiles() -> List[ClassProfile]:
    return [
        ClassProfile(0, "Sano", 15000, 4.9, 0.35, 90, 4.5, 7.0, 1.5, 260, 55, 56, 6, 32, 5, 2.5, 1.2, 0.1, 0.2, 13.0, 0.8),
        ClassProfile(1, "Anemia Ferropenica", 6000, 3.6, 0.4, 70, 5.0, 7.2, 1.8, 280, 65, 58, 7, 30, 5, 2.8, 1.4, 0.1, 0.2, 17.5, 1.8),
        ClassProfile(2, "Anemia Megaloblastica", 4000, 3.2, 0.35, 108, 6.0, 6.8, 1.7, 240, 55, 52, 7, 36, 6, 2.5, 1.1, 0.2, 0.3, 16.8, 1.6),
        ClassProfile(3, "Infeccion Bacteriana", 5000, 4.7, 0.4, 89, 5.0, 15.0, 4.5, 290, 70, 78, 7, 15, 5, 2.0, 1.0, 0.2, 0.3, 13.8, 1.1),
        ClassProfile(4, "Infeccion Viral", 5000, 4.6, 0.35, 90, 4.5, 10.5, 3.6, 250, 60, 34, 7, 52, 8, 3.0, 1.5, 0.2, 0.3, 13.6, 1.0),
        ClassProfile(5, "Trombocitopenia", 2500, 4.5, 0.35, 91, 4.8, 7.3, 1.8, 55, 22, 55, 6, 33, 6, 2.6, 1.2, 0.2, 0.2, 13.7, 1.0),
        ClassProfile(6, "Policitemia Vera", 2500, 6.5, 0.5, 90, 4.5, 9.2, 2.1, 420, 85, 60, 6, 28, 5, 2.2, 1.0, 0.3, 0.3, 13.5, 0.9, hct_multiplier=1.08, hgb_multiplier=1.08),
        ClassProfile(7, "Leucemia", 1200, 3.8, 0.55, 94, 9.0, 85.0, 38.0, 130, 95, 45, 18, 28, 15, 4.5, 3.5, 22.0, 12.0, 15.4, 2.2),
        ClassProfile(8, "Alergias", 5000, 4.8, 0.35, 89, 4.8, 8.8, 2.0, 270, 60, 48, 7, 30, 6, 16.5, 4.5, 0.1, 0.2, 13.8, 1.0),
        ClassProfile(9, "Deshidratacion", 3800, 5.1, 0.4, 90, 4.6, 7.8, 1.8, 300, 70, 57, 7, 31, 5, 2.5, 1.2, 0.1, 0.2, 13.5, 0.9, hct_multiplier=1.12, hgb_multiplier=1.12),
    ]


def generate_dataset(output_csv: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    profiles = build_profiles()

    all_rows = [_generate_profile_rows(profile, rng) for profile in profiles]
    data = pd.concat(all_rows, ignore_index=True)
    data = data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    data.to_csv(output_csv, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CBC dataset for multiclass health classification.")
    parser.add_argument("--output", default="hemograma_data.csv", help="Output CSV path")
    args = parser.parse_args()

    data = generate_dataset(args.output)

    print(f"Dataset generated: {args.output}")
    print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
    print("Class distribution:")
    print(data["condition_name"].value_counts())


if __name__ == "__main__":
    main()
