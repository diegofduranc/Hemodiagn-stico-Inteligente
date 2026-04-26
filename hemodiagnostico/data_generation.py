import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from hemodiagnostico.config import DataGenerationConfig


FEATURE_COLUMNS = [
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
]


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


def _normalize_differential(data: pd.DataFrame) -> None:
    diff_cols = ["Neutrophils_pct", "Lymphocytes_pct", "Eosinophils_pct", "Monocytes_pct"]
    data[diff_cols] = data[diff_cols].clip(lower=0.0, upper=95.0)
    diff_sum = data[diff_cols].sum(axis=1).replace(0.0, 1.0)
    data[diff_cols] = data[diff_cols].div(diff_sum, axis=0) * 100.0


def _apply_borderline_overlap(data: pd.DataFrame, rng: np.random.Generator) -> None:
    overlap_rules = [
        (0, 9, 0.12),
        (9, 0, 0.10),
        (1, 2, 0.16),
        (2, 1, 0.16),
        (3, 4, 0.18),
        (4, 3, 0.18),
    ]

    for src_id, donor_id, frac in overlap_rules:
        src_idx = data.index[data["condition_id"] == src_id].to_numpy()
        donor_idx = data.index[data["condition_id"] == donor_id].to_numpy()
        if len(src_idx) == 0 or len(donor_idx) == 0:
            continue

        n_mix = max(1, int(len(src_idx) * frac))
        chosen_src = rng.choice(src_idx, size=n_mix, replace=False)
        chosen_donor = rng.choice(donor_idx, size=n_mix, replace=True)

        alpha = rng.uniform(0.55, 0.82, size=n_mix)
        src_values = data.loc[chosen_src, FEATURE_COLUMNS].to_numpy()
        donor_values = data.loc[chosen_donor, FEATURE_COLUMNS].to_numpy()
        blended = src_values * alpha[:, None] + donor_values * (1.0 - alpha)[:, None]
        data.loc[chosen_src, FEATURE_COLUMNS] = blended


def _inject_leukemia_outliers(data: pd.DataFrame, rng: np.random.Generator) -> None:
    leukemia_idx = data.index[data["condition_id"] == 7].to_numpy()
    if len(leukemia_idx) == 0:
        return

    n_outliers = max(1, int(0.14 * len(leukemia_idx)))
    chosen = rng.choice(leukemia_idx, size=n_outliers, replace=False)

    data.loc[chosen, "WBC_K_uL"] = _clip(
        data.loc[chosen, "WBC_K_uL"].to_numpy() * rng.uniform(1.5, 2.4, size=n_outliers),
        1.0,
        250.0,
    )
    data.loc[chosen, "Blasts_pct"] = _clip(
        data.loc[chosen, "Blasts_pct"].to_numpy() + rng.normal(12.0, 5.0, size=n_outliers),
        0.0,
        85.0,
    )
    data.loc[chosen, "Platelets_K_uL"] = _clip(
        data.loc[chosen, "Platelets_K_uL"].to_numpy() * rng.uniform(0.45, 0.9, size=n_outliers),
        5.0,
        900.0,
    )


def _apply_post_constraints(data: pd.DataFrame, rng: np.random.Generator) -> None:
    data["RBC_Mil_uL"] = _clip(data["RBC_Mil_uL"].to_numpy(), 2.0, 8.0)
    data["MCV_fL"] = _clip(data["MCV_fL"].to_numpy(), 50.0, 130.0)
    data["RDW_pct"] = _clip(data["RDW_pct"].to_numpy(), 10.0, 30.0)
    data["WBC_K_uL"] = _clip(data["WBC_K_uL"].to_numpy(), 1.0, 250.0)
    data["Platelets_K_uL"] = _clip(data["Platelets_K_uL"].to_numpy(), 5.0, 900.0)
    data["Blasts_pct"] = _clip(data["Blasts_pct"].to_numpy(), 0.0, 85.0)

    data["HCT_pct"] = (data["RBC_Mil_uL"] * data["MCV_fL"] / 10.0) + rng.normal(0.0, 1.6, len(data))
    data["HCT_pct"] = _clip(data["HCT_pct"].to_numpy(), 18.0, 70.0)
    data["HGB_g_dL"] = (data["HCT_pct"] / 2.95) + rng.normal(0.0, 0.6, len(data))
    data["HGB_g_dL"] = _clip(data["HGB_g_dL"].to_numpy(), 5.0, 22.0)

    _normalize_differential(data)


def build_profiles() -> List[ClassProfile]:
    return [
        ClassProfile(0, "Sano", 15000, 4.95, 0.45, 90, 5.8, 7.4, 2.0, 265, 70, 55, 8, 33, 7, 2.8, 1.5, 0.1, 0.2, 13.2, 1.1),
        ClassProfile(1, "Anemia Ferropenica", 6000, 3.65, 0.45, 74, 8.0, 7.4, 2.0, 275, 72, 56, 8, 31, 7, 2.9, 1.5, 0.1, 0.2, 17.2, 2.0),
        ClassProfile(2, "Anemia Megaloblastica", 4000, 3.35, 0.42, 103, 10.0, 7.0, 2.0, 245, 65, 51, 8, 37, 8, 2.7, 1.3, 0.2, 0.3, 16.4, 1.9),
        ClassProfile(3, "Infeccion Bacteriana", 5000, 4.7, 0.45, 89, 5.8, 13.2, 4.8, 285, 75, 72, 10, 20, 8, 2.2, 1.1, 0.3, 0.4, 13.9, 1.2),
        ClassProfile(4, "Infeccion Viral", 5000, 4.6, 0.4, 90, 5.6, 9.8, 3.8, 255, 70, 40, 10, 45, 10, 3.1, 1.6, 0.3, 0.4, 13.7, 1.1),
        ClassProfile(5, "Trombocitopenia", 2500, 4.5, 0.35, 91, 4.8, 7.3, 1.8, 55, 22, 55, 6, 33, 6, 2.6, 1.2, 0.2, 0.2, 13.7, 1.0),
        ClassProfile(6, "Policitemia Vera", 2500, 6.4, 0.6, 90, 5.2, 9.4, 2.6, 405, 95, 59, 8, 29, 7, 2.4, 1.2, 0.3, 0.3, 13.5, 1.0, hct_multiplier=1.08, hgb_multiplier=1.08),
        ClassProfile(7, "Leucemia", 1200, 3.9, 0.6, 94, 11.0, 78.0, 42.0, 125, 100, 45, 20, 29, 16, 4.6, 3.8, 20.0, 14.0, 15.6, 2.4),
        ClassProfile(8, "Alergias", 5000, 4.8, 0.38, 89, 5.3, 8.9, 2.4, 268, 65, 47, 8, 31, 8, 15.2, 5.0, 0.1, 0.2, 13.8, 1.1),
        ClassProfile(9, "Deshidratacion", 3800, 5.2, 0.48, 90, 5.4, 8.0, 2.2, 302, 78, 56, 8, 32, 7, 2.6, 1.3, 0.1, 0.2, 13.6, 1.0, hct_multiplier=1.12, hgb_multiplier=1.12),
    ]


def generate_dataset(config: DataGenerationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    profiles = build_profiles()

    all_rows = [_generate_profile_rows(profile, rng) for profile in profiles]
    data = pd.concat(all_rows, ignore_index=True)

    _apply_borderline_overlap(data, rng)
    _inject_leukemia_outliers(data, rng)
    _apply_post_constraints(data, rng)

    data = data.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    data.to_csv(config.output_csv, index=False)
    return data


def run_from_args(args: argparse.Namespace) -> None:
    config = DataGenerationConfig(output_csv=args.output, random_seed=args.random_seed)
    data = generate_dataset(config)

    print(f"Dataset generated: {config.output_csv}")
    print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
    print("Class distribution:")
    print(data["condition_name"].value_counts())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic CBC dataset for multiclass health classification.")
    parser.add_argument("--output", default="hemograma_data.csv", help="Output CSV path")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for reproducibility")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
