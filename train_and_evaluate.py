import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, label_binarize


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


RANDOM_SEED = 42
CRITICAL_CLASS_IDS = [5, 7]  # Trombocitopenia, Leucemia


def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[int, str]]:
    feature_cols = [
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

    missing = [c for c in feature_cols + ["condition_id", "condition_name"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    class_map = (
        df[["condition_id", "condition_name"]]
        .drop_duplicates()
        .sort_values("condition_id")
        .set_index("condition_id")["condition_name"]
        .to_dict()
    )

    X = df[feature_cols].copy()
    y = df["condition_id"].astype(int).copy()
    return X, y, class_map


def compute_class_weights(y_train: pd.Series) -> Dict[int, float]:
    counts = y_train.value_counts().to_dict()
    total = len(y_train)
    n_classes = len(counts)

    class_weights = {
        cls: total / (n_classes * count)
        for cls, count in counts.items()
    }

    # Increase penalty for missing critical diseases.
    for critical_cls in CRITICAL_CLASS_IDS:
        if critical_cls in class_weights:
            class_weights[critical_cls] *= 2.2

    return class_weights


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_map: Dict[int, str],
    output_dir: Path,
    sample_weight: np.ndarray = None,
) -> Dict[str, float]:
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    labels = sorted(class_map.keys())
    target_names = [class_map[i] for i in labels]

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    y_test_bin = label_binarize(y_test, classes=labels)
    auc_roc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

    model_dir = output_dir / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write(f"\nF1 Macro: {f1_macro:.4f}\n")
        f.write(f"AUC-ROC Macro (OvR): {auc_roc_macro:.4f}\n")

    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        model_dir / "confusion_matrix.csv", index=True
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    critical_recalls = {}
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    for cls_id in CRITICAL_CLASS_IDS:
        cls_name = class_map[cls_id]
        critical_recalls[cls_name] = report_dict[cls_name]["recall"]

    return {
        "f1_macro": f1_macro,
        "auc_roc_macro": auc_roc_macro,
        **{f"recall_{k}": v for k, v in critical_recalls.items()},
    }


def export_rf_for_esp32(rf_pipeline: Pipeline, feature_names: List[str], output_dir: Path) -> None:
    export_dir = output_dir / "esp32_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_pipeline, export_dir / "rf_pipeline.joblib")

    try:
        from micromlgen import port

        model = rf_pipeline.named_steps["clf"]
        model_code = port(model)
        with open(export_dir / "esp32_rf_model.h", "w", encoding="utf-8") as f:
            f.write("// Auto-generated model for embedded inference on ESP32\n")
            f.write("// Feature order must match exactly:\n")
            for idx, name in enumerate(feature_names):
                f.write(f"// {idx}: {name}\n")
            f.write("\n")
            f.write(model_code)
    except ImportError:
        print("micromlgen is not installed. Skipping C header export for ESP32.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate multiclass CBC classifiers.")
    parser.add_argument("--data", default="hemograma_data.csv", help="Path to dataset CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store metrics and plots")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y, class_map = build_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    class_weights = compute_class_weights(y_train)
    train_weights = y_train.map(class_weights).to_numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rf_pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight=class_weights,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\n=== Training Random Forest ===")
    rf_metrics = evaluate_model(
        model_name="Random Forest",
        model=rf_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_map=class_map,
        output_dir=output_dir,
    )

    print("Random Forest metrics:")
    for k, v in rf_metrics.items():
        print(f"  {k}: {v:.4f}")

    all_metrics = {"Random Forest": rf_metrics}

    if XGBOOST_AVAILABLE:
        xgb_pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(class_map),
                        n_estimators=220,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="mlogloss",
                        random_state=RANDOM_SEED,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        print("\n=== Training XGBoost ===")
        xgb_metrics = evaluate_model(
            model_name="XGBoost",
            model=xgb_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_map=class_map,
            output_dir=output_dir,
            sample_weight=train_weights,
        )

        print("XGBoost metrics:")
        for k, v in xgb_metrics.items():
            print(f"  {k}: {v:.4f}")

        all_metrics["XGBoost"] = xgb_metrics
    else:
        print("\nXGBoost is not installed. Install it with: pip install xgboost")

    feature_names = X.columns.tolist()
    export_rf_for_esp32(rf_pipeline, feature_names, output_dir)

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=True)

    print("\nSaved artifacts in:", output_dir)
    print(metrics_df)


if __name__ == "__main__":
    main()
