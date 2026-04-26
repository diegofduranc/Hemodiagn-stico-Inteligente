import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable

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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, label_binarize

from hemodiagnostico.config import TrainingConfig


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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
    class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}

    for critical_cls in CRITICAL_CLASS_IDS:
        if critical_cls in class_weights:
            class_weights[critical_cls] *= 2.2

    return class_weights


def _build_selection_metrics(y_true: pd.Series, y_pred: np.ndarray, class_map: Dict[int, str]) -> Dict[str, float]:
    labels = sorted(class_map.keys())
    target_names = [class_map[i] for i in labels]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    critical_recalls = {}
    for cls_id in CRITICAL_CLASS_IDS:
        cls_name = class_map[cls_id]
        critical_recalls[f"recall_{cls_name}"] = report_dict[cls_name]["recall"]

    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mean_recall_critical": float(np.mean(list(critical_recalls.values()))),
        "min_recall_critical": float(np.min(list(critical_recalls.values()))),
        **critical_recalls,
    }


def _is_better_candidate(curr: Dict[str, float], best: Dict[str, float]) -> bool:
    if best is None:
        return True
    curr_key = (curr["min_recall_critical"], curr["mean_recall_critical"], curr["f1_macro"])
    best_key = (best["min_recall_critical"], best["mean_recall_critical"], best["f1_macro"])
    return curr_key > best_key


def tune_random_forest(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_map: Dict[int, str],
    class_weights: Dict[int, float],
    output_dir: Path,
    random_seed: int,
) -> Dict[str, int]:
    param_grid = [
        {
            "n_estimators": [140, 220],
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.7],
        }
    ]

    tuning_rows = []
    best_metrics = None
    best_params = None

    print("\n--- Tuning Random Forest (clinical-priority objective) ---")
    for params in ParameterGrid(param_grid):
        model = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight=class_weights,
                        random_state=random_seed,
                        n_jobs=-1,
                        **params,
                    ),
                ),
            ]
        )
        model.fit(X_fit, y_fit)
        val_pred = model.predict(X_val)
        metrics = _build_selection_metrics(y_val, val_pred, class_map)

        row = {**params, **metrics}
        tuning_rows.append(row)

        if _is_better_candidate(metrics, best_metrics):
            best_metrics = metrics
            best_params = params

    pd.DataFrame(tuning_rows).sort_values(
        by=["min_recall_critical", "mean_recall_critical", "f1_macro"],
        ascending=False,
    ).to_csv(output_dir / "rf_tuning_results.csv", index=False)

    print("Best RF params:", best_params)
    print("Best RF selection metrics:", best_metrics)
    return best_params


def tune_xgboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_map: Dict[int, str],
    fit_weights: np.ndarray,
    n_classes: int,
    output_dir: Path,
    random_seed: int,
) -> Dict[str, float]:
    param_grid = [
        {
            "n_estimators": [180, 260],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.85, 1.0],
            "colsample_bytree": [0.85, 1.0],
            "reg_lambda": [1.0, 1.8],
        }
    ]

    tuning_rows = []
    best_metrics = None
    best_params = None

    print("\n--- Tuning XGBoost (clinical-priority objective) ---")
    for params in ParameterGrid(param_grid):
        model = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=n_classes,
                        eval_metric="mlogloss",
                        random_state=random_seed,
                        tree_method="hist",
                        **params,
                    ),
                ),
            ]
        )

        model.fit(X_fit, y_fit, clf__sample_weight=fit_weights)
        val_pred = model.predict(X_val)
        metrics = _build_selection_metrics(y_val, val_pred, class_map)

        row = {**params, **metrics}
        tuning_rows.append(row)

        if _is_better_candidate(metrics, best_metrics):
            best_metrics = metrics
            best_params = params

    pd.DataFrame(tuning_rows).sort_values(
        by=["min_recall_critical", "mean_recall_critical", "f1_macro"],
        ascending=False,
    ).to_csv(output_dir / "xgb_tuning_results.csv", index=False)

    print("Best XGB params:", best_params)
    print("Best XGB selection metrics:", best_metrics)
    return best_params


def run_stratified_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    class_map: Dict[int, str],
    output_dir: Path,
    n_splits: int,
    model_builder: Callable[[pd.Series], Pipeline],
    use_sample_weights: bool,
    random_seed: int,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    labels = sorted(class_map.keys())
    fold_rows = []

    print(f"\n--- Stratified CV: {model_name} ({n_splits} folds) ---")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        fold_weights_map = compute_class_weights(y_train_fold)
        model = model_builder(y_train_fold)

        fit_kwargs: Dict[str, Any] = {}
        if use_sample_weights:
            fit_kwargs["clf__sample_weight"] = y_train_fold.map(fold_weights_map).to_numpy()

        model.fit(X_train_fold, y_train_fold, **fit_kwargs)
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)

        sel_metrics = _build_selection_metrics(y_test_fold, y_pred, class_map)
        y_test_bin = label_binarize(y_test_fold, classes=labels)
        auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

        row = {
            "fold": fold_idx,
            "f1_macro": sel_metrics["f1_macro"],
            "auc_roc_macro": auc_macro,
            "mean_recall_critical": sel_metrics["mean_recall_critical"],
            "min_recall_critical": sel_metrics["min_recall_critical"],
        }
        for cls_id in CRITICAL_CLASS_IDS:
            cls_name = class_map[cls_id]
            row[f"recall_{cls_name}"] = sel_metrics[f"recall_{cls_name}"]

        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    model_slug = model_name.lower().replace(" ", "_")
    folds_df.to_csv(output_dir / f"{model_slug}_cv_folds.csv", index=False)

    summary = {"model": model_name}
    for col in folds_df.columns:
        if col == "fold":
            continue
        summary[f"{col}_mean"] = float(folds_df[col].mean())
        summary[f"{col}_std"] = float(folds_df[col].std(ddof=0))

    print("CV summary:", summary)
    return summary


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


def _to_c_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in values.tolist())


def export_rf_for_esp32(
    rf_pipeline: Pipeline,
    feature_names: List[str],
    class_map: Dict[int, str],
    output_dir: Path,
    deployment_summary: Dict[str, Any],
) -> None:
    export_dir = output_dir / "esp32_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_pipeline, export_dir / "rf_pipeline.joblib")

    scaler = rf_pipeline.named_steps["scaler"]
    medians = np.asarray(scaler.center_, dtype=np.float64)
    iqrs = np.asarray(scaler.scale_, dtype=np.float64)
    iqrs = np.where(np.abs(iqrs) < 1e-12, 1.0, iqrs)

    with open(export_dir / "esp32_preprocess.h", "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("// Auto-generated RobustScaler parameters for ESP32 inference.\n")
        f.write(f"static const int N_FEATURES = {len(feature_names)};\n")
        f.write(f"static const float ROBUST_CENTER[{len(feature_names)}] = {{{_to_c_array(medians)}}};\n")
        f.write(f"static const float ROBUST_IQR[{len(feature_names)}] = {{{_to_c_array(iqrs)}}};\n\n")
        f.write("inline void robust_scale(const float in_features[N_FEATURES], float out_features[N_FEATURES]) {\n")
        f.write("  for (int i = 0; i < N_FEATURES; ++i) {\n")
        f.write("    out_features[i] = (in_features[i] - ROBUST_CENTER[i]) / ROBUST_IQR[i];\n")
        f.write("  }\n")
        f.write("}\n")

    pd.DataFrame(
        {
            "feature_index": list(range(len(feature_names))),
            "feature_name": feature_names,
            "robust_center": medians,
            "robust_iqr": iqrs,
        }
    ).to_csv(export_dir / "feature_manifest.csv", index=False)

    pd.DataFrame(
        {
            "condition_id": sorted(class_map.keys()),
            "condition_name": [class_map[i] for i in sorted(class_map.keys())],
        }
    ).to_csv(export_dir / "class_labels.csv", index=False)

    with open(export_dir / "deployment_summary.json", "w", encoding="utf-8") as f:
        json.dump(deployment_summary, f, indent=2, ensure_ascii=True)

    with open(export_dir / "esp32_inference_template.ino", "w", encoding="utf-8") as f:
        f.write('#include "esp32_preprocess.h"\n')
        f.write('#include "esp32_rf_model.h"\n\n')
        f.write("float raw_features[N_FEATURES] = {\n")
        f.write("  4.9f, 14.2f, 43.0f, 90.0f, 13.5f, 7.2f, 55.0f, 33.0f, 2.5f, 6.5f, 250.0f, 0.2f\n")
        f.write("};\n\n")
        f.write("void setup() {\n")
        f.write("  Serial.begin(115200);\n")
        f.write("  while (!Serial) { }\n")
        f.write("  float scaled[N_FEATURES];\n")
        f.write("  robust_scale(raw_features, scaled);\n")
        f.write("  int prediction = Eloquent::ML::Port::RandomForest().predict(scaled);\n")
        f.write('  Serial.print("Predicted class id: ");\n')
        f.write("  Serial.println(prediction);\n")
        f.write("}\n\n")
        f.write("void loop() { }\n")

    try:
        from micromlgen import port

        model = rf_pipeline.named_steps["clf"]
        model_code = port(model)
        with open(export_dir / "esp32_rf_model.h", "w", encoding="utf-8") as f:
            f.write("// Auto-generated model for embedded inference on ESP32\n")
            f.write("// Feature order must match exactly:\n")
            for idx, name in enumerate(feature_names):
                f.write(f"// {idx}: {name}\n")
            f.write("// IMPORTANT: apply RobustScaler first using esp32_preprocess.h\n\n")
            f.write(model_code)
    except ImportError:
        print("micromlgen is not installed. Skipping C header export for ESP32.")


def build_deployment_summary(
    all_metrics: Dict[str, Dict[str, float]],
    cv_summaries: List[Dict[str, Any]],
    rf_params: Dict[str, Any],
    esp32_rf_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "deployment_target": "ESP32",
        "selected_model": "Random Forest ESP32",
        "selection_reason": "Selected for ESP32 as compact forest balancing memory footprint and critical recall.",
        "test_metrics": all_metrics,
        "cv_summary": cv_summaries,
        "chosen_hyperparameters": {
            "Random Forest": rf_params,
            "Random Forest ESP32": esp32_rf_params,
            "XGBoost": xgb_params,
        },
    }


def run_training(config: TrainingConfig) -> None:
    df = pd.read_csv(config.data_path)
    X, y, class_map = build_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.random_seed,
    )

    class_weights = compute_class_weights(y_train)
    train_weights = y_train.map(class_weights).to_numpy()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.tune:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            stratify=y_train,
            random_state=config.random_seed,
        )
        fit_weights = y_fit.map(class_weights).to_numpy()

        rf_best_params = tune_random_forest(
            X_fit=X_fit,
            y_fit=y_fit,
            X_val=X_val,
            y_val=y_val,
            class_map=class_map,
            class_weights=class_weights,
            output_dir=output_dir,
            random_seed=config.random_seed,
        )
    else:
        rf_best_params = {
            "n_estimators": 120,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }

    rf_pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight=class_weights,
                    random_state=config.random_seed,
                    n_jobs=-1,
                    **rf_best_params,
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
    xgb_best_params: Dict[str, Any] = {}

    esp32_rf_params = {
        "n_estimators": 32,
        "max_depth": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    }
    esp32_rf_pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight=class_weights,
                    random_state=config.random_seed,
                    n_jobs=-1,
                    **esp32_rf_params,
                ),
            ),
        ]
    )

    print("\n=== Training Random Forest ESP32 (compact) ===")
    esp32_rf_metrics = evaluate_model(
        model_name="Random Forest ESP32",
        model=esp32_rf_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_map=class_map,
        output_dir=output_dir,
    )

    print("Random Forest ESP32 metrics:")
    for k, v in esp32_rf_metrics.items():
        print(f"  {k}: {v:.4f}")

    all_metrics["Random Forest ESP32"] = esp32_rf_metrics

    if XGBOOST_AVAILABLE:
        if config.tune:
            xgb_best_params = tune_xgboost(
                X_fit=X_fit,
                y_fit=y_fit,
                X_val=X_val,
                y_val=y_val,
                class_map=class_map,
                fit_weights=fit_weights,
                n_classes=len(class_map),
                output_dir=output_dir,
                random_seed=config.random_seed,
            )
        else:
            xgb_best_params = {
                "n_estimators": 220,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
            }

        xgb_pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(class_map),
                        eval_metric="mlogloss",
                        random_state=config.random_seed,
                        tree_method="hist",
                        **xgb_best_params,
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

    cv_summaries: List[Dict[str, Any]] = []
    if config.cv_folds and config.cv_folds >= 2:
        def rf_builder(y_train_fold: pd.Series) -> Pipeline:
            fold_class_weights = compute_class_weights(y_train_fold)
            return Pipeline(
                [
                    ("scaler", RobustScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight=fold_class_weights,
                            random_state=config.random_seed,
                            n_jobs=-1,
                            **rf_best_params,
                        ),
                    ),
                ]
            )

        cv_summaries.append(
            run_stratified_cv(
                model_name="Random Forest",
                X=X,
                y=y,
                class_map=class_map,
                output_dir=output_dir,
                n_splits=config.cv_folds,
                model_builder=rf_builder,
                use_sample_weights=False,
                random_seed=config.random_seed,
            )
        )

        if XGBOOST_AVAILABLE:
            def xgb_builder(_: pd.Series) -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        (
                            "clf",
                            XGBClassifier(
                                objective="multi:softprob",
                                num_class=len(class_map),
                                eval_metric="mlogloss",
                                random_state=config.random_seed,
                                tree_method="hist",
                                **xgb_best_params,
                            ),
                        ),
                    ]
                )

            cv_summaries.append(
                run_stratified_cv(
                    model_name="XGBoost",
                    X=X,
                    y=y,
                    class_map=class_map,
                    output_dir=output_dir,
                    n_splits=config.cv_folds,
                    model_builder=xgb_builder,
                    use_sample_weights=True,
                    random_seed=config.random_seed,
                )
            )

        pd.DataFrame(cv_summaries).to_csv(output_dir / "cv_summary.csv", index=False)
    elif config.cv_folds == 1:
        print("\nSkipping CV: cv_folds must be 0 or >= 2")

    feature_names = X.columns.tolist()
    deployment_summary = build_deployment_summary(
        all_metrics=all_metrics,
        cv_summaries=cv_summaries,
        rf_params=rf_best_params,
        esp32_rf_params=esp32_rf_params,
        xgb_params=xgb_best_params,
    )
    export_rf_for_esp32(
        rf_pipeline=esp32_rf_pipeline,
        feature_names=feature_names,
        class_map=class_map,
        output_dir=output_dir,
        deployment_summary=deployment_summary,
    )

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=True)

    print("\nSaved artifacts in:", output_dir)
    print(metrics_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate multiclass CBC classifiers.")
    parser.add_argument("--data", default="hemograma_data.csv", help="Path to dataset CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store metrics and plots")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning prioritizing recall for critical classes before final evaluation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for stratified cross-validation. Set 0 to skip CV.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for reproducibility")
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    cfg = TrainingConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        tune=args.tune,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
    )
    run_training(cfg)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
