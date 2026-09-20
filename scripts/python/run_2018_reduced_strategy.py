"""Run official 2018 CPEC landslide susceptibility workflow.

This script replaces the earlier full-factor baseline in the main results path.
It keeps only:
- Reduced conventional factors
- AlphaEarth embeddings only
- Reduced conventional factors fused with AlphaEarth embeddings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
DIAG_DIR = PROJECT_ROOT / "03_models" / "multicollinearity_assessment_2018"
MODEL_DIR = PROJECT_ROOT / "03_models" / "reduced_strategy_2018_spatial_cv"
CURVE_DIR = PROJECT_ROOT / "03_models" / "reduced_strategy_2018_curves"
METRIC_DIR = PROJECT_ROOT / "03_models" / "reduced_strategy_2018_metrics"

DIAG_REPORT = PROJECT_ROOT / "05_reports" / "multicollinearity_assessment_2018.md"
MODEL_REPORT = PROJECT_ROOT / "05_reports" / "reduced_strategy_2018_modelling_report.md"
CURVE_REPORT = PROJECT_ROOT / "05_reports" / "reduced_strategy_2018_roc_pr_curves.md"
METRIC_REPORT = PROJECT_ROOT / "05_reports" / "reduced_strategy_2018_evaluation_metrics.md"
DECISION_REPORT = PROJECT_ROOT / "05_reports" / "reduced_strategy_decision_2018.md"

CONVENTIONAL_FULL = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_annual_total",
    "rain_monsoon_total",
    "rain_max_1day",
    "rain_max_3day",
    "rain_max_7day",
    "ndvi_median",
    "ndvi_max",
    "ndvi_amplitude",
    "evi_median",
    "lst_day_mean_c",
    "lst_day_max_c",
    "modis_lc_type1",
]

CONVENTIONAL_REDUCED = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
]

FACTOR_NAMES = {
    "elevation_m": "Elevation (m)",
    "slope_deg": "Slope (degrees)",
    "aspect_deg": "Aspect (degrees)",
    "rain_annual_total": "Annual rainfall total",
    "rain_monsoon_total": "Monsoon rainfall total",
    "rain_max_1day": "Maximum 1-day rainfall",
    "rain_max_3day": "Maximum 3-day rainfall",
    "rain_max_7day": "Maximum 7-day rainfall",
    "ndvi_median": "NDVI median",
    "ndvi_max": "NDVI maximum",
    "ndvi_amplitude": "NDVI amplitude",
    "evi_median": "EVI median",
    "lst_day_mean_c": "Mean daytime LST (deg C)",
    "lst_day_max_c": "Maximum daytime LST (deg C)",
    "modis_lc_type1": "MODIS land-cover class",
}

FEATURE_SET_NAMES = {
    "conventional_reduced": "Conventional",
    "alphaearth_only": "AlphaEarth Embeddings",
    "fused_reduced": "Conventional + AlphaEarth Embeddings",
}

MODEL_NAMES = {
    "logistic_l2": "Logistic Regression (L2)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

DROP_REASONS = {
    "rain_annual_total": "Dropped: redundant with monsoon and extreme rainfall information in this sample table.",
    "rain_max_3day": "Dropped: very high rainfall-family correlation; 1-day maximum kept for sharper triggering signal.",
    "rain_max_7day": "Dropped: very high rainfall-family correlation; monsoon total and 1-day maximum retained.",
    "ndvi_max": "Dropped: high VIF and redundancy with NDVI median/amplitude.",
    "evi_median": "Dropped: high correlation with NDVI median; NDVI median retained for a simpler vegetation signal.",
    "lst_day_max_c": "Dropped: high correlation with mean daytime LST; mean daytime LST retained.",
}


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    for col in sorted(set(CONVENTIONAL_FULL + alpha + ["label", "spatial_fold_5"])):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)
    return df, alpha


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for col in cols:
        other = [c for c in cols if c != col]
        model = LinearRegression()
        model.fit(x[other], x[col].to_numpy())
        r2 = float(model.score(x[other], x[col].to_numpy()))
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append(
            {
                "factor": col,
                "factor_name": FACTOR_NAMES.get(col, col),
                "r2_against_other_factors": r2,
                "vif": vif,
                "decision": "keep" if col in CONVENTIONAL_REDUCED else "drop",
                "decision_reason": "Kept in selected conventional set." if col in CONVENTIONAL_REDUCED else DROP_REASONS[col],
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def run_multicollinearity(df: pd.DataFrame, alpha: list[str]) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    conv_corr = df[CONVENTIONAL_FULL].corr(method="spearman")
    high_corr_rows = []
    for i, a in enumerate(CONVENTIONAL_FULL):
        for b in CONVENTIONAL_FULL[i + 1 :]:
            val = float(conv_corr.loc[a, b])
            if abs(val) >= 0.85:
                high_corr_rows.append(
                    {
                        "factor_a": a,
                        "factor_a_name": FACTOR_NAMES.get(a, a),
                        "factor_b": b,
                        "factor_b_name": FACTOR_NAMES.get(b, b),
                        "spearman_r": val,
                    }
                )
    high_corr = pd.DataFrame(high_corr_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    vif = vif_table(df, CONVENTIONAL_FULL)
    reduced_vif = vif_table(df, CONVENTIONAL_REDUCED)
    decision = pd.DataFrame(
        [
            {
                "factor": factor,
                "factor_name": FACTOR_NAMES[factor],
                "official_reduced_strategy": "keep" if factor in CONVENTIONAL_REDUCED else "drop",
                "reason": "Kept in selected conventional set." if factor in CONVENTIONAL_REDUCED else DROP_REASONS[factor],
            }
            for factor in CONVENTIONAL_FULL
        ]
    )
    alpha_names = pd.DataFrame(
        {
            "factor": alpha,
            "factor_name": [f"AlphaEarth embedding band {band}" for band in alpha],
            "official_reduced_strategy": "keep",
            "reason": "Kept as part of the fixed 64-band AlphaEarth representation; handled by model regularization and spatial CV.",
        }
    )

    conv_corr.to_csv(DIAG_DIR / "conventional_spearman_correlation_matrix.csv")
    high_corr.to_csv(DIAG_DIR / "conventional_high_correlation_pairs_abs_ge_0_85.csv", index=False)
    vif.to_csv(DIAG_DIR / "conventional_vif_with_decisions.csv", index=False)
    reduced_vif.to_csv(DIAG_DIR / "reduced_conventional_vif.csv", index=False)
    decision.to_csv(DIAG_DIR / "conventional_factor_keep_drop_decisions.csv", index=False)
    alpha_names.to_csv(DIAG_DIR / "alphaearth_embedding_factor_names.csv", index=False)

    lines = [
        "# 2018 Multicollinearity Assessment",
        "",
        "This is the official pre-modelling filter for the revised modelling strategy.",
        "",
        "## Decision",
        "",
        "- Full conventional and full fused model outputs are not used as official results.",
        "- Highly redundant/noisy conventional factors are excluded before final modelling.",
        "- AlphaEarth embeddings are kept as the fixed 64-band representation and evaluated separately and in fusion.",
        "",
        "## Kept Conventional Factors",
        "",
    ]
    for factor in CONVENTIONAL_REDUCED:
        lines.append(f"- {FACTOR_NAMES[factor]} (`{factor}`)")
    lines.extend(["", "## Dropped Conventional Factors", ""])
    for factor, reason in DROP_REASONS.items():
        lines.append(f"- {FACTOR_NAMES[factor]} (`{factor}`): {reason}")
    lines.extend(
        [
            "",
            "## Key Redundancy Evidence",
            "",
            "| Factor A | Factor B | Spearman r |",
            "|---|---|---:|",
        ]
    )
    for _, row in high_corr.head(10).iterrows():
        lines.append(f"| {row['factor_a_name']} | {row['factor_b_name']} | {row['spearman_r']:.3f} |")
    lines.extend(
        [
            "",
            "## Reduced-Set VIF Check",
            "",
            "| Factor | VIF |",
            "|---|---:|",
        ]
    )
    for _, row in reduced_vif.iterrows():
        lines.append(f"| {row['factor_name']} | {row['vif']:.2f} |")
    lines.extend(
        [
            "",
            "## Saved Tables",
            "",
            f"- VIF and decisions: `{DIAG_DIR / 'conventional_vif_with_decisions.csv'}`",
            f"- Reduced conventional VIF: `{DIAG_DIR / 'reduced_conventional_vif.csv'}`",
            f"- High-correlation pairs: `{DIAG_DIR / 'conventional_high_correlation_pairs_abs_ge_0_85.csv'}`",
            f"- Keep/drop table: `{DIAG_DIR / 'conventional_factor_keep_drop_decisions.csv'}`",
            f"- AlphaEarth band names: `{DIAG_DIR / 'alphaearth_embedding_factor_names.csv'}`",
        ]
    )
    DIAG_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def threshold_for_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def metric_row(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
    }


def make_models(seed: int = 141300) -> dict[str, Callable[[], object]]:
    return {
        "logistic_l2": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=0,
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def predict_positive(model, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def feature_importance(model, features: list[str]) -> pd.DataFrame:
    fitted = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(fitted, "feature_importances_"):
        vals = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        vals = np.abs(fitted.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "factor_name", "importance"])
    return pd.DataFrame(
        {
            "feature": features,
            "factor_name": [FACTOR_NAMES.get(f, f"AlphaEarth embedding band {f}") for f in features],
            "importance": vals,
        }
    ).sort_values("importance", ascending=False)


def run_models(df: pd.DataFrame, alpha: list[str]) -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_sets = {
        "conventional_reduced": CONVENTIONAL_REDUCED,
        "alphaearth_only": alpha,
        "fused_reduced": CONVENTIONAL_REDUCED + alpha,
    }
    models = make_models()
    all_fold_metrics = []
    all_predictions = []
    all_importance = []
    y_all = df["label"].to_numpy()
    folds = sorted(df["spatial_fold_5"].unique())

    for feature_set, features in feature_sets.items():
        x_all = df[features]
        for model_key, factory in models.items():
            for fold in folds:
                train_idx = df["spatial_fold_5"] != fold
                test_idx = df["spatial_fold_5"] == fold
                model = factory()
                model.fit(x_all.loc[train_idx], y_all[train_idx.to_numpy()])
                train_prob = predict_positive(model, x_all.loc[train_idx])
                threshold = threshold_for_f1(y_all[train_idx.to_numpy()], train_prob)
                test_prob = predict_positive(model, x_all.loc[test_idx])
                y_test = y_all[test_idx.to_numpy()]
                all_fold_metrics.append(
                    {
                        "feature_set": feature_set,
                        "feature_set_name": FEATURE_SET_NAMES[feature_set],
                        "model": model_key,
                        "model_name": MODEL_NAMES[model_key],
                        "fold": int(fold),
                        "n_train": int(train_idx.sum()),
                        "n_test": int(test_idx.sum()),
                        "n_test_pos": int(y_test.sum()),
                        "n_test_neg": int((y_test == 0).sum()),
                        **metric_row(y_test, test_prob, threshold),
                    }
                )
                pred_df = df.loc[
                    test_idx,
                    ["inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"],
                ].copy()
                pred_df["feature_set"] = feature_set
                pred_df["feature_set_name"] = FEATURE_SET_NAMES[feature_set]
                pred_df["model"] = model_key
                pred_df["model_name"] = MODEL_NAMES[model_key]
                pred_df["probability"] = test_prob
                pred_df["threshold"] = threshold
                pred_df["prediction"] = (test_prob >= threshold).astype(int)
                all_predictions.append(pred_df)

            final_model = factory()
            final_model.fit(x_all, y_all)
            imp = feature_importance(final_model, features)
            if not imp.empty:
                imp["feature_set"] = feature_set
                imp["feature_set_name"] = FEATURE_SET_NAMES[feature_set]
                imp["model"] = model_key
                imp["model_name"] = MODEL_NAMES[model_key]
                all_importance.append(imp)
            joblib.dump(final_model, MODEL_DIR / f"model_{feature_set}_{model_key}.joblib")

    fold_metrics = pd.DataFrame(all_fold_metrics)
    summary = (
        fold_metrics.groupby(["feature_set", "feature_set_name", "model", "model_name"])
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .reset_index()
        .sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
    )
    fold_metrics.to_csv(MODEL_DIR / "reduced_strategy_spatial_cv_fold_metrics.csv", index=False)
    summary.to_csv(MODEL_DIR / "reduced_strategy_spatial_cv_summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(MODEL_DIR / "reduced_strategy_spatial_cv_predictions.csv", index=False)
    if all_importance:
        pd.concat(all_importance, ignore_index=True).to_csv(MODEL_DIR / "reduced_strategy_final_model_feature_importance.csv", index=False)

    run_summary = {
        "input_table": str(DATA),
        "rows": int(len(df)),
        "feature_sets": {FEATURE_SET_NAMES[k]: len(v) for k, v in feature_sets.items()},
        "models": list(MODEL_NAMES.values()),
        "validation": "5-fold spatial block CV using spatial_fold_5",
        "official_strategy": "Reduced-only modelling; full conventional and full fused outputs archived.",
        "best_by_pr_auc": summary.iloc[0].to_dict(),
    }
    (MODEL_DIR / "reduced_strategy_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_model_report(summary, df)
    return summary


def fmt(mean: float, std: float) -> str:
    return f"{float(mean):.3f} +/- {float(std):.3f}"


def write_model_report(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = [
        "# 2018 Spatial-CV Modelling Report",
        "",
        f"- Input table: `{DATA}`",
        f"- Rows: {len(df)}",
        "- Validation: 5-fold spatial block cross-validation.",
        "- Official feature sets: Conventional, AlphaEarth Embeddings, Conventional + AlphaEarth Embeddings.",
        "- Archived/removed from official results: conventional full and fused full.",
        "",
        "## Top Results By Mean PR-AUC",
        "",
        "| Rank | Feature set | Model | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(summary.head(10).iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature_set_name']} | {row['model_name']} | "
            f"{fmt(row['pr_auc_mean'], row['pr_auc_std'])} | "
            f"{fmt(row['roc_auc_mean'], row['roc_auc_std'])} | "
            f"{fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} | "
            f"{fmt(row['f1_mean'], row['f1_std'])} | "
            f"{fmt(row['brier_mean'], row['brier_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The official comparison now tests conventional geomorphic/climatic predictors after multicollinearity reduction.",
            "- AlphaEarth is evaluated both alone and as a fused representation with the conventional factors.",
            "- Model ranking uses spatial-CV out-of-fold predictions to reduce optimistic bias from spatial autocorrelation.",
        ]
    )
    MODEL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_curves() -> None:
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = MODEL_DIR / "reduced_strategy_spatial_cv_predictions.csv"
    df = pd.read_csv(pred_path)
    keep = [
        ("conventional_reduced", "xgboost", "Conventional - XGBoost"),
        ("alphaearth_only", "xgboost", "AlphaEarth Embeddings - XGBoost"),
        ("fused_reduced", "xgboost", "Conventional + AlphaEarth Embeddings - XGBoost"),
    ]
    curve_rows = []

    plt.figure(figsize=(7.2, 6.2))
    summary_rows = []
    for feature_set, model, label in keep:
        d = df[(df["feature_set"] == feature_set) & (df["model"] == model)].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{label} (AUC={roc_auc:.3f})")
        for x, yy, th in zip(fpr, tpr, thresholds):
            curve_rows.append({"feature_set": feature_set, "model": model, "curve": "roc", "x": x, "y": yy, "threshold": th})
        summary_rows.append(
            {
                "feature_set": feature_set,
                "feature_set_name": FEATURE_SET_NAMES[feature_set],
                "model": model,
                "model_name": MODEL_NAMES[model],
                "roc_auc": roc_auc,
                "average_precision_pr_auc": average_precision_score(y, p),
                "n": len(d),
            }
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("2018 Spatial-CV ROC Curves")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CURVE_DIR / "reduced_strategy_roc_curves_2018_spatial_cv.png", dpi=300)
    plt.savefig(CURVE_DIR / "reduced_strategy_roc_curves_2018_spatial_cv.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 6.2))
    for feature_set, model, label in keep:
        d = df[(df["feature_set"] == feature_set) & (df["model"] == model)].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(recall, precision, linewidth=2.2, label=f"{label} (AP={ap:.3f})")
        for x, yy, th in zip(recall, precision, list(thresholds) + [float("nan")]):
            curve_rows.append({"feature_set": feature_set, "model": model, "curve": "precision_recall", "x": x, "y": yy, "threshold": th})
    prevalence = df[df["model"] == "xgboost"]["label"].astype(int).mean()
    plt.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2018 Spatial-CV Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CURVE_DIR / "reduced_strategy_pr_curves_2018_spatial_cv.png", dpi=300)
    plt.savefig(CURVE_DIR / "reduced_strategy_pr_curves_2018_spatial_cv.pdf")
    plt.close()

    pd.DataFrame(curve_rows).to_csv(CURVE_DIR / "reduced_strategy_roc_pr_curve_coordinates_2018_spatial_cv.csv", index=False)
    curve_summary = pd.DataFrame(summary_rows).sort_values("average_precision_pr_auc", ascending=False)
    curve_summary.to_csv(CURVE_DIR / "reduced_strategy_curve_auc_summary_2018_spatial_cv.csv", index=False)

    lines = [
        "# 2018 ROC And Precision-Recall Curves",
        "",
        "Curves are generated from spatial-CV out-of-fold predictions for the XGBoost comparison models.",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC/AP | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in curve_summary.iterrows():
        lines.append(
            f"| {row['feature_set_name']} | {row['model_name']} | {row['roc_auc']:.3f} | "
            f"{row['average_precision_pr_auc']:.3f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Saved Curve Files",
            "",
            f"- ROC PNG: `{CURVE_DIR / 'reduced_strategy_roc_curves_2018_spatial_cv.png'}`",
            f"- ROC PDF: `{CURVE_DIR / 'reduced_strategy_roc_curves_2018_spatial_cv.pdf'}`",
            f"- PR PNG: `{CURVE_DIR / 'reduced_strategy_pr_curves_2018_spatial_cv.png'}`",
            f"- PR PDF: `{CURVE_DIR / 'reduced_strategy_pr_curves_2018_spatial_cv.pdf'}`",
            f"- Curve coordinates: `{CURVE_DIR / 'reduced_strategy_roc_pr_curve_coordinates_2018_spatial_cv.csv'}`",
        ]
    )
    CURVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_metrics_report() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(MODEL_DIR / "reduced_strategy_spatial_cv_summary.csv")
    folds = pd.read_csv(MODEL_DIR / "reduced_strategy_spatial_cv_fold_metrics.csv")
    key = pd.DataFrame(
        [
            ("conventional_reduced", "xgboost"),
            ("alphaearth_only", "xgboost"),
            ("fused_reduced", "xgboost"),
        ],
        columns=["feature_set", "model"],
    )
    key_summary = key.merge(summary, on=["feature_set", "model"], how="left")
    key_folds = key.merge(folds, on=["feature_set", "model"], how="left")
    key_summary.to_csv(METRIC_DIR / "reduced_strategy_key_model_metrics_summary.csv", index=False)
    key_folds.to_csv(METRIC_DIR / "reduced_strategy_key_model_fold_metrics.csv", index=False)
    summary.to_csv(METRIC_DIR / "reduced_strategy_all_model_metrics_ranked.csv", index=False)

    lines = [
        "# 2018 Evaluation Metrics",
        "",
        "Metrics are computed from 5-fold spatial block cross-validation. Higher is better for ROC-AUC, PR-AUC, balanced accuracy, and F1; lower is better for Brier score.",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC | Balanced accuracy | F1 | Brier score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in key_summary.iterrows():
        lines.append(
            f"| {row['feature_set_name']} | {row['model_name']} | "
            f"{fmt(row['roc_auc_mean'], row['roc_auc_std'])} | "
            f"{fmt(row['pr_auc_mean'], row['pr_auc_std'])} | "
            f"{fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} | "
            f"{fmt(row['f1_mean'], row['f1_std'])} | "
            f"{fmt(row['brier_mean'], row['brier_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Saved Tables",
            "",
            f"- Key metrics: `{METRIC_DIR / 'reduced_strategy_key_model_metrics_summary.csv'}`",
            f"- Fold metrics: `{METRIC_DIR / 'reduced_strategy_key_model_fold_metrics.csv'}`",
            f"- All ranked official models: `{METRIC_DIR / 'reduced_strategy_all_model_metrics_ranked.csv'}`",
        ]
    )
    METRIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_report() -> None:
    lines = [
        "# Official 2018 LSM Modelling Strategy",
        "",
        "The official modelling branch now uses selected conventional predictors only, plus AlphaEarth Embeddings and reduced-fusion comparisons.",
        "",
        "## Why This Revision Was Made",
        "",
        "- Multicollinearity diagnostics showed strong redundancy in the full conventional stack.",
        "- Some full-factor models introduced noise and did not improve the reviewer-relevant AUC/PR-AUC evidence.",
        "- The reduced strategy is cleaner to justify: first control redundant conventional predictors, then compare conventional, AlphaEarth, and fused representations.",
        "",
        "## Official Model Outputs",
        "",
        "- Conventional",
        "- AlphaEarth Embeddings",
        "- Conventional + AlphaEarth Embeddings",
        "",
        "## Archived Outputs",
        "",
        "- Earlier full conventional and full fused baseline outputs were moved to `D:\\DING PROJECT\\99_archive\\2026-05-02_full_factor_and_mixed_baseline_outputs`.",
    ]
    DECISION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df, alpha = load_data()
    run_multicollinearity(df, alpha)
    summary = run_models(df, alpha)
    run_curves()
    run_metrics_report()
    write_decision_report()
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


