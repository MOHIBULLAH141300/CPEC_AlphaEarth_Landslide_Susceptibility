"""Interpret and robustness-check the 2018 baseline LSM models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
BASELINE_DIR = PROJECT_ROOT / "03_models" / "baseline_2018_spatial_cv"
OUT_DIR = PROJECT_ROOT / "03_models" / "baseline_2018_interpretation"
REPORT = PROJECT_ROOT / "05_reports" / "baseline_2018_interpretation_report.md"

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


def make_xgb(seed: int = 141300) -> XGBClassifier:
    return XGBClassifier(
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
    )


def scorer_pr_auc(estimator, x, y):
    return average_precision_score(y, estimator.predict_proba(x)[:, 1])


def load_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    feature_sets = {
        "conventional_reduced": CONVENTIONAL_REDUCED,
        "alphaearth_only": alpha,
        "fused_reduced": CONVENTIONAL_REDUCED + alpha,
    }
    for col in sorted(set().union(*[set(v) for v in feature_sets.values()])) + ["label", "spatial_fold_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)
    return df, feature_sets


def run_spatial_permutation(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for feature_set, features in feature_sets.items():
        for fold in sorted(df["spatial_fold_5"].unique()):
            train_idx = df["spatial_fold_5"] != fold
            test_idx = df["spatial_fold_5"] == fold
            model = make_xgb()
            model.fit(df.loc[train_idx, features], df.loc[train_idx, "label"])
            result = permutation_importance(
                model,
                df.loc[test_idx, features],
                df.loc[test_idx, "label"],
                scoring=scorer_pr_auc,
                n_repeats=8,
                random_state=141300 + int(fold),
                n_jobs=-1,
            )
            for feature, mean, std in zip(features, result.importances_mean, result.importances_std):
                rows.append(
                    {
                        "feature_set": feature_set,
                        "fold": fold,
                        "feature": feature,
                        "pr_auc_drop_mean": mean,
                        "pr_auc_drop_std": std,
                    }
                )
    imp = pd.DataFrame(rows)
    imp.to_csv(OUT_DIR / "spatial_fold_permutation_importance.csv", index=False)
    summary = (
        imp.groupby(["feature_set", "feature"])
        .agg(
            pr_auc_drop_mean=("pr_auc_drop_mean", "mean"),
            pr_auc_drop_std_across_folds=("pr_auc_drop_mean", "std"),
        )
        .reset_index()
        .sort_values(["feature_set", "pr_auc_drop_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT_DIR / "spatial_fold_permutation_importance_summary.csv", index=False)
    return summary


def run_random_cv_diagnostic(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=141300)
    for feature_set, features in feature_sets.items():
        x = df[features].to_numpy()
        y = df["label"].to_numpy()
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
            model = make_xgb()
            model.fit(x[train_idx], y[train_idx])
            prob = model.predict_proba(x[test_idx])[:, 1]
            rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y[test_idx], prob),
                    "pr_auc": average_precision_score(y[test_idx], prob),
                    "validation": "random_stratified_cv",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "random_cv_xgboost_fold_metrics.csv", index=False)
    summary = (
        out.groupby("feature_set")
        .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
        .reset_index()
        .sort_values("pr_auc_mean", ascending=False)
    )
    summary.to_csv(OUT_DIR / "random_cv_xgboost_summary.csv", index=False)
    return summary


def run_shap_best(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    model = joblib.load(BASELINE_DIR / "model_fused_reduced_xgboost.joblib")
    x = df[features]
    sample = x.sample(n=min(1200, len(x)), random_state=141300)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    vals = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": features, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False)
    out.to_csv(OUT_DIR / "fused_reduced_xgboost_shap_importance.csv", index=False)

    top = out.head(25)
    plt.figure(figsize=(9, 8))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#315a9a")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top SHAP Features - Fused Reduced XGBoost")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fused_reduced_xgboost_shap_top25.png", dpi=220)
    plt.close()
    return out


def plot_permutation(summary: pd.DataFrame) -> None:
    for feature_set in summary["feature_set"].unique():
        top = summary[summary["feature_set"] == feature_set].head(20)
        plt.figure(figsize=(9, 7))
        plt.barh(top["feature"][::-1], top["pr_auc_drop_mean"][::-1], color="#476f52")
        plt.xlabel("Mean PR-AUC drop when permuted")
        plt.title(f"Permutation Importance - {feature_set}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"permutation_importance_{feature_set}_top20.png", dpi=220)
        plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_sets = load_data()
    permutation_summary = run_spatial_permutation(df, feature_sets)
    random_summary = run_random_cv_diagnostic(df, feature_sets)
    shap_summary = run_shap_best(df, feature_sets["fused_reduced"])
    plot_permutation(permutation_summary)

    spatial_summary = pd.read_csv(BASELINE_DIR / "spatial_cv_summary.csv")
    spatial_xgb = spatial_summary[spatial_summary["model"] == "xgboost"][
        ["feature_set", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"]
    ].copy()
    spatial_xgb = spatial_xgb[spatial_xgb["feature_set"].isin(feature_sets.keys())]
    compare = spatial_xgb.merge(random_summary, on="feature_set", suffixes=("_spatial", "_random"))
    compare["pr_auc_random_minus_spatial"] = compare["pr_auc_mean_random"] - compare["pr_auc_mean_spatial"]
    compare["roc_auc_random_minus_spatial"] = compare["roc_auc_mean_random"] - compare["roc_auc_mean_spatial"]
    compare.to_csv(OUT_DIR / "spatial_vs_random_xgboost_comparison.csv", index=False)

    top_perm = permutation_summary[permutation_summary["feature_set"] == "fused_reduced"].head(15)
    top_shap = shap_summary.head(15)
    lines = [
        "# 2018 Baseline Interpretation And Robustness",
        "",
        "## What Was Added",
        "",
        "- Spatial-fold permutation importance for XGBoost on three key feature sets.",
        "- SHAP importance for the best Conventional + AlphaEarth Embeddings XGBoost model.",
        "- Random stratified CV diagnostic for XGBoost, used only to compare with spatial CV.",
        "",
        "## Top Fused Reduced Permutation Features",
        "",
        "| Rank | Feature | Mean PR-AUC drop | Fold stability SD |",
        "|---:|---|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top_perm.iterrows(), start=1):
        lines.append(f"| {rank} | {row['feature']} | {row['pr_auc_drop_mean']:.4f} | {row['pr_auc_drop_std_across_folds']:.4f} |")
    lines.extend(["", "## Top Fused Reduced SHAP Features", "", "| Rank | Feature | Mean abs SHAP |", "|---:|---|---:|"])
    for rank, (_, row) in enumerate(top_shap.iterrows(), start=1):
        lines.append(f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
    lines.extend(
        [
            "",
            "## Spatial Vs Random CV Diagnostic",
            "",
            "| Feature set | Spatial PR-AUC | Random PR-AUC | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in compare.iterrows():
        lines.append(
            f"| {row['feature_set']} | {row['pr_auc_mean_spatial']:.3f} | "
            f"{row['pr_auc_mean_random']:.3f} | {row['pr_auc_random_minus_spatial']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The interpretation layer should be treated as first-pass evidence, not final causal proof.",
            "- AlphaEarth features dominate the fused model, but conventional variables such as elevation/LST still appear among useful predictors.",
            "- Random CV is expected to be easier than spatial CV; the manuscript should emphasize spatial CV as the main estimate.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()

