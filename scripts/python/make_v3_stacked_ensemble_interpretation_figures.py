"""Create paper-facing interpretation figures for v3 stacked ensembles.

Public labels intentionally omit "v3":
- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

SHAP here explains the stacked meta-learner using base-model probabilities as
features. This is the correct interpretation level for the stacked ensemble.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(r"D:\DING PROJECT")
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
PRED = MODEL_DIR / "v3_model_family_spatial_cv_predictions.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "stacked_ensemble_interpretation"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "stacked_ensemble_interpretation_2018.md"

FEATURE_SET_LABELS = {
    "conventional_v3": "Conventional",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_v3_alphaearth_embeddings": "Conventional + AlphaEarth Embeddings",
}

FILE_LABELS = {
    "conventional_v3": "conventional",
    "alphaearth_embeddings": "alphaearth_embeddings",
    "conventional_v3_alphaearth_embeddings": "conventional_alphaearth_embeddings",
}

BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]
BASE_MODEL_LABELS = {
    "logistic_l2": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def meta_matrix(pred: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    d = pred[pred["feature_set"] == feature_set].copy()
    wide = d.pivot_table(index="sample_id", columns="model", values="probability", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide[BASE_MODEL_KEYS]


def meta_shap(feature_set: str, x: pd.DataFrame) -> pd.DataFrame:
    model_path = MODEL_DIR / f"model_{feature_set}_stacked_l2_logistic.joblib"
    model = joblib.load(model_path)
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["model"]
    x_scaled = pd.DataFrame(scaler.transform(x), columns=x.columns)
    x_display = x.rename(columns=BASE_MODEL_LABELS)
    explainer = shap.LinearExplainer(clf, x_scaled)
    shap_values = explainer.shap_values(x_scaled)
    vals = np.asarray(shap_values)
    imp = pd.DataFrame(
        {
            "feature_set": FEATURE_SET_LABELS[feature_set],
            "base_model": [BASE_MODEL_LABELS[col] for col in x.columns],
            "mean_abs_shap": np.abs(vals).mean(axis=0),
            "mean_shap": vals.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT_DIR / f"stacked_meta_shap_importance_{FILE_LABELS[feature_set]}.csv", index=False)

    plt.figure(figsize=(7.4, 4.8))
    top = imp.iloc[::-1]
    plt.barh(top["base_model"], top["mean_abs_shap"], color="#2f6f73")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Stacked Ensemble SHAP Importance: {FEATURE_SET_LABELS[feature_set]}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"figure_stacked_shap_importance_2018_{FILE_LABELS[feature_set]}.png", dpi=300)
    plt.savefig(FIG_DIR / f"figure_stacked_shap_importance_2018_{FILE_LABELS[feature_set]}.pdf")
    plt.close()

    plt.figure(figsize=(7.8, 5.4))
    shap.summary_plot(vals, x_display, show=False, max_display=6)
    plt.title(f"Stacked Ensemble SHAP Beeswarm: {FEATURE_SET_LABELS[feature_set]}", pad=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"figure_stacked_shap_beeswarm_2018_{FILE_LABELS[feature_set]}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"figure_stacked_shap_beeswarm_2018_{FILE_LABELS[feature_set]}.pdf", bbox_inches="tight")
    plt.close()
    return imp


def meta_coefficients() -> pd.DataFrame:
    weights = pd.read_csv(MODEL_DIR / "v3_model_family_meta_coefficients.csv")
    weights = weights[weights["feature_set"].isin(FEATURE_SET_LABELS)].copy()
    weights["feature_set_name_public"] = weights["feature_set"].map(FEATURE_SET_LABELS)
    weights["base_model_name_public"] = weights["base_model"].map(BASE_MODEL_LABELS)
    summary = (
        weights.groupby(["feature_set", "feature_set_name_public", "base_model", "base_model_name_public"])
        .agg(
            mean_coefficient=("meta_coefficient", "mean"),
            std_coefficient=("meta_coefficient", "std"),
            min_coefficient=("meta_coefficient", "min"),
            max_coefficient=("meta_coefficient", "max"),
        )
        .reset_index()
    )
    for feature_set, label in FEATURE_SET_LABELS.items():
        d = summary[summary["feature_set"] == feature_set].sort_values("mean_coefficient", ascending=True)
        plt.figure(figsize=(7.4, 4.8))
        plt.barh(d["base_model_name_public"], d["mean_coefficient"], color="#725a9c")
        plt.xlabel("Mean meta-learner coefficient")
        plt.title(f"Stacked Meta-Coefficients: {label}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"figure_stacked_meta_coefficients_2018_{FILE_LABELS[feature_set]}.png", dpi=300)
        plt.savefig(FIG_DIR / f"figure_stacked_meta_coefficients_2018_{FILE_LABELS[feature_set]}.pdf")
        plt.close()
    public_summary = pd.DataFrame(
        {
            "feature_set": summary["feature_set_name_public"],
            "base_model": summary["base_model_name_public"],
            "mean_coefficient": summary["mean_coefficient"],
            "std_coefficient": summary["std_coefficient"],
            "min_coefficient": summary["min_coefficient"],
            "max_coefficient": summary["max_coefficient"],
        }
    )
    public_summary.to_csv(OUT_DIR / "stacked_meta_coefficient_summary_all_feature_sets.csv", index=False)
    return summary


def calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(7.2, 6.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Perfect calibration")
    for feature_set, label in FEATURE_SET_LABELS.items():
        d = pred[(pred["feature_set"] == feature_set) & (pred["model"] == "stacked_l2_logistic")].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2.3, label=f"{label} (Brier={brier:.3f})")
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
            rows.append(
                {
                    "feature_set": label,
                    "bin": i,
                    "mean_predicted_probability": mp,
                    "fraction_observed_landslide": fp,
                    "brier": brier,
                }
            )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed landslide fraction")
    plt.title("Stacked Ensemble Calibration")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_stacked_calibration_2018_all_feature_sets.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_stacked_calibration_2018_all_feature_sets.pdf")
    plt.close()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "stacked_calibration_all_feature_sets.csv", index=False)
    return out


def write_report(shap_tables: list[pd.DataFrame], coef: pd.DataFrame, cal: pd.DataFrame) -> None:
    shap_all = pd.concat(shap_tables, ignore_index=True)
    shap_all.to_csv(OUT_DIR / "stacked_meta_shap_importance_all_feature_sets.csv", index=False)
    lines = [
        "# Stacked Ensemble Interpretation For Three Feature Sets",
        "",
        "Public labels used in figures and reports:",
        "",
        "- Conventional",
        "- AlphaEarth Embeddings",
        "- Conventional + AlphaEarth Embeddings",
        "",
        "SHAP is computed at the stacked meta-learner level. The SHAP variables are the base-model probability outputs, not the original terrain/rainfall/embedding predictors.",
        "",
        "## Top Meta-SHAP Base Models",
        "",
        "| Feature set | Rank | Base model | Mean |SHAP| |",
        "|---|---:|---|---:|",
    ]
    for feature_set, label in FEATURE_SET_LABELS.items():
        d = shap_all[shap_all["feature_set"] == label].sort_values("mean_abs_shap", ascending=False).head(6)
        for rank, (_, row) in enumerate(d.iterrows(), start=1):
            lines.append(f"| {label} | {rank} | {row['base_model']} | {row['mean_abs_shap']:.5f} |")
    lines.extend(
        [
            "",
            "## Saved Figures",
            "",
        ]
    )
    for feature_set, label in FEATURE_SET_LABELS.items():
        fl = FILE_LABELS[feature_set]
        lines.append(f"- SHAP importance, {label}: `{FIG_DIR / f'figure_stacked_shap_importance_2018_{fl}.png'}`")
        lines.append(f"- SHAP beeswarm, {label}: `{FIG_DIR / f'figure_stacked_shap_beeswarm_2018_{fl}.png'}`")
        lines.append(f"- Meta-coefficients, {label}: `{FIG_DIR / f'figure_stacked_meta_coefficients_2018_{fl}.png'}`")
    lines.append(f"- Calibration curve: `{FIG_DIR / 'figure_stacked_calibration_2018_all_feature_sets.png'}`")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED)
    shap_tables = []
    for feature_set in FEATURE_SET_LABELS:
        x = meta_matrix(pred, feature_set)
        shap_tables.append(meta_shap(feature_set, x))
    coef = meta_coefficients()
    cal = calibration(pred)
    write_report(shap_tables, coef, cal)
    print(REPORT)


if __name__ == "__main__":
    main()
