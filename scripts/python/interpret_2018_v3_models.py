"""Interpret v3 final candidate models with SHAP and calibration."""

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
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
OUT_DIR = PROJECT_ROOT / "03_models" / "v3_interpretation_2018"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "v3_shap_calibration_interpretation_2018.md"
RECOMMENDATION = PROJECT_ROOT / "05_reports" / "v3_final_map_model_recommendation_2026-05-03.md"

CONVENTIONAL = pd.read_csv(PROJECT_ROOT / "03_models" / "multicollinearity_assessment_2018_v3" / "v3_final_selected_conventional_factors.csv")[
    "factor"
].tolist()
ALPHA = [f"A{i:02d}" for i in range(64)]
FUSED = CONVENTIONAL + ALPHA

FACTOR_LABELS = {
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "aspect_deg": "Aspect",
    "rain_monsoon_total": "Monsoon rainfall",
    "rain_max_1day": "Max 1-day rainfall",
    "ndvi_median": "NDVI median",
    "ndvi_amplitude": "NDVI amplitude",
    "modis_lc_type1": "MODIS land cover",
    "log1p_dist_road_m": "Distance to roads",
    "log1p_dist_river_m": "Distance to rivers/streams",
    "log1p_dist_fault_m": "Distance to active faults",
    "lithology_code": "Lithology/geology",
    "profile_curvature": "Profile curvature",
    "plan_curvature": "Plan curvature",
    "tri": "Terrain ruggedness",
    "twi": "TWI",
    "valley_depth": "Valley depth",
    "soil_type": "Soil type",
    "eq_density_ms5": "Earthquake density > Ms5",
}
for band in ALPHA:
    FACTOR_LABELS[band] = f"AlphaEarth {band}"


def load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    for col in FUSED + ["label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    x = df[FUSED].copy()
    y = df["label"].astype(int).to_numpy()
    return x, y


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def shap_for_xgboost(x: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / "model_conventional_v3_alphaearth_embeddings_xgboost.joblib")
    imputer = model.named_steps["imputer"]
    fitted = model.named_steps["model"]
    x_imp = pd.DataFrame(imputer.transform(x), columns=FUSED)
    background = shap.sample(x_imp, min(500, len(x_imp)), random_state=141300)
    explainer = shap.TreeExplainer(fitted, data=background, feature_perturbation="interventional")
    sample = shap.sample(x_imp, min(1500, len(x_imp)), random_state=141300)
    shap_values = explainer.shap_values(sample)
    vals = np.asarray(shap_values)
    importance = pd.DataFrame(
        {
            "feature": FUSED,
            "feature_name": [FACTOR_LABELS.get(f, f) for f in FUSED],
            "mean_abs_shap": np.abs(vals).mean(axis=0),
            "mean_shap": vals.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT_DIR / "v3_fused_xgboost_shap_importance.csv", index=False)

    plt.figure(figsize=(8.2, 6.4))
    top = importance.head(20).iloc[::-1]
    plt.barh(top["feature_name"], top["mean_abs_shap"], color="#2f6f73")
    plt.xlabel("Mean |SHAP value|")
    plt.title("v3 Fused XGBoost TreeSHAP Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_fused_xgboost_shap_importance.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_v3_fused_xgboost_shap_importance.pdf")
    plt.close()

    plt.figure(figsize=(8.5, 7.0))
    shap.summary_plot(vals, sample.rename(columns=FACTOR_LABELS), show=False, max_display=20)
    plt.title("v3 Fused XGBoost TreeSHAP Summary", pad=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_fused_xgboost_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure_v3_fused_xgboost_shap_beeswarm.pdf", bbox_inches="tight")
    plt.close()
    return importance


def calibration_analysis(y: np.ndarray) -> pd.DataFrame:
    pred = pd.read_csv(MODEL_DIR / "v3_model_family_spatial_cv_predictions.csv")
    keep = [
        ("conventional_v3_alphaearth_embeddings", "xgboost", "Fused v3 XGBoost"),
        ("conventional_v3_alphaearth_embeddings", "catboost", "Fused v3 CatBoost"),
        ("conventional_v3_alphaearth_embeddings", "stacked_l2_logistic", "Fused v3 Stacked Ensemble"),
    ]
    rows = []
    plt.figure(figsize=(7.2, 6.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Perfect calibration")
    for fs, model, label in keep:
        d = pred[(pred["feature_set"] == fs) & (pred["model"] == model)].copy()
        yy = d["label"].astype(int).to_numpy()
        pp = d["probability"].astype(float).to_numpy()
        frac_pos, mean_pred = calibration_curve(yy, pp, n_bins=10, strategy="quantile")
        brier = brier_score_loss(yy, pp)
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=f"{label} (Brier={brier:.3f})")
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
            rows.append(
                {
                    "feature_set": fs,
                    "model": model,
                    "label": label,
                    "bin": i,
                    "mean_predicted_probability": mp,
                    "fraction_observed_landslide": fp,
                    "brier": brier,
                    "n": len(d),
                }
            )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed landslide fraction")
    plt.title("v3 Fused Model Calibration")
    plt.legend(frameon=False, loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_fused_model_calibration.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_v3_fused_model_calibration.pdf")
    plt.close()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v3_fused_model_calibration_curve.csv", index=False)
    return out


def meta_coefficients() -> pd.DataFrame:
    weights = pd.read_csv(MODEL_DIR / "v3_model_family_meta_coefficients.csv")
    d = weights[weights["feature_set_name"] == "Conventional v3 + AlphaEarth Embeddings"].copy()
    summary = (
        d.groupby(["base_model", "base_model_name"])
        .agg(
            meta_coefficient_mean=("meta_coefficient", "mean"),
            meta_coefficient_std=("meta_coefficient", "std"),
            meta_coefficient_min=("meta_coefficient", "min"),
            meta_coefficient_max=("meta_coefficient", "max"),
        )
        .reset_index()
        .sort_values("meta_coefficient_mean", ascending=False)
    )
    summary.to_csv(OUT_DIR / "v3_fused_stacked_meta_coefficient_summary.csv", index=False)
    plt.figure(figsize=(7.2, 4.6))
    plt.barh(summary["base_model_name"].iloc[::-1], summary["meta_coefficient_mean"].iloc[::-1], color="#725a9c")
    plt.xlabel("Mean meta-learner coefficient")
    plt.title("v3 Fused Stacked Ensemble Meta-Coefficients")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_fused_stacked_meta_coefficients.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_v3_fused_stacked_meta_coefficients.pdf")
    plt.close()
    return summary


def write_reports(shap_imp: pd.DataFrame, cal: pd.DataFrame, coef: pd.DataFrame) -> None:
    top = shap_imp.head(15)
    brier = cal.groupby("label")["brier"].first().sort_values()
    lines = [
        "# v3 SHAP And Calibration Interpretation",
        "",
        "## Literature/Methods Gate",
        "",
        "After model evaluation, the next standard step is explainability and calibration before final raster export. TreeSHAP supports interpretation of the best tree model, and calibration/Brier score checks whether probabilities are reliable enough for map use.",
        "",
        "## Top v3 Fused XGBoost SHAP Variables",
        "",
        "| Rank | Feature | Mean |SHAP| | Mean SHAP |",
        "|---:|---|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(f"| {rank} | {row['feature_name']} | {row['mean_abs_shap']:.5f} | {row['mean_shap']:.5f} |")
    lines.extend(
        [
            "",
            "## Calibration Ranking By Brier Score",
            "",
            "| Model | Brier score |",
            "|---|---:|",
        ]
    )
    for label, val in brier.items():
        lines.append(f"| {label} | {val:.5f} |")
    lines.extend(
        [
            "",
            "## Stacked Ensemble Meta-Coefficients",
            "",
            "| Base model | Mean coefficient |",
            "|---|---:|",
        ]
    )
    for _, row in coef.iterrows():
        lines.append(f"| {row['base_model_name']} | {row['meta_coefficient_mean']:.4f} |")
    lines.extend(
        [
            "",
            "## Saved Figures",
            "",
            f"- SHAP importance: `{FIG_DIR / 'figure_v3_fused_xgboost_shap_importance.png'}`",
            f"- SHAP beeswarm: `{FIG_DIR / 'figure_v3_fused_xgboost_shap_beeswarm.png'}`",
            f"- Calibration: `{FIG_DIR / 'figure_v3_fused_model_calibration.png'}`",
            f"- Meta-coefficients: `{FIG_DIR / 'figure_v3_fused_stacked_meta_coefficients.png'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    RECOMMENDATION.write_text(
        "\n".join(
            [
                "# v3 Final Map-Model Recommendation",
                "",
                "## Recommendation",
                "",
                "Use `Conventional v3 + AlphaEarth Embeddings / XGBoost` as the primary susceptibility ranking map model, and use the `Spatial-CV Stacked Ensemble` as calibration and robustness support.",
                "",
                "## Reason",
                "",
                "- XGBoost has the best ROC-AUC and PR-AUC/AP for the fused v3 feature set.",
                "- The stacked ensemble has the best Brier score and threshold-based metrics, so it supports probability calibration and robustness discussion.",
                "- This two-part interpretation is defensible: ranking maps prioritize XGBoost, while calibrated probability/uncertainty discussion can reference the stacked ensemble.",
                "",
                "## Next Step",
                "",
                "Prepare the final v3 XGBoost probability raster, then run raster QA for value range, internal NoData, extent, and resolution.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_xy()
    shap_imp = shap_for_xgboost(x)
    cal = calibration_analysis(y)
    coef = meta_coefficients()
    write_reports(shap_imp, cal, coef)
    print(REPORT)
    print(RECOMMENDATION)


if __name__ == "__main__":
    main()
