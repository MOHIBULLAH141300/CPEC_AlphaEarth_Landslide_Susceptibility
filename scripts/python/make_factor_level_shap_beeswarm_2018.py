"""Create factor-level TreeSHAP figures for the three official 2018 feature sets.

These figures explain the XGBoost base learner for each feature set. The stacked
ensemble is explained separately at the meta-learner level because its direct
inputs are base-model probabilities, not original conditioning factors.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
CONVENTIONAL_LIST = (
    PROJECT_ROOT
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)
OUT_DIR = PROJECT_ROOT / "03_models" / "factor_level_shap_2018"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "factor_level_shap_2018.md"

ALPHA = [f"A{i:02d}" for i in range(64)]

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

FEATURE_SETS = {
    "conventional": {
        "internal": "conventional_v3",
        "label": "Conventional",
        "features": None,
    },
    "alphaearth_embeddings": {
        "internal": "alphaearth_embeddings",
        "label": "AlphaEarth Embeddings",
        "features": ALPHA,
    },
    "conventional_alphaearth_embeddings": {
        "internal": "conventional_v3_alphaearth_embeddings",
        "label": "Conventional + AlphaEarth Embeddings",
        "features": None,
    },
}


def load_feature_sets() -> dict[str, dict[str, object]]:
    conventional = pd.read_csv(CONVENTIONAL_LIST)["factor"].tolist()
    FEATURE_SETS["conventional"]["features"] = conventional
    FEATURE_SETS["conventional_alphaearth_embeddings"]["features"] = conventional + ALPHA
    return FEATURE_SETS


def shap_for_feature_set(df: pd.DataFrame, key: str, cfg: dict[str, object]) -> pd.DataFrame:
    internal = str(cfg["internal"])
    label = str(cfg["label"])
    features = list(cfg["features"])
    model_path = MODEL_DIR / f"model_{internal}_xgboost.joblib"
    model = joblib.load(model_path)
    imputer = model.named_steps["imputer"]
    fitted = model.named_steps["model"]

    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_imp = pd.DataFrame(imputer.transform(x), columns=features)
    sample = shap.sample(x_imp, min(1600, len(x_imp)), random_state=141300)
    background = shap.sample(x_imp, min(500, len(x_imp)), random_state=141300)

    explainer = shap.TreeExplainer(fitted, data=background, feature_perturbation="interventional")
    values = np.asarray(explainer.shap_values(sample))
    display_names = [FACTOR_LABELS.get(f, f) for f in features]

    importance = pd.DataFrame(
        {
            "feature_set": label,
            "feature": features,
            "feature_name": display_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT_DIR / f"factor_shap_importance_2018_{key}.csv", index=False)

    max_display = 20
    plt.figure(figsize=(8.4, 6.8))
    explanation = shap.Explanation(
        values=values,
        data=sample.to_numpy(),
        feature_names=display_names,
    )
    shap.plots.beeswarm(
        explanation,
        show=False,
        max_display=max_display,
        group_remaining_features=True,
        color_bar_label="Feature value",
    )
    plt.title(f"Factor-Level SHAP Beeswarm: {label}", pad=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"figure_factor_shap_beeswarm_2018_{key}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"figure_factor_shap_beeswarm_2018_{key}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.2, 6.4))
    top = importance.head(max_display).iloc[::-1]
    plt.barh(top["feature_name"], top["mean_abs_shap"], color="#2f6f73")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Factor-Level SHAP Importance: {label}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"figure_factor_shap_importance_2018_{key}.png", dpi=300)
    plt.savefig(FIG_DIR / f"figure_factor_shap_importance_2018_{key}.pdf")
    plt.close()
    return importance


def write_report(tables: list[pd.DataFrame]) -> None:
    all_imp = pd.concat(tables, ignore_index=True)
    all_imp.to_csv(OUT_DIR / "factor_shap_importance_2018_all_feature_sets.csv", index=False)
    lines = [
        "# Factor-Level SHAP For 2018 Feature Sets",
        "",
        "These figures explain the XGBoost base learner for each official feature set. They complement the stacked-ensemble meta-SHAP figures: meta-SHAP explains base-model contributions, while these plots explain original conditioning-factor contributions.",
        "",
        "## Top Factors",
        "",
        "| Feature set | Rank | Factor | Mean |SHAP| |",
        "|---|---:|---|---:|",
    ]
    for label in ["Conventional", "AlphaEarth Embeddings", "Conventional + AlphaEarth Embeddings"]:
        d = all_imp[all_imp["feature_set"] == label].head(15)
        for rank, (_, row) in enumerate(d.iterrows(), start=1):
            lines.append(f"| {label} | {rank} | {row['feature_name']} | {row['mean_abs_shap']:.5f} |")
    lines.extend(
        [
            "",
            "## Saved Figures",
            "",
            f"- SHAP beeswarm, Conventional: `{FIG_DIR / 'figure_factor_shap_beeswarm_2018_conventional.png'}`",
            f"- SHAP beeswarm, AlphaEarth Embeddings: `{FIG_DIR / 'figure_factor_shap_beeswarm_2018_alphaearth_embeddings.png'}`",
            f"- SHAP beeswarm, Conventional + AlphaEarth Embeddings: `{FIG_DIR / 'figure_factor_shap_beeswarm_2018_conventional_alphaearth_embeddings.png'}`",
            f"- SHAP importance, Conventional: `{FIG_DIR / 'figure_factor_shap_importance_2018_conventional.png'}`",
            f"- SHAP importance, AlphaEarth Embeddings: `{FIG_DIR / 'figure_factor_shap_importance_2018_alphaearth_embeddings.png'}`",
            f"- SHAP importance, Conventional + AlphaEarth Embeddings: `{FIG_DIR / 'figure_factor_shap_importance_2018_conventional_alphaearth_embeddings.png'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    tables = []
    for key, cfg in load_feature_sets().items():
        tables.append(shap_for_feature_set(df, key, cfg))
    write_report(tables)
    print(REPORT)


if __name__ == "__main__":
    main()
