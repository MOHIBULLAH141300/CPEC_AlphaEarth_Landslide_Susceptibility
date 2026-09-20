"""Create TreeSHAP article figures for the 2018 XGBoost LSM results."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
MODEL_DIR = PROJECT_ROOT / "03_models" / "reduced_strategy_2018_spatial_cv"
OUT_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "treeshap_article_figures_2018.md"

CONVENTIONAL = [
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
ALPHA = [f"A{i:02d}" for i in range(64)]

FEATURE_SETS = {
    "conventional": {
        "model_path": MODEL_DIR / "model_conventional_reduced_xgboost.joblib",
        "display": "Conventional",
        "features": CONVENTIONAL,
    },
    "alphaearth_embeddings": {
        "model_path": MODEL_DIR / "model_alphaearth_only_xgboost.joblib",
        "display": "AlphaEarth Embeddings",
        "features": ALPHA,
    },
    "conventional_alphaearth_embeddings": {
        "model_path": MODEL_DIR / "model_fused_reduced_xgboost.joblib",
        "display": "Conventional + AlphaEarth Embeddings",
        "features": CONVENTIONAL + ALPHA,
    },
}

NAMES = {
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "aspect_deg": "Aspect",
    "rain_monsoon_total": "Monsoon rainfall",
    "rain_max_1day": "Max 1-day rainfall",
    "ndvi_median": "NDVI median",
    "ndvi_amplitude": "NDVI amplitude",
    "lst_day_mean_c": "Mean daytime LST",
    "modis_lc_type1": "MODIS land cover",
}


def display_name(feature: str) -> str:
    if feature in NAMES:
        return NAMES[feature]
    if feature.startswith("A") and feature[1:].isdigit():
        return f"AlphaEarth {feature}"
    return feature


def load_named_data(features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    x = df[features].copy()
    x.columns = [display_name(c) for c in features]
    return x


def treeshap(model, x: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    values = explainer(x)
    return values


def save_beeswarm(key: str, cfg: dict) -> Path:
    model = joblib.load(cfg["model_path"])
    x = load_named_data(cfg["features"])
    values = treeshap(model, x)

    plt.figure(figsize=(8.4, 7.2))
    shap.plots.beeswarm(values, max_display=20, show=False, color_bar=True)
    plt.title(f"TreeSHAP Summary: {cfg['display']}", fontsize=13, fontweight="bold")
    plt.xlabel("SHAP value: impact on XGBoost model output", fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / f"figure_treeshap_beeswarm_2018_{key}.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()
    return out


def save_waterfall_for_fused() -> tuple[Path, int, float]:
    cfg = FEATURE_SETS["conventional_alphaearth_embeddings"]
    model = joblib.load(cfg["model_path"])
    x = load_named_data(cfg["features"])
    values = treeshap(model, x)
    prob = model.predict_proba(pd.read_csv(DATA, encoding="utf-8-sig")[cfg["features"]])[:, 1]
    idx = int(np.argsort(prob)[int(0.95 * len(prob))])

    plt.figure(figsize=(8.2, 7.4))
    shap.plots.waterfall(values[idx], max_display=15, show=False)
    plt.title("TreeSHAP Waterfall: High-Susceptibility Example\nConventional + AlphaEarth Embeddings", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "figure_treeshap_waterfall_2018_fused_high_susceptibility.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()
    return out, idx, float(prob[idx])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beeswarm_paths = []
    for key, cfg in FEATURE_SETS.items():
        beeswarm_paths.append((cfg["display"], save_beeswarm(key, cfg)))
    waterfall_path, idx, probability = save_waterfall_for_fused()

    lines = [
        "# TreeSHAP Article Figures 2018",
        "",
        "TreeSHAP figures were generated for the final 2018 XGBoost landslide susceptibility models.",
        "",
        "## Figures",
        "",
    ]
    for label, path in beeswarm_paths:
        lines.append(f"- {label} beeswarm summary: `{path}`")
    lines.append(f"- Fused-model high-susceptibility waterfall: `{waterfall_path}`")
    lines.extend(
        [
            "",
            "## Selected Waterfall Case",
            "",
            f"- Sample row index: {idx}",
            f"- Predicted susceptibility probability: {probability:.4f}",
            "",
            "## Suggested Method Text",
            "",
            "TreeSHAP was used to interpret the final XGBoost susceptibility models. TreeSHAP computes exact Shapley values for tree ensembles by exploiting the decision-path structure of trees, providing local feature contributions that sum with the expected model output to recover each prediction. Global variable importance was calculated as the mean absolute SHAP value across all samples.",
            "",
            "## Suggested Caption",
            "",
            "TreeSHAP interpretation of the 2018 XGBoost landslide susceptibility models. Beeswarm plots show both the magnitude and direction of predictor effects, where each point represents one sample and color indicates the predictor value. The waterfall plot illustrates how the most influential predictors shifted the model output for a representative high-susceptibility case.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(str(path) for _, path in beeswarm_paths))
    print(waterfall_path)


if __name__ == "__main__":
    main()
