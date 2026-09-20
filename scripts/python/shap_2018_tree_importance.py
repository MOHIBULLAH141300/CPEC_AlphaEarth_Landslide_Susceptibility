"""Create SHAP tree variable-importance outputs for 2018 XGBoost models."""

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
OUT_DIR = PROJECT_ROOT / "03_models" / "shap_2018_tree_importance"
REPORT = PROJECT_ROOT / "05_reports" / "shap_2018_tree_variable_importance.md"

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

FACTOR_NAMES = {
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "aspect_deg": "Aspect",
    "rain_monsoon_total": "Monsoon Rainfall",
    "rain_max_1day": "Maximum 1-Day Rainfall",
    "ndvi_median": "NDVI Median",
    "ndvi_amplitude": "NDVI Amplitude",
    "lst_day_mean_c": "Mean Daytime LST",
    "modis_lc_type1": "MODIS Land Cover",
}

FEATURE_SETS = {
    "conventional_reduced": {
        "display_name": "Conventional",
        "output_key": "conventional",
        "model": MODEL_DIR / "model_conventional_reduced_xgboost.joblib",
        "features": CONVENTIONAL,
    },
    "alphaearth_only": {
        "display_name": "AlphaEarth Embeddings",
        "output_key": "alphaearth_embeddings",
        "model": MODEL_DIR / "model_alphaearth_only_xgboost.joblib",
        "features": [f"A{i:02d}" for i in range(64)],
    },
    "fused_reduced": {
        "display_name": "Conventional + AlphaEarth Embeddings",
        "output_key": "conventional_alphaearth_embeddings",
        "model": MODEL_DIR / "model_fused_reduced_xgboost.joblib",
        "features": CONVENTIONAL + [f"A{i:02d}" for i in range(64)],
    },
}


def feature_name(feature: str) -> str:
    if feature in FACTOR_NAMES:
        return FACTOR_NAMES[feature]
    if feature.startswith("A") and feature[1:].isdigit():
        return f"AlphaEarth {feature}"
    return feature


def shap_values_for_model(model, x: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]
    return np.asarray(values)


def save_bar_plot(importance: pd.DataFrame, title: str, path: Path, top_n: int = 25) -> None:
    top = importance.head(top_n).sort_values("mean_abs_shap")
    plt.figure(figsize=(8.4, max(5.2, 0.28 * len(top))))
    plt.barh(top["feature_name"], top["mean_abs_shap"], color="#2563eb")
    plt.xlabel("Mean absolute SHAP value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_beeswarm(values: np.ndarray, x: pd.DataFrame, display_name: str, path: Path, max_display: int = 25) -> None:
    x_named = x.copy()
    x_named.columns = [feature_name(c) for c in x_named.columns]
    shap.summary_plot(values, x_named, max_display=max_display, show=False)
    plt.title(f"{display_name} - SHAP Summary")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    for col in set().union(*(set(cfg["features"]) for cfg in FEATURE_SETS.values())):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report_lines = [
        "# 2018 SHAP Tree Variable Importance",
        "",
        "SHAP values were computed for the final XGBoost models fitted on all 2018 training samples.",
        "",
        "| Feature set | Top variable | Top mean abs SHAP | CSV | Plot |",
        "|---|---|---:|---|---|",
    ]
    combined = []

    for key, cfg in FEATURE_SETS.items():
        model = joblib.load(cfg["model"])
        x = df[cfg["features"]].copy()
        values = shap_values_for_model(model, x)
        imp = pd.DataFrame(
            {
                "feature": cfg["features"],
                "feature_name": [feature_name(f) for f in cfg["features"]],
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
                "std_abs_shap": np.abs(values).std(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        imp["feature_set"] = key
        imp["feature_set_name"] = cfg["display_name"]
        output_key = cfg["output_key"]
        csv_path = OUT_DIR / f"shap_importance_{output_key}_xgboost.csv"
        bar_path = OUT_DIR / f"shap_importance_{output_key}_xgboost_bar.png"
        beeswarm_path = OUT_DIR / f"shap_importance_{output_key}_xgboost_beeswarm.png"
        imp.to_csv(csv_path, index=False)
        save_bar_plot(imp, f"{cfg['display_name']} - XGBoost SHAP Importance", bar_path)
        save_beeswarm(values, x, cfg["display_name"], beeswarm_path)
        combined.append(imp)
        top = imp.iloc[0]
        report_lines.append(
            f"| {cfg['display_name']} | {top['feature_name']} | {top['mean_abs_shap']:.6f} | "
            f"`{csv_path}` | `{bar_path}` |"
        )

    combined_df = pd.concat(combined, ignore_index=True)
    combined_path = OUT_DIR / "shap_importance_all_xgboost.csv"
    combined_df.to_csv(combined_path, index=False)
    report_lines.extend(
        [
            "",
            "## Saved Outputs",
            "",
            f"- Combined SHAP table: `{combined_path}`",
            f"- Output folder: `{OUT_DIR}`",
            "",
            "## Note",
            "",
            "For AlphaEarth embedding bands, feature names are reported as AlphaEarth A00, AlphaEarth A01, etc. These bands are latent representation dimensions, not direct physical variables.",
        ]
    )
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(combined_df.groupby("feature_set_name").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
