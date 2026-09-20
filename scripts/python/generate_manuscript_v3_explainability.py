"""Correctly attribute TreeSHAP to the XGBoost base learners in manuscript v3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import PROJECT_ROOT, load_data  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv" / "models"
OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_xgboost_treeshap"

DISPLAY = {
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "rain_monsoon_total": "Monsoon rainfall",
    "rain_max_1day": "Maximum 1-day rainfall",
    "ndvi_median": "Median NDVI",
    "ndvi_amplitude": "NDVI amplitude",
    "log1p_dist_road_m": "Distance to roads",
    "log1p_dist_river_m": "Distance to rivers/streams",
    "log1p_dist_fault_m": "Distance to active faults",
    "profile_curvature": "Profile curvature",
    "plan_curvature": "Plan curvature",
    "tri": "Terrain ruggedness index",
    "twi": "Topographic wetness index",
    "valley_depth": "Valley depth",
    "aspect": "Aspect (cyclic)",
    "modis_lc_type1": "Land-cover class",
    "lithology_code": "Lithology",
    "soil_type": "Soil class",
    "eq_density_ms5": "M>=5 earthquake-density class",
}


def original_group(transformed_name: str, categorical: list[str]) -> str:
    name = transformed_name.split("__", 1)[-1]
    if name in {"aspect_sin", "aspect_cos"}:
        return "aspect"
    for factor in categorical:
        if name == factor or name.startswith(f"{factor}_"):
            return factor
    return name


def display_name(group: str) -> str:
    if group.startswith("A") and len(group) == 3 and group[1:].isdigit():
        return f"AlphaEarth {group}"
    return DISPLAY.get(group, group.replace("_", " ").title())


def grouped_feature_value(df: pd.DataFrame, group: str) -> np.ndarray:
    if group == "aspect":
        return pd.to_numeric(df["aspect_deg"], errors="coerce").to_numpy(float)
    return pd.to_numeric(df[group], errors="coerce").to_numpy(float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    importance_rows = []
    manifests = {}
    for key, spec in specs.items():
        bundle = joblib.load(MODEL_DIR / f"nested_spatial_stack_{key}.joblib")
        pipeline = bundle["base_models"]["xgboost"]
        preprocess = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        transformed = preprocess.transform(df[spec.columns])
        transformed_names = preprocess.get_feature_names_out()
        contributions = model.get_booster().predict(
            xgb.DMatrix(transformed, feature_names=None), pred_contribs=True
        )
        bias = contributions[:, -1]
        shap_values = contributions[:, :-1]
        groups = [original_group(name, spec.categorical) for name in transformed_names]
        unique_groups = list(dict.fromkeys(groups))
        grouped_shap = np.column_stack(
            [shap_values[:, np.asarray(groups) == group].sum(axis=1) for group in unique_groups]
        )
        grouped_values = np.column_stack(
            [grouped_feature_value(df, group) for group in unique_groups]
        )
        shap_table = pd.DataFrame(grouped_shap, columns=unique_groups)
        shap_table.insert(0, "sample_id", np.arange(len(df)))
        value_table = pd.DataFrame(grouped_values, columns=unique_groups)
        value_table.insert(0, "sample_id", np.arange(len(df)))
        shap_table.to_parquet(OUT_DIR / f"xgboost_treeshap_values_{key}.parquet", index=False)
        value_table.to_parquet(OUT_DIR / f"xgboost_feature_values_{key}.parquet", index=False)
        for group, values in zip(unique_groups, grouped_shap.T):
            importance_rows.append(
                {
                    "feature_set": key,
                    "feature_set_name": spec.display_name,
                    "explained_model": "XGBoost base learner",
                    "factor": group,
                    "factor_name": display_name(group),
                    "mean_absolute_treeshap": float(np.mean(np.abs(values))),
                    "mean_signed_treeshap": float(np.mean(values)),
                }
            )
        manifests[key] = {
            "n_samples": len(df),
            "n_transformed_features": len(transformed_names),
            "n_grouped_factors": len(unique_groups),
            "mean_bias_margin": float(np.mean(bias)),
            "method": "exact XGBoost TreeSHAP contributions in model-margin units",
            "categorical_aggregation": "one-hot contributions summed to source predictor",
            "aspect_aggregation": "sine and cosine contributions summed as cyclic aspect",
        }
        print(f"{spec.display_name}: {len(unique_groups)} grouped factors", flush=True)
    importance = pd.DataFrame(importance_rows).sort_values(
        ["feature_set", "mean_absolute_treeshap"], ascending=[True, False]
    )
    importance.to_csv(OUT_DIR / "xgboost_treeshap_global_importance.csv", index=False)
    manifest = {
        "analysis": "manuscript_v3_xgboost_base_learner_treeshap",
        "interpretation_limit": "TreeSHAP explains the XGBoost base learner, not the heterogeneous stacked ensemble",
        "stack_explanation": "The stacked ensemble is described separately using spatial-fold meta-learner coefficients",
        "feature_sets": manifests,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(importance.groupby("feature_set").head(10).to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
