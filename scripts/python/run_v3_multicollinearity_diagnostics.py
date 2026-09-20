"""Run v3 multicollinearity diagnostics before re-modelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "multicollinearity_assessment_2018_v3"
REPORT = PROJECT_ROOT / "05_reports" / "multicollinearity_assessment_2018_v3.md"

FACTOR_NAMES = {
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "aspect_deg": "Aspect",
    "rain_annual_total": "Annual rainfall total",
    "rain_monsoon_total": "Monsoon rainfall total",
    "rain_max_1day": "Maximum 1-day rainfall",
    "rain_max_3day": "Maximum 3-day rainfall",
    "rain_max_7day": "Maximum 7-day rainfall",
    "ndvi_median": "NDVI median",
    "ndvi_max": "NDVI maximum",
    "ndvi_amplitude": "NDVI amplitude",
    "evi_median": "EVI median",
    "lst_day_mean_c": "Mean daytime LST",
    "lst_day_max_c": "Maximum daytime LST",
    "modis_lc_type1": "MODIS land cover",
    "dist_road_m": "Distance to roads",
    "log1p_dist_road_m": "Distance to roads",
    "dist_river_m": "Distance to rivers/streams",
    "log1p_dist_river_m": "Distance to rivers/streams",
    "dist_fault_m": "Distance to active faults",
    "log1p_dist_fault_m": "Distance to active faults",
    "lithology_code": "Lithology/geology class",
    "profile_curvature": "Profile curvature",
    "plan_curvature": "Plan curvature",
    "tri": "Terrain ruggedness index",
    "twi": "Topographic wetness index",
    "valley_depth": "Valley depth",
    "relief": "Relief",
    "ls_factor": "LS factor",
    "soil_type": "Soil type",
    "eq_density_ms5": "Earthquake density > Ms5",
    "population_density": "Population density",
    "night_lights": "Night lights",
    "dist_river_raster_m": "Legacy river-distance raster",
}

V3_CANDIDATES = [
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
    "dist_road_m",
    "log1p_dist_road_m",
    "dist_river_m",
    "log1p_dist_river_m",
    "dist_fault_m",
    "log1p_dist_fault_m",
    "lithology_code",
    "profile_curvature",
    "plan_curvature",
    "tri",
    "twi",
    "valley_depth",
    "relief",
    "ls_factor",
    "soil_type",
    "eq_density_ms5",
    "population_density",
    "night_lights",
    "dist_river_raster_m",
]

INITIAL_EXCLUDE = {
    "night_lights": "Excluded before modelling because local samples are 100% missing.",
    "dist_river_raster_m": "Excluded because 60% of samples are missing and vector-derived distance to rivers is complete.",
    "population_density": "Excluded from susceptibility modelling because it is better treated as exposure/vulnerability, not a physical instability control.",
}

PREFER_DROP = {
    "dist_road_m": "Dropped in favor of log1p_dist_road_m to reduce extreme skew and avoid duplicate distance encoding.",
    "dist_river_m": "Dropped in favor of log1p_dist_river_m to reduce extreme skew and avoid duplicate distance encoding.",
    "dist_fault_m": "Dropped in favor of log1p_dist_fault_m to reduce extreme skew and avoid duplicate distance encoding.",
    "rain_annual_total": "Dropped if redundant with monsoon/extreme rainfall metrics; dynamic rainfall literature favors event/seasonal windows.",
    "rain_max_3day": "Dropped if highly collinear with 1-day/7-day rainfall.",
    "rain_max_7day": "Dropped if highly collinear with 1-day/3-day rainfall.",
    "ndvi_max": "Dropped if redundant with NDVI median/amplitude.",
    "evi_median": "Dropped if redundant with NDVI median.",
    "lst_day_max_c": "Dropped if redundant with mean daytime LST.",
    "ls_factor": "Dropped if too redundant with slope/relief/TWI terrain morphology.",
}

SEED_KEEP = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
    "log1p_dist_road_m",
    "log1p_dist_river_m",
    "log1p_dist_fault_m",
    "lithology_code",
    "profile_curvature",
    "plan_curvature",
    "tri",
    "twi",
    "valley_depth",
    "relief",
    "soil_type",
    "eq_density_ms5",
]


def numeric_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=cols)


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = numeric_table(df, cols)
    rows = []
    for col in cols:
        other = [c for c in cols if c != col]
        model = LinearRegression()
        model.fit(x[other], x[col])
        r2 = float(model.score(x[other], x[col]))
        rows.append(
            {
                "factor": col,
                "factor_name": FACTOR_NAMES.get(col, col),
                "r2_against_other_factors": r2,
                "vif": np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def high_corr_table(df: pd.DataFrame, cols: list[str], threshold: float = 0.85) -> pd.DataFrame:
    x = numeric_table(df, cols)
    corr = x.corr(method="spearman")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = float(corr.loc[a, b])
            if abs(val) >= threshold:
                rows.append(
                    {
                        "factor_a": a,
                        "factor_a_name": FACTOR_NAMES.get(a, a),
                        "factor_b": b,
                        "factor_b_name": FACTOR_NAMES.get(b, b),
                        "spearman_r": val,
                    }
                )
    return pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)


def decision_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[V3_CANDIDATES].apply(lambda s: pd.to_numeric(s, errors="coerce").isna().mean() * 100)
    rows = []
    keep_set = set(SEED_KEEP)
    for col in V3_CANDIDATES:
        if col in INITIAL_EXCLUDE:
            decision = "exclude"
            reason = INITIAL_EXCLUDE[col]
        elif col in keep_set:
            decision = "candidate_keep"
            reason = "Literature-supported v3 candidate; final retention depends on VIF/correlation and model evidence."
        elif col in PREFER_DROP:
            decision = "candidate_drop"
            reason = PREFER_DROP[col]
        else:
            decision = "candidate_review"
            reason = "Review after multicollinearity and model evidence."
        rows.append(
            {
                "factor": col,
                "factor_name": FACTOR_NAMES.get(col, col),
                "missing_percent": missing[col],
                "initial_decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    candidates = [c for c in V3_CANDIDATES if c in df.columns]
    analysis_cols = [c for c in candidates if c not in INITIAL_EXCLUDE]
    selected_cols = [c for c in SEED_KEEP if c in df.columns]

    corr = numeric_table(df, analysis_cols).corr(method="spearman")
    high_corr = high_corr_table(df, analysis_cols)
    vif_all = vif_table(df, analysis_cols)
    vif_selected = vif_table(df, selected_cols)
    decisions = decision_table(df)

    corr.to_csv(OUT_DIR / "v3_spearman_correlation_matrix.csv", encoding="utf-8-sig")
    high_corr.to_csv(OUT_DIR / "v3_high_correlation_pairs_abs_ge_0_85.csv", index=False, encoding="utf-8-sig")
    vif_all.to_csv(OUT_DIR / "v3_candidate_vif.csv", index=False, encoding="utf-8-sig")
    vif_selected.to_csv(OUT_DIR / "v3_seed_selected_vif.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUT_DIR / "v3_initial_keep_drop_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": selected_cols, "factor_name": [FACTOR_NAMES.get(c, c) for c in selected_cols]}).to_csv(
        OUT_DIR / "v3_seed_selected_factor_list.csv", index=False, encoding="utf-8-sig"
    )

    lines = [
        "# 2018 v3 Multicollinearity Assessment",
        "",
        "## Literature Gate",
        "",
        "The v3 diagnostic set adds standard landslide-conditioning factors supported by recent literature: road proximity, river/stream proximity, fault proximity, lithology, terrain morphology, TWI, seismic background, vegetation, rainfall, thermal, land-cover, and AlphaEarth embedding branches.",
        "",
        "## Pre-Modelling Exclusions",
        "",
    ]
    for factor, reason in INITIAL_EXCLUDE.items():
        lines.append(f"- {FACTOR_NAMES.get(factor, factor)} (`{factor}`): {reason}")
    lines.extend(
        [
            "",
            "## Seed Selected Conventional v3 Factors",
            "",
        ]
    )
    for factor in selected_cols:
        lines.append(f"- {FACTOR_NAMES.get(factor, factor)} (`{factor}`)")
    lines.extend(
        [
            "",
            "## Top High-Correlation Pairs",
            "",
            "| Factor A | Factor B | Spearman r |",
            "|---|---|---:|",
        ]
    )
    for _, row in high_corr.head(20).iterrows():
        lines.append(f"| {row['factor_a_name']} | {row['factor_b_name']} | {row['spearman_r']:.3f} |")
    lines.extend(
        [
            "",
            "## Seed Selected VIF",
            "",
            "| Factor | VIF |",
            "|---|---:|",
        ]
    )
    for _, row in vif_selected.iterrows():
        lines.append(f"| {row['factor_name']} | {row['vif']:.2f} |")
    lines.extend(
        [
            "",
            "## Saved Tables",
            "",
            f"- Correlation matrix: `{OUT_DIR / 'v3_spearman_correlation_matrix.csv'}`",
            f"- High-correlation pairs: `{OUT_DIR / 'v3_high_correlation_pairs_abs_ge_0_85.csv'}`",
            f"- Candidate VIF: `{OUT_DIR / 'v3_candidate_vif.csv'}`",
            f"- Seed selected VIF: `{OUT_DIR / 'v3_seed_selected_vif.csv'}`",
            f"- Initial decisions: `{OUT_DIR / 'v3_initial_keep_drop_decisions.csv'}`",
            f"- Seed selected factor list: `{OUT_DIR / 'v3_seed_selected_factor_list.csv'}`",
            "",
            "## Next Step",
            "",
            "Review high-VIF factors and finalize the v3 conventional feature set before rerunning models. If several terrain factors remain collinear, keep the most physically interpretable and best-performing subset.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(vif_selected.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
