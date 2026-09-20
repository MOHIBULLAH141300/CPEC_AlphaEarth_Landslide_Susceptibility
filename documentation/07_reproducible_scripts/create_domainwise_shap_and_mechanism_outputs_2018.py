"""Create domain-wise TreeSHAP and transferability mechanism outputs for 2018 V3.

The main predictive model remains the CPEC-wide Spatial-CV Stacked Ensemble.
This script explains the XGBoost base learner within each official feature set
because TreeSHAP provides original-factor contributions for tree models.

Outputs are written into:
D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\10_domain_specific_2018_results
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj.datadir
import seaborn as sns
import shap

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
DOMAIN_ROOT = PACKAGE / "10_domain_specific_2018_results"
SUMMARY_ROOT = DOMAIN_ROOT / "00_domain_comparison_summary"
SCRIPT_COPY_DIR = PACKAGE / "07_reproducible_scripts"
DATA = PROJECT / "03_models" / "v3_admin_transferability_domain_tests" / (
    "cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv"
)
MODEL_DIR = PROJECT / "03_models" / "v3_model_family_2018_spatial_cv"
CONVENTIONAL_LIST = (
    PROJECT
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)
TRANSFER_METRICS = (
    PROJECT / "03_models" / "v3_admin_transferability_domain_tests" / (
        "v3_admin_transferability_leave_one_domain_metrics.csv"
    )
)
AOA_BY_DOMAIN = (
    PROJECT / "04_maps" / "area_of_applicability_transfer_confidence_250m" / (
        "cpec_2018_area_of_applicability_by_subdomain.csv"
    )
)
RELIABILITY_BY_DOMAIN = (
    PROJECT / "04_maps" / "reliability_aware_susceptibility_2018_250m" / (
        "cpec_2018_reliability_aware_subdomain_area_summary.csv"
    )
)
ROAD_BY_DOMAIN = (
    PROJECT / "04_maps" / "reliability_aware_susceptibility_2018_250m" / (
        "reliability_aware_road_exposure_summary_by_domain_250m.csv"
    )
)

ALPHA = [f"A{i:02d}" for i in range(64)]
PUBLIC_DOMAINS = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]

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

DISPLAY_NAMES = {
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
    DISPLAY_NAMES[band] = f"AlphaEarth {band}"

GROUPS = {
    "Topography/morphometry": {
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "profile_curvature",
        "plan_curvature",
        "tri",
        "twi",
        "valley_depth",
    },
    "Hydro-climate": {"rain_monsoon_total", "rain_max_1day"},
    "Vegetation/land cover": {"ndvi_median", "ndvi_amplitude", "modis_lc_type1"},
    "Proximity controls": {"log1p_dist_road_m", "log1p_dist_river_m"},
    "Geology/soil/seismicity": {
        "log1p_dist_fault_m",
        "lithology_code",
        "soil_type",
        "eq_density_ms5",
    },
}


def slug(text: str) -> str:
    out = text.lower().replace("+", "and")
    out = re.sub(r"[^a-z0-9]+", "_", out)
    return out.strip("_")


def feature_group(feature: str) -> str:
    if feature.startswith("A") and feature[1:].isdigit():
        return "AlphaEarth embeddings"
    for group, members in GROUPS.items():
        if feature in members:
            return group
    return "Other"


def ensure_feature_sets() -> dict[str, dict[str, object]]:
    conventional = pd.read_csv(CONVENTIONAL_LIST)["factor"].tolist()
    FEATURE_SETS["conventional"]["features"] = conventional
    FEATURE_SETS["conventional_alphaearth_embeddings"]["features"] = conventional + ALPHA
    return FEATURE_SETS


def domain_dir(domain: str) -> Path:
    return DOMAIN_ROOT / slug(domain)


def shap_values_for_feature_set(df: pd.DataFrame, cfg: dict[str, object]) -> tuple[pd.DataFrame, np.ndarray]:
    features = list(cfg["features"])
    internal = str(cfg["internal"])
    model = joblib.load(MODEL_DIR / f"model_{internal}_xgboost.joblib")
    imputer = model.named_steps["imputer"]
    fitted = model.named_steps["model"]
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_imp = pd.DataFrame(imputer.transform(x), columns=features)
    explainer = shap.TreeExplainer(fitted)
    values = np.asarray(explainer.shap_values(x_imp))
    return x_imp, values


def importance_table(
    x_imp: pd.DataFrame,
    values: np.ndarray,
    feature_set_key: str,
    feature_set_label: str,
    domain: str,
    row_mask: np.ndarray,
) -> pd.DataFrame:
    features = list(x_imp.columns)
    domain_values = values[row_mask, :]
    domain_x = x_imp.loc[row_mask, :]
    rows = []
    for idx, feature in enumerate(features):
        vals = domain_values[:, idx]
        xvals = domain_x[feature].to_numpy()
        finite = np.isfinite(vals) & np.isfinite(xvals)
        corr = np.nan
        if finite.sum() >= 5 and np.nanstd(vals[finite]) > 0 and np.nanstd(xvals[finite]) > 0:
            corr = float(np.corrcoef(vals[finite], xvals[finite])[0, 1])
        rows.append(
            {
                "domain": domain,
                "feature_set_key": feature_set_key,
                "feature_set": feature_set_label,
                "feature": feature,
                "feature_name": DISPLAY_NAMES.get(feature, feature),
                "feature_group": feature_group(feature),
                "mean_abs_shap": float(np.mean(np.abs(vals))),
                "mean_shap": float(np.mean(vals)),
                "median_shap": float(np.median(vals)),
                "feature_value_mean": float(np.nanmean(xvals)),
                "feature_value_median": float(np.nanmedian(xvals)),
                "feature_value_shap_corr": corr,
                "n_domain_samples": int(row_mask.sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    total = out["mean_abs_shap"].sum()
    out["domain_relative_importance_percent"] = np.where(
        total > 0, out["mean_abs_shap"] / total * 100.0, 0.0
    )
    return out


def grouped_table(importance: pd.DataFrame) -> pd.DataFrame:
    group = (
        importance.groupby(["domain", "feature_set_key", "feature_set", "feature_group"], as_index=False)
        .agg(
            group_mean_abs_shap=("mean_abs_shap", "sum"),
            group_feature_count=("feature", "count"),
        )
        .sort_values(["domain", "feature_set", "group_mean_abs_shap"], ascending=[True, True, False])
    )
    total = group.groupby(["domain", "feature_set_key"])["group_mean_abs_shap"].transform("sum")
    group["group_relative_importance_percent"] = np.where(
        total > 0, group["group_mean_abs_shap"] / total * 100.0, 0.0
    )
    return group


def save_importance_plot(imp: pd.DataFrame, out_dir: Path, feature_set_key: str, title: str) -> None:
    top = imp.head(18).iloc[::-1]
    colors = top["feature_group"].map(
        {
            "Topography/morphometry": "#2b6cb0",
            "Hydro-climate": "#2f855a",
            "Vegetation/land cover": "#84a98c",
            "Proximity controls": "#d97706",
            "Geology/soil/seismicity": "#7b2cbf",
            "AlphaEarth embeddings": "#c2410c",
            "Other": "#555555",
        }
    )
    plt.figure(figsize=(8.4, 6.8))
    plt.barh(top["feature_name"], top["mean_abs_shap"], color=colors)
    plt.xlabel("Mean |TreeSHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"domain_treeshap_importance_{feature_set_key}.png", dpi=300)
    plt.savefig(out_dir / f"domain_treeshap_importance_{feature_set_key}.pdf")
    plt.close()


def save_beeswarm_plot(
    x_imp: pd.DataFrame,
    values: np.ndarray,
    row_mask: np.ndarray,
    out_dir: Path,
    feature_set_key: str,
    title: str,
) -> None:
    features = list(x_imp.columns)
    idx = np.flatnonzero(row_mask)
    if len(idx) > 450:
        rng = np.random.default_rng(141300)
        idx = np.sort(rng.choice(idx, size=450, replace=False))
    names = [DISPLAY_NAMES.get(f, f) for f in features]
    explanation = shap.Explanation(values=values[idx, :], data=x_imp.iloc[idx, :].to_numpy(), feature_names=names)
    plt.figure(figsize=(8.6, 6.8))
    shap.plots.beeswarm(
        explanation,
        show=False,
        max_display=18,
        group_remaining_features=True,
        color_bar_label="Feature value",
    )
    plt.title(title, pad=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"domain_treeshap_beeswarm_{feature_set_key}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"domain_treeshap_beeswarm_{feature_set_key}.pdf", bbox_inches="tight")
    plt.close()


def save_group_plot(group_df: pd.DataFrame, out_dir: Path, domain: str) -> None:
    d = group_df[group_df["domain"].eq(domain)].copy()
    order = [
        "Topography/morphometry",
        "Hydro-climate",
        "Vegetation/land cover",
        "Proximity controls",
        "Geology/soil/seismicity",
        "AlphaEarth embeddings",
    ]
    pivot = d.pivot_table(
        index="feature_group",
        columns="feature_set",
        values="group_relative_importance_percent",
        aggfunc="sum",
        fill_value=0,
    ).reindex(order).fillna(0)
    plt.figure(figsize=(8.6, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "% of domain SHAP importance"})
    plt.title(f"Domain Mechanism Groups: {domain}")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "domain_mechanism_group_shap_heatmap.png", dpi=300)
    plt.savefig(out_dir / "domain_mechanism_group_shap_heatmap.pdf")
    plt.close()


def read_domain_comparison() -> pd.DataFrame:
    metrics = pd.read_csv(TRANSFER_METRICS)
    stacked = metrics.loc[metrics["model"].eq("Spatial-CV Stacked Ensemble")].copy()
    wide = stacked.pivot_table(
        index="held_out_domain",
        columns="feature_set",
        values=["roc_auc", "pr_auc_ap", "brier", "f1", "balanced_accuracy"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{slug(feature_set)}" for metric, feature_set in wide.columns]
    wide = wide.reset_index().rename(columns={"held_out_domain": "domain"})

    aoa = pd.read_csv(AOA_BY_DOMAIN)
    fused_aoa = aoa.loc[aoa["feature_set"].eq("Conventional + AlphaEarth Embeddings")][
        ["domain", "in_area_of_applicability_percent", "mean_transfer_confidence"]
    ].rename(
        columns={
            "in_area_of_applicability_percent": "fused_aoa_coverage_percent",
            "mean_transfer_confidence": "fused_mean_transfer_confidence",
        }
    )
    reliability = pd.read_csv(RELIABILITY_BY_DOMAIN)
    road = pd.read_csv(ROAD_BY_DOMAIN)
    kkh = road.loc[road["road_group"].str.contains("KKH", case=False, na=False)].copy()
    if not kkh.empty:
        kkh = kkh[
            [
                "domain",
                "total_length_km",
                "reliable_high_weighted_length_km",
                "uncertain_high_weighted_length_km",
                "mean_probability_length_weighted",
            ]
        ].rename(
            columns={
                "total_length_km": "kkh_proxy_length_km",
                "reliable_high_weighted_length_km": "kkh_reliable_high_weighted_length_km",
                "uncertain_high_weighted_length_km": "kkh_uncertain_high_weighted_length_km",
                "mean_probability_length_weighted": "kkh_mean_fused_probability",
            }
        )
    out = wide.merge(fused_aoa, on="domain", how="left").merge(reliability, on="domain", how="left")
    if not kkh.empty:
        out = out.merge(kkh, on="domain", how="left")
    return out


def save_summary_figures(imp_all: pd.DataFrame, group_all: pd.DataFrame, comparison: pd.DataFrame) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    fused = comparison.set_index("domain")
    cols = [
        "roc_auc_conventional",
        "roc_auc_alphaearth_embeddings",
        "roc_auc_conventional_and_alphaearth_embeddings",
        "fused_aoa_coverage_percent",
        "mean_fused_transfer_confidence",
        "uncertain_high_area_percent",
    ]
    available = [c for c in cols if c in fused.columns]
    heat = fused.loc[PUBLIC_DOMAINS, available].copy()
    normalized = heat.copy()
    for c in normalized.columns:
        vals = pd.to_numeric(normalized[c], errors="coerce")
        if vals.max() > vals.min():
            normalized[c] = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            normalized[c] = 0.0
    label_map = {
        "roc_auc_conventional": "AUC\nConventional",
        "roc_auc_alphaearth_embeddings": "AUC\nAlphaEarth",
        "roc_auc_conventional_and_alphaearth_embeddings": "AUC\nFused",
        "fused_aoa_coverage_percent": "Fused AOA\ncoverage",
        "mean_fused_transfer_confidence": "Mean transfer\nconfidence",
        "uncertain_high_area_percent": "Uncertain high\narea",
    }
    normalized.columns = [label_map.get(c, c) for c in normalized.columns]
    plt.figure(figsize=(10.2, 5.2))
    sns.heatmap(normalized, annot=heat.round(3), fmt="", cmap="viridis", cbar_kws={"label": "Column-normalized score"})
    plt.title("Domain Transferability and Reliability Comparison")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(SUMMARY_ROOT / "figure_domain_transferability_reliability_comparison_heatmap.png", dpi=300)
    plt.savefig(SUMMARY_ROOT / "figure_domain_transferability_reliability_comparison_heatmap.pdf")
    plt.close()

    fused_imp = imp_all.loc[imp_all["feature_set"].eq("Conventional + AlphaEarth Embeddings")].copy()
    top_features = (
        fused_imp.groupby(["feature", "feature_name"], as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)["feature"]
        .tolist()
    )
    heat = fused_imp.loc[fused_imp["feature"].isin(top_features)].pivot_table(
        index="domain",
        columns="feature_name",
        values="domain_relative_importance_percent",
        aggfunc="first",
        fill_value=0,
    )
    ordered_names = [
        DISPLAY_NAMES.get(f, f) for f in top_features if DISPLAY_NAMES.get(f, f) in heat.columns
    ]
    heat = heat.reindex(PUBLIC_DOMAINS)[ordered_names]
    plt.figure(figsize=(13, 5.2))
    sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "% of domain SHAP importance"})
    plt.title("Domain-Specific Top Factors: Fused XGBoost TreeSHAP")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(SUMMARY_ROOT / "figure_domain_top_factor_shap_heatmap_fused.png", dpi=300)
    plt.savefig(SUMMARY_ROOT / "figure_domain_top_factor_shap_heatmap_fused.pdf")
    plt.close()

    fused_groups = group_all.loc[group_all["feature_set"].eq("Conventional + AlphaEarth Embeddings")].copy()
    gpiv = fused_groups.pivot_table(
        index="domain",
        columns="feature_group",
        values="group_relative_importance_percent",
        aggfunc="first",
        fill_value=0,
    ).reindex(PUBLIC_DOMAINS).fillna(0)
    plt.figure(figsize=(10.5, 4.8))
    sns.heatmap(gpiv, cmap="PuBuGn", annot=True, fmt=".1f", cbar_kws={"label": "% of fused SHAP importance"})
    plt.title("Domain Mechanism Groups: Fused Feature Set")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(SUMMARY_ROOT / "figure_domain_mechanism_group_heatmap_fused.png", dpi=300)
    plt.savefig(SUMMARY_ROOT / "figure_domain_mechanism_group_heatmap_fused.pdf")
    plt.close()


def mechanism_sentence(domain: str, imp: pd.DataFrame, group: pd.DataFrame, comparison: pd.DataFrame) -> str:
    fused_imp = imp.loc[
        imp["domain"].eq(domain) & imp["feature_set"].eq("Conventional + AlphaEarth Embeddings")
    ].head(5)
    top_features = ", ".join(fused_imp["feature_name"].tolist())
    fused_group = group.loc[
        group["domain"].eq(domain) & group["feature_set"].eq("Conventional + AlphaEarth Embeddings")
    ].head(2)
    top_groups = ", ".join(
        f"{r.feature_group} ({r.group_relative_importance_percent:.1f}%)"
        for r in fused_group.itertuples()
    )
    crow = comparison.loc[comparison["domain"].eq(domain)].iloc[0]
    auc_val = crow.get("roc_auc_conventional_and_alphaearth_embeddings", np.nan)
    aoa_val = crow.get("fused_aoa_coverage_percent", np.nan)
    return (
        f"- **{domain}**: fused leave-one-domain-out AUC = {auc_val:.3f}; "
        f"fused AOA coverage = {aoa_val:.1f}%. Dominant mechanism groups are {top_groups}. "
        f"Top fused factors are {top_features}."
    )


def write_reports(imp_all: pd.DataFrame, group_all: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Domain-Wise Mechanism and Transferability Interpretation",
        "",
        "## What Was Done",
        "",
        "The CPEC-wide 2018 V3 Spatial-CV Stacked Ensemble remains the main prediction model. "
        "For domain-wise factor mechanisms, TreeSHAP was computed for the XGBoost base learner "
        "inside each official feature set. This keeps the modelling strategy comparable across "
        "domains while still showing why the model behaves differently in each domain.",
        "",
        "## What This Adds Scientifically",
        "",
        "- It tests model generalization using leave-one-domain-out transferability.",
        "- It shows where the model is inside or outside its Area of Applicability.",
        "- It identifies which controlling mechanisms dominate each CPEC subdomain.",
        "- It avoids weak separate regional retraining while preserving domain-level interpretation.",
        "",
        "## Domain Mechanism Summary",
        "",
    ]
    for domain in PUBLIC_DOMAINS:
        lines.append(mechanism_sentence(domain, imp_all, group_all, comparison))
    lines.extend(
        [
            "",
            "## Main Tables",
            "",
            "- `domain_transferability_mechanism_comparison_table.csv`",
            "- `domainwise_treeshap_importance_all_feature_sets.csv`",
            "- `domainwise_treeshap_group_importance_all_feature_sets.csv`",
            "",
            "## Main Figures",
            "",
            "- `figure_domain_transferability_reliability_comparison_heatmap.png`",
            "- `figure_domain_top_factor_shap_heatmap_fused.png`",
            "- `figure_domain_mechanism_group_heatmap_fused.png`",
            "",
            "## Suggested Paper Wording",
            "",
            "A domain-wise transferability and explanation analysis was conducted to test whether "
            "the CPEC-wide model generalized across politically and physiographically contrasting "
            "subdomains. Instead of fitting independent regional models, which would reduce sample "
            "size and weaken comparability, we used leave-one-domain-out validation, Area of "
            "Applicability mapping, and domain-specific TreeSHAP summaries of the XGBoost base "
            "learner. This design quantifies both predictive transferability and spatially varying "
            "conditioning mechanisms.",
        ]
    )
    (SUMMARY_ROOT / "domainwise_mechanism_and_transferability_interpretation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    for domain in PUBLIC_DOMAINS:
        ddir = domain_dir(domain)
        expl_dir = ddir / "06_domain_explainability"
        report = [
            f"# Domain-Wise SHAP and Mechanism Outputs: {domain}",
            "",
            "These outputs explain original conditioning factors for the XGBoost base learner inside each feature set. "
            "They complement, but do not replace, the main Spatial-CV Stacked Ensemble results.",
            "",
            mechanism_sentence(domain, imp_all, group_all, comparison).replace(f"- **{domain}**: ", ""),
            "",
            "## Files",
            "",
            "- `domain_treeshap_importance_all_feature_sets.csv`",
            "- `domain_treeshap_group_importance_all_feature_sets.csv`",
            "- `domain_treeshap_importance_*.png`",
            "- `domain_treeshap_beeswarm_*.png`",
            "- `domain_mechanism_group_shap_heatmap.png`",
        ]
        (expl_dir / "README_domain_shap_mechanisms.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY_DIR.mkdir(parents=True, exist_ok=True)

    feature_sets = ensure_feature_sets()
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    all_imp = []

    for key, cfg in feature_sets.items():
        x_imp, values = shap_values_for_feature_set(df, cfg)
        for domain in PUBLIC_DOMAINS:
            row_mask = df["cpec_admin_transfer_domain"].eq(domain).to_numpy()
            imp = importance_table(x_imp, values, key, str(cfg["label"]), domain, row_mask)
            all_imp.append(imp)

            out_dir = domain_dir(domain) / "06_domain_explainability"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_importance_plot(
                imp,
                out_dir,
                key,
                f"Domain TreeSHAP Importance: {domain}\n{cfg['label']}",
            )
            save_beeswarm_plot(
                x_imp,
                values,
                row_mask,
                out_dir,
                key,
                f"Domain TreeSHAP Beeswarm: {domain}\n{cfg['label']}",
            )

    imp_all = pd.concat(all_imp, ignore_index=True)
    group_all = grouped_table(imp_all)
    comparison = read_domain_comparison()

    imp_all.to_csv(SUMMARY_ROOT / "domainwise_treeshap_importance_all_feature_sets.csv", index=False)
    group_all.to_csv(SUMMARY_ROOT / "domainwise_treeshap_group_importance_all_feature_sets.csv", index=False)
    comparison.to_csv(SUMMARY_ROOT / "domain_transferability_mechanism_comparison_table.csv", index=False)

    for domain in PUBLIC_DOMAINS:
        out_dir = domain_dir(domain) / "06_domain_explainability"
        imp_all.loc[imp_all["domain"].eq(domain)].to_csv(
            out_dir / "domain_treeshap_importance_all_feature_sets.csv", index=False
        )
        group_all.loc[group_all["domain"].eq(domain)].to_csv(
            out_dir / "domain_treeshap_group_importance_all_feature_sets.csv", index=False
        )
        save_group_plot(group_all, out_dir, domain)

    save_summary_figures(imp_all, group_all, comparison)
    write_reports(imp_all, group_all, comparison)

    shutil.copy2(Path(__file__), SCRIPT_COPY_DIR / Path(__file__).name)
    print(SUMMARY_ROOT)


if __name__ == "__main__":
    main()
