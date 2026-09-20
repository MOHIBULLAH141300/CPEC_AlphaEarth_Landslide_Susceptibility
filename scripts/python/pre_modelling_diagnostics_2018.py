"""Run pre-modelling diagnostics for the clean 2018 CPEC LSM sample table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "pre_modelling_diagnostics_2018"
REPORT = PROJECT_ROOT / "05_reports" / "pre_modelling_diagnostics_2018.md"


CONVENTIONAL = [
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


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for col in cols:
        y = x[col].to_numpy()
        other = [c for c in cols if c != col]
        model = LinearRegression()
        model.fit(x[other], y)
        r2 = model.score(x[other], y)
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "r2": r2, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    predictors = CONVENTIONAL + alpha

    # Ensure numeric conversion for all predictors and label/fold fields.
    for col in predictors + ["label", "spatial_fold_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = (
        df[predictors]
        .isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    ranges = df[predictors].agg(["min", "max", "mean", "std"]).T.reset_index().rename(columns={"index": "feature"})
    near_zero = ranges[(ranges["std"].fillna(0) < 1e-8) | (ranges["min"] == ranges["max"])].copy()

    label_fold = (
        df.groupby(["spatial_fold_5", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["spatial_fold_5", "label"])
    )
    label_counts = df["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")

    conv_corr = df[CONVENTIONAL].corr(method="spearman")
    high_corr_rows = []
    for i, a in enumerate(CONVENTIONAL):
        for b in CONVENTIONAL[i + 1 :]:
            val = conv_corr.loc[a, b]
            if abs(val) >= 0.85:
                high_corr_rows.append({"feature_a": a, "feature_b": b, "spearman_r": val})
    high_corr = pd.DataFrame(high_corr_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)

    # VIF can be unstable with categorical land cover; keep it in the table but
    # interpret it as a redundancy screen, not a causal statement.
    vif = vif_table(df, CONVENTIONAL)

    alpha_corr = df[alpha].corr(method="spearman").abs()
    alpha_pairs = []
    for i, a in enumerate(alpha):
        for b in alpha[i + 1 :]:
            val = alpha_corr.loc[a, b]
            if val >= 0.95:
                alpha_pairs.append({"feature_a": a, "feature_b": b, "abs_spearman_r": val})
    alpha_high_corr = pd.DataFrame(alpha_pairs, columns=["feature_a", "feature_b", "abs_spearman_r"])
    if not alpha_high_corr.empty:
        alpha_high_corr = alpha_high_corr.sort_values("abs_spearman_r", ascending=False)

    missing.to_csv(OUT_DIR / "missing_rates.csv", index=False)
    ranges.to_csv(OUT_DIR / "predictor_ranges.csv", index=False)
    near_zero.to_csv(OUT_DIR / "near_zero_variance.csv", index=False)
    label_fold.to_csv(OUT_DIR / "fold_label_balance.csv", index=False)
    label_counts.to_csv(OUT_DIR / "label_counts.csv", index=False)
    conv_corr.to_csv(OUT_DIR / "conventional_spearman_correlation.csv")
    high_corr.to_csv(OUT_DIR / "conventional_high_correlation_pairs.csv", index=False)
    vif.to_csv(OUT_DIR / "conventional_vif.csv", index=False)
    alpha_high_corr.to_csv(OUT_DIR / "alphaearth_high_correlation_pairs.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "predictor_count": len(predictors),
        "conventional_count": len(CONVENTIONAL),
        "alphaearth_count": len(alpha),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "max_missing_rate": float(missing["missing_rate"].max()),
        "near_zero_variance_count": int(len(near_zero)),
        "conventional_high_corr_pair_count_abs_ge_0_85": int(len(high_corr)),
        "alphaearth_high_corr_pair_count_abs_ge_0_95": int(len(alpha_high_corr)),
        "max_conventional_vif": float(vif["vif"].replace(np.inf, np.nan).max()),
    }
    (OUT_DIR / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Pre-Modelling Diagnostics 2018",
        "",
        f"- Rows: {summary['rows']}",
        f"- Predictors: {summary['predictor_count']} ({summary['conventional_count']} conventional + {summary['alphaearth_count']} AlphaEarth)",
        f"- Label counts: {summary['label_counts']}",
        f"- Maximum missing rate: {summary['max_missing_rate']:.4f}",
        f"- Near-zero-variance predictors: {summary['near_zero_variance_count']}",
        f"- Conventional high-correlation pairs |r| >= 0.85: {summary['conventional_high_corr_pair_count_abs_ge_0_85']}",
        f"- AlphaEarth high-correlation pairs |r| >= 0.95: {summary['alphaearth_high_corr_pair_count_abs_ge_0_95']}",
        f"- Maximum finite conventional VIF: {summary['max_conventional_vif']:.2f}",
        "",
        "## Decision Rule",
        "",
        "- If missingness is low and every fold contains both classes, modelling can proceed.",
        "- Conventional predictors with high correlation/VIF should be handled by feature-set ablation, not silently ignored.",
        "- AlphaEarth redundancy should be reported as an embedding diagnostic; do not remove bands by VIF before the first fair comparison.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
