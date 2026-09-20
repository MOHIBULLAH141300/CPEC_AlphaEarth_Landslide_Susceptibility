"""Create domain-wise VIF diagnostics for the final 2018 V3 conventional factors.

These are diagnostic only. The final factor selection remains global so the
domain comparisons remain comparable and reviewer-safe.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj.datadir
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
DOMAIN_ROOT = PACKAGE / "10_domain_specific_2018_results"
SUMMARY_ROOT = DOMAIN_ROOT / "00_domain_comparison_summary"
SCRIPT_COPY_DIR = PACKAGE / "07_reproducible_scripts"
DATA = PROJECT / "03_models" / "v3_admin_transferability_domain_tests" / (
    "cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv"
)
CONVENTIONAL_LIST = (
    PROJECT
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)

PUBLIC_DOMAINS = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]

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


def slug(text: str) -> str:
    out = text.lower().replace("+", "and")
    out = re.sub(r"[^a-z0-9]+", "_", out)
    return out.strip("_")


def calculate_vif(x: pd.DataFrame) -> pd.DataFrame:
    features = list(x.columns)
    arr = x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    arr = SimpleImputer(strategy="median").fit_transform(arr)
    arr = StandardScaler().fit_transform(arr)
    rows = []
    for i, feature in enumerate(features):
        y = arr[:, i]
        other = np.delete(arr, i, axis=1)
        if np.nanstd(y) == 0:
            vif = np.inf
            r2 = 1.0
        else:
            r2 = LinearRegression().fit(other, y).score(other, y)
            vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append(
            {
                "factor": feature,
                "factor_name": DISPLAY_NAMES.get(feature, feature),
                "r2_against_other_factors": float(r2),
                "vif": float(vif),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    factors = pd.read_csv(CONVENTIONAL_LIST)["factor"].tolist()
    rows = []
    for domain in PUBLIC_DOMAINS:
        d = df.loc[df["cpec_admin_transfer_domain"].eq(domain)].copy()
        vif = calculate_vif(d[factors])
        vif.insert(0, "domain", domain)
        vif.insert(1, "n_domain_samples", len(d))
        vif["diagnostic_role"] = "Domain diagnostic only; global VIF remains the final factor-selection gate."
        rows.append(vif)

        domain_table_dir = DOMAIN_ROOT / slug(domain) / "03_tables"
        domain_table_dir.mkdir(parents=True, exist_ok=True)
        vif.to_csv(domain_table_dir / "domain_conventional_vif_diagnostic.csv", index=False)

    all_vif = pd.concat(rows, ignore_index=True)
    all_vif.to_csv(SUMMARY_ROOT / "domainwise_conventional_vif_diagnostics.csv", index=False)

    pivot = all_vif.pivot_table(index="domain", columns="factor_name", values="vif", aggfunc="first")
    top_factors = (
        all_vif.groupby("factor_name", as_index=False)["vif"]
        .max()
        .sort_values("vif", ascending=False)
        .head(14)["factor_name"]
        .tolist()
    )
    plot_data = pivot.reindex(PUBLIC_DOMAINS)[top_factors].clip(upper=20)
    plt.figure(figsize=(12.4, 5.2))
    sns.heatmap(plot_data, annot=True, fmt=".1f", cmap="OrRd", cbar_kws={"label": "VIF, clipped at 20"})
    plt.title("Domain-Wise Conventional-Factor VIF Diagnostics")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(SUMMARY_ROOT / "figure_domainwise_conventional_vif_diagnostics.png", dpi=300)
    plt.savefig(SUMMARY_ROOT / "figure_domainwise_conventional_vif_diagnostics.pdf")
    plt.close()

    note = """# Domain-Wise VIF Diagnostics

These VIF tables are diagnostic only. The final Conventional feature set was
selected using the global V3 multicollinearity gate so that all domain
comparisons use the same factors.

Domain-wise VIF can be higher or lower because each domain has a narrower
environmental range and fewer samples. High local VIF therefore indicates local
covariate redundancy, not a reason to change the main model separately for that
domain.
"""
    (SUMMARY_ROOT / "domainwise_vif_diagnostics_note.md").write_text(note, encoding="utf-8")
    shutil.copy2(Path(__file__), SCRIPT_COPY_DIR / Path(__file__).name)
    print(SUMMARY_ROOT / "domainwise_conventional_vif_diagnostics.csv")


if __name__ == "__main__":
    main()
