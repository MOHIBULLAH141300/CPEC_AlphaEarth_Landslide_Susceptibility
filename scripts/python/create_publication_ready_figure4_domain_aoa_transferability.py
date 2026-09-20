from pathlib import Path
import textwrap

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj


ROOT = Path(r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS")
OUTDIR = Path(r"D:\DING PROJECT\05_reports\manuscript_drafts\assembled_figures")

DOMAIN_GPKG = ROOT / "10_domain_specific_2018_results/cpec_2018_v3_transfer_domains.gpkg"
AOA_CSV = ROOT / "02_model_performance_tables/cpec_2018_area_of_applicability_by_subdomain.csv"
DOMAIN_CSV = ROOT / "10_domain_specific_2018_results/00_domain_comparison_summary/domain_transferability_mechanism_comparison_table.csv"

OUT_PNG = OUTDIR / "figure_R4_subdomains_aoa_transferability_pubready.png"
OUT_PDF = OUTDIR / "figure_R4_subdomains_aoa_transferability_pubready.pdf"


DOMAIN_ORDER = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]

DOMAIN_LABELS = {
    "Kashgar (Xinjiang, China)": "Kashgar\n(Xinjiang)",
    "Gilgit-Baltistan": "Gilgit-\nBaltistan",
    "KP-AJK": "KPK-AJK",
    "Balochistan": "Balochistan",
    "Punjab-Sindh lowland corridor": "Punjab-\nSindh",
}

DOMAIN_COLORS = {
    "Kashgar (Xinjiang, China)": "#7E57C2",
    "Gilgit-Baltistan": "#0072B2",
    "KP-AJK": "#009E73",
    "Balochistan": "#D55E00",
    "Punjab-Sindh lowland corridor": "#E69F00",
}

FEATURE_ORDER = [
    "Conventional",
    "AlphaEarth Embeddings",
    "Conventional + AlphaEarth Embeddings",
]

FEATURE_LABELS = {
    "Conventional": "Conventional",
    "AlphaEarth Embeddings": "AlphaEarth",
    "Conventional + AlphaEarth Embeddings": "Conv. + AlphaEarth",
}

FEATURE_COLORS = {
    "Conventional": "#6B7280",
    "AlphaEarth Embeddings": "#2563EB",
    "Conventional + AlphaEarth Embeddings": "#059669",
}


def setup_pyproj():
    proj_dir = Path(r"D:\MINICONDA\Library\share\proj")
    if proj_dir.exists():
        pyproj.datadir.set_data_dir(str(proj_dir))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def make_figure():
    setup_pyproj()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    domains = gpd.read_file(DOMAIN_GPKG).to_crs("EPSG:4326")
    domains["domain"] = pd.Categorical(domains["domain"], DOMAIN_ORDER, ordered=True)
    domains = domains.sort_values("domain")

    aoa = pd.read_csv(AOA_CSV)
    aoa["domain"] = pd.Categorical(aoa["domain"], DOMAIN_ORDER, ordered=True)
    aoa["feature_set"] = pd.Categorical(aoa["feature_set"], FEATURE_ORDER, ordered=True)
    aoa = aoa.sort_values(["domain", "feature_set"])

    domain = pd.read_csv(DOMAIN_CSV)
    domain["domain"] = pd.Categorical(domain["domain"], DOMAIN_ORDER, ordered=True)
    domain = domain.sort_values("domain")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig = plt.figure(figsize=(7.2, 9.4), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.0, 1.02], hspace=0.72)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_aoa = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[2, 0])

    for _, row in domains.iterrows():
        gpd.GeoSeries([row.geometry], crs=domains.crs).plot(
            ax=ax_map,
            color=DOMAIN_COLORS[str(row["domain"])],
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.78,
        )
        pt = row.geometry.representative_point()
        ax_map.text(
            pt.x,
            pt.y,
            DOMAIN_LABELS[str(row["domain"])],
            ha="center",
            va="center",
            fontsize=8.0,
            weight="bold",
            color="white",
            path_effects=[],
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
        )
    ax_map.set_title("(a) CPEC study-area subdomains", loc="left", weight="bold")
    ax_map.set_xlabel("Longitude (degrees)")
    ax_map.set_ylabel("Latitude (degrees)")
    ax_map.grid(color="#D1D5DB", linewidth=0.4, alpha=0.6)
    ax_map.set_aspect("equal")
    style_axes(ax_map)

    pivot = aoa.pivot(index="domain", columns="feature_set", values="in_area_of_applicability_percent").loc[DOMAIN_ORDER]
    y = np.arange(len(DOMAIN_ORDER))
    offsets = [-0.23, 0.0, 0.23]
    bar_h = 0.20
    for feature, offset in zip(FEATURE_ORDER, offsets):
        vals = pivot[feature].values
        bars = ax_aoa.barh(
            y + offset,
            vals,
            height=bar_h,
            color=FEATURE_COLORS[feature],
            label=FEATURE_LABELS[feature],
        )
        for bar, val in zip(bars, vals):
            label_x = min(val + 1.1, 98.2)
            ax_aoa.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left" if val < 94 else "right",
                fontsize=7.2,
                color="#111827",
            )
    ax_aoa.set_yticks(y)
    ax_aoa.set_yticklabels([DOMAIN_LABELS[d].replace("\n", " ") for d in DOMAIN_ORDER])
    ax_aoa.invert_yaxis()
    ax_aoa.set_xlim(0, 105)
    ax_aoa.set_xlabel("Pixels inside area of applicability (%)")
    ax_aoa.set_title("(b) Area-of-applicability coverage", loc="left", weight="bold")
    ax_aoa.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.7)
    ax_aoa.legend(frameon=False, ncol=3, fontsize=7.0, loc="upper right", bbox_to_anchor=(1.0, 1.22))
    style_axes(ax_aoa)

    heat_cols = [
        ("roc_auc_conventional", "AUC\nConventional", "{:.3f}"),
        ("roc_auc_alphaearth_embeddings", "AUC\nAlphaEarth", "{:.3f}"),
        ("roc_auc_conventional_and_alphaearth_embeddings", "AUC\nFused", "{:.3f}"),
        ("mean_fused_transfer_confidence", "Transfer\nconfidence", "{:.3f}"),
        ("uncertain_high_area_percent", "Uncertain\nhigh area (%)", "{:.1f}"),
    ]
    raw = domain[[c[0] for c in heat_cols]].astype(float).to_numpy()
    norm = np.zeros_like(raw, dtype=float)
    for j in range(raw.shape[1]):
        mn = np.nanmin(raw[:, j])
        mx = np.nanmax(raw[:, j])
        norm[:, j] = 0.5 if mx == mn else (raw[:, j] - mn) / (mx - mn)

    im = ax_heat.imshow(norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(heat_cols)))
    ax_heat.set_xticklabels([c[1] for c in heat_cols])
    ax_heat.set_yticks(np.arange(len(DOMAIN_ORDER)))
    ax_heat.set_yticklabels([DOMAIN_LABELS[d].replace("\n", " ") for d in DOMAIN_ORDER])
    ax_heat.set_title("(c) Leave-one-domain-out transferability and reliability", loc="left", weight="bold")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            text_color = "white" if norm[i, j] < 0.35 else "#111827"
            ax_heat.text(
                j,
                i,
                heat_cols[j][2].format(raw[i, j]),
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
                weight="bold" if j == 2 else "normal",
            )
    ax_heat.tick_params(axis="x", rotation=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("Column-scaled color intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(top=0.985)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    make_figure()
