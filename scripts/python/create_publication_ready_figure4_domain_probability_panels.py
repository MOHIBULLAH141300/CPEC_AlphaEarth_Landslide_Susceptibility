from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS")
OUTDIR = Path(r"D:\DING PROJECT\05_reports\manuscript_drafts\assembled_figures")
OUT_PNG = OUTDIR / "figure_R4_domain_specific_probability_panels_pubready.png"
OUT_PDF = OUTDIR / "figure_R4_domain_specific_probability_panels_pubready.pdf"

DOMAIN_DIR = ROOT / "10_domain_specific_2018_results"
DOMAIN_CSV = DOMAIN_DIR / "00_domain_comparison_summary/domain_transferability_mechanism_comparison_table.csv"

DOMAINS = [
    ("kashgar_xinjiang_china", "Kashgar\n(Xinjiang)"),
    ("gilgit_baltistan", "Gilgit-\nBaltistan"),
    ("kp_ajk", "KPK-AJK"),
    ("balochistan", "Balochistan"),
    ("punjab_sindh_lowland_corridor", "Punjab-\nSindh"),
]


def read_downsampled(path, max_dim=760):
    with rasterio.open(path) as src:
        scale = max(1, int(np.ceil(max(src.width, src.height) / max_dim)))
        out_shape = (max(1, src.height // scale), max(1, src.width // scale))
        data = src.read(1, out_shape=out_shape, masked=True)
        transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
        left, top = transform * (0, 0)
        right, bottom = transform * (out_shape[1], out_shape[0])
        extent = [left, right, bottom, top]
        return data, extent


def fused_path(domain_slug):
    return (
        DOMAIN_DIR
        / domain_slug
        / "01_probability_maps_250m"
        / f"cpec_2018_conventional_alphaearth_embeddings_stacked_probability_250m_{domain_slug}.tif"
    )


def metrics_panel(ax):
    df = pd.read_csv(DOMAIN_CSV).replace({"domain": {"KP-AJK": "KPK-AJK"}})
    order = [
        "Kashgar (Xinjiang, China)",
        "Gilgit-Baltistan",
        "KPK-AJK",
        "Balochistan",
        "Punjab-Sindh lowland corridor",
    ]
    df["domain_order"] = pd.Categorical(df["domain"], order, ordered=True)
    df = df.sort_values("domain_order")
    labels = ["Kashgar", "Gilgit-Baltistan", "KPK-AJK", "Balochistan", "Punjab-Sindh"]
    data = np.column_stack(
        [
            df["roc_auc_conventional_and_alphaearth_embeddings"].astype(float),
            df["fused_aoa_coverage_percent"].astype(float) / 100.0,
            df["mean_fused_transfer_confidence"].astype(float),
            df["uncertain_high_area_percent"].astype(float) / 100.0,
        ]
    )
    col_labels = ["ROC-AUC", "AoA\ncoverage", "Transfer\nconfidence", "Uncertain\nhigh area"]
    shown = np.column_stack(
        [
            df["roc_auc_conventional_and_alphaearth_embeddings"].astype(float),
            df["fused_aoa_coverage_percent"].astype(float),
            df["mean_fused_transfer_confidence"].astype(float),
            df["uncertain_high_area_percent"].astype(float),
        ]
    )
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = shown[i, j]
            text = f"{value:.3f}" if j in (0, 2) else f"{value:.1f}%"
            color = "white" if data[i, j] < 0.35 else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color=color)
    ax.set_title("(f) Fused stacked-ensemble diagnostics", loc="left", fontsize=9.5, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.1), dpi=300, constrained_layout=True)
    axes = axes.ravel()
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    last_im = None
    for ax, (slug, label), letter in zip(axes[:5], DOMAINS, letters):
        data, extent = read_downsampled(fused_path(slug))
        im = ax.imshow(data, extent=extent, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        last_im = im
        ax.set_title(f"{letter} {label}", loc="left", fontsize=9.5, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#374151")
    metrics_panel(axes[5])
    cbar = fig.colorbar(last_im, ax=axes[:5].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Susceptibility probability", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle("Domain-specific fused susceptibility and transferability diagnostics", fontsize=11.2, weight="bold")
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
