"""Create the revised comparative transferability and AoA figures for V6."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
import shapefile


PROJECT = Path(r"D:\DING PROJECT")
LODO = PROJECT / "03_models" / "manuscript_v3_leave_one_domain_out"
V6 = PROJECT / "03_models" / "manuscript_v6_audit_resolutions"
RASTERS = PROJECT / "04_maps" / "manuscript_v6_lodo_aoa_250m"
DOMAIN_SHP = (
    PROJECT
    / "FINAL PAPER"
    / "00_Study_Area"
    / "Figure_1_ArcMap_layers"
    / "02_CPEC_transfer_subdomains.shp"
)
OUT = PROJECT / "ding review" / "v6_2026-08-01" / "figures"

PREDICTIONS = LODO / "leave_one_domain_out_predictions.csv"
INTERVALS = LODO / "leave_one_domain_out_block_bootstrap_ci.csv"
PAIRED = V6 / "paired_lodo_feature_set_differences.csv"
SENSITIVITY = V6 / "aoa_sensitivity_summary.csv"

FEATURE_ORDER = [
    "conventional",
    "alphaearth_embeddings",
    "conventional_alphaearth_embeddings",
]
FEATURE_LABEL = {
    "conventional": "Conventional",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_alphaearth_embeddings": "Conventional + AlphaEarth Embeddings",
}
FEATURE_COLOR = {
    "conventional": "#2878B5",
    "alphaearth_embeddings": "#E07A2D",
    "conventional_alphaearth_embeddings": "#1B9E77",
}
FEATURE_MARKER = {
    "conventional": "o",
    "alphaearth_embeddings": "^",
    "conventional_alphaearth_embeddings": "s",
}
DOMAIN_ORDER = ["Kashgar", "Gilgit-Baltistan", "KPK-AJK", "Balochistan", "Punjab-Sindh"]
EXTENT = (60.5, 80.1, 23.4, 41.5)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_domain_boundaries(ax, linewidth: float = 0.75, color: str = "#31363B") -> None:
    reader = shapefile.Reader(str(DOMAIN_SHP), encoding="latin1")
    for shape_record in reader.iterShapeRecords():
        points = np.asarray(shape_record.shape.points)
        parts = list(shape_record.shape.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            part = points[start:end]
            ax.plot(part[:, 0], part[:, 1], color=color, linewidth=linewidth, zorder=5)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.012,
        0.988,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="#333333", linewidth=0.6),
        zorder=20,
    )


def interval_lookup(intervals: pd.DataFrame, feature_set: str, domain: str, metric: str):
    row = intervals[
        (intervals.feature_set == feature_set)
        & (intervals.held_out_domain == domain)
        & (intervals.metric == metric)
    ].iloc[0]
    return float(row.estimate), float(row.ci95_low), float(row.ci95_high)


def plot_metric_intervals(ax, intervals: pd.DataFrame, metric: str, title: str, xlim) -> None:
    y_base = np.arange(len(DOMAIN_ORDER))[::-1]
    offsets = {FEATURE_ORDER[0]: 0.20, FEATURE_ORDER[1]: 0.0, FEATURE_ORDER[2]: -0.20}
    for feature_set in FEATURE_ORDER:
        estimates, lower, upper = [], [], []
        for domain in DOMAIN_ORDER:
            estimate, lo, hi = interval_lookup(intervals, feature_set, domain, metric)
            estimates.append(estimate)
            lower.append(estimate - lo)
            upper.append(hi - estimate)
        y = y_base + offsets[feature_set]
        ax.errorbar(
            estimates,
            y,
            xerr=np.vstack([lower, upper]),
            fmt=FEATURE_MARKER[feature_set],
            markersize=4.5,
            markerfacecolor=FEATURE_COLOR[feature_set],
            markeredgecolor="white",
            markeredgewidth=0.45,
            color=FEATURE_COLOR[feature_set],
            ecolor=FEATURE_COLOR[feature_set],
            elinewidth=1.0,
            capsize=2.0,
            label=FEATURE_LABEL[feature_set],
            zorder=3,
        )
    ax.set_yticks(y_base, DOMAIN_ORDER)
    ax.set_xlim(*xlim)
    ax.set_xlabel(title)
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.55)
    ax.set_axisbelow(True)


def make_figure_7() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    intervals = pd.read_csv(INTERVALS)
    paired = pd.read_csv(PAIRED)
    fused = predictions[
        predictions.feature_set == "conventional_alphaearth_embeddings"
    ].copy()

    fig = plt.figure(figsize=(13.0, 7.25), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.14, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.985,
        bottom=0.145,
        top=0.94,
        wspace=0.38,
        hspace=0.43,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    ax_roc = fig.add_subplot(grid[0, 1])
    ax_pr = fig.add_subplot(grid[0, 2])
    ax_delta = fig.add_subplot(grid[1, 1:])

    ax_map.set_facecolor("#F4F6F7")
    control = fused.label == 0
    positive = fused.label == 1
    ax_map.scatter(
        fused.loc[control, "longitude"],
        fused.loc[control, "latitude"],
        c=fused.loc[control, "stacked_score"],
        cmap="cividis",
        vmin=0,
        vmax=1,
        s=7,
        marker="x",
        linewidths=0.45,
        alpha=0.45,
        zorder=2,
    )
    scatter = ax_map.scatter(
        fused.loc[positive, "longitude"],
        fused.loc[positive, "latitude"],
        c=fused.loc[positive, "stacked_score"],
        cmap="cividis",
        vmin=0,
        vmax=1,
        s=12,
        marker="o",
        linewidths=0.25,
        edgecolors="white",
        alpha=0.78,
        zorder=3,
    )
    draw_domain_boundaries(ax_map, linewidth=0.85)
    ax_map.set_xlim(EXTENT[0], EXTENT[1])
    ax_map.set_ylim(EXTENT[2], EXTENT[3])
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_title("Fused scores for samples predicted only when their domain was held out", pad=7)
    ax_map.grid(color="#D9DEE3", linewidth=0.4, alpha=0.65)
    cbar = fig.colorbar(scatter, ax=ax_map, orientation="horizontal", fraction=0.035, pad=0.075)
    cbar.set_label("LODO case-control susceptibility score")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_map.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="#777777", markeredgecolor="white", label="Slope failure"),
            Line2D([], [], marker="x", linestyle="", markersize=5, color="#777777", label="Control"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        ncol=2,
        borderpad=0.45,
    )
    panel_label(ax_map, "(a)")

    plot_metric_intervals(ax_roc, intervals, "roc_auc", "Held-out ROC-AUC (95% block-bootstrap CI)", (0.76, 1.005))
    ax_roc.set_title("Cross-domain discrimination")
    panel_label(ax_roc, "(b)")
    plot_metric_intervals(ax_pr, intervals, "pr_auc", "Held-out PR-AUC (95% block-bootstrap CI)", (0.68, 1.005))
    ax_pr.set_title("Cross-domain precision-recall performance")
    ax_pr.set_yticklabels([])
    ax_pr.tick_params(axis="y", length=0)
    panel_label(ax_pr, "(c)")

    delta = paired[paired.comparison == "Fusion - Conventional"].copy()
    y_base = np.arange(len(DOMAIN_ORDER))[::-1]
    metric_style = {
        "roc_auc": (0.22, "o", "#2878B5", "ROC-AUC difference"),
        "pr_auc": (0.0, "s", "#1B9E77", "PR-AUC difference"),
        "brier": (-0.22, "D", "#D55E00", "Brier reduction"),
    }
    for metric, (offset, marker, color, label) in metric_style.items():
        estimates, lows, highs = [], [], []
        for domain in DOMAIN_ORDER:
            row = delta[(delta.held_out_domain == domain) & (delta.metric == metric)].iloc[0]
            estimate, lo, hi = row.difference_a_minus_b, row.ci95_low, row.ci95_high
            if metric == "brier":
                estimate, lo, hi = -estimate, -hi, -lo
            estimates.append(estimate)
            lows.append(estimate - lo)
            highs.append(hi - estimate)
        ax_delta.errorbar(
            estimates,
            y_base + offset,
            xerr=np.vstack([lows, highs]),
            fmt=marker,
            markersize=4.6,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=1.0,
            capsize=2,
            label=label,
        )
    ax_delta.axvline(0, color="#252A2E", linewidth=0.85, linestyle="--")
    ax_delta.set_yticks(y_base, DOMAIN_ORDER)
    ax_delta.set_xlim(-0.075, 0.085)
    ax_delta.set_xlabel("Fusion benefit relative to Conventional (paired 95% block-bootstrap CI)")
    ax_delta.set_title("Direction and uncertainty of the paired feature-set contrast")
    ax_delta.grid(axis="x", color="#D9DEE3", linewidth=0.55)
    ax_delta.set_axisbelow(True)
    metric_handles, metric_labels = ax_delta.get_legend_handles_labels()
    panel_label(ax_delta, "(d)")

    handles = [
        Line2D([], [], color=FEATURE_COLOR[feature_set], marker=FEATURE_MARKER[feature_set], linestyle="", markersize=5, label=FEATURE_LABEL[feature_set])
        for feature_set in FEATURE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.665, 0.995), ncol=3, frameon=False)
    fig.legend(
        metric_handles,
        metric_labels,
        loc="lower center",
        bbox_to_anchor=(0.72, 0.022),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    path = OUT / "Figure_7_comparative_lodo_transferability_v6"
    fig.savefig(path.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def raster_inside(path: Path, step: int = 4):
    with rasterio.open(path) as src:
        data = src.read(1)[::step, ::step]
        valid = np.isfinite(data) & (data != src.nodata)
        inside = np.ma.masked_where(~valid, (data <= 1).astype(float))
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return inside, extent


def make_figure_8() -> None:
    sensitivity = pd.read_csv(SENSITIVITY)
    raster_paths = {
        feature_set: RASTERS / f"cpec_baseline_{feature_set}_lodo_dissimilarity_index_250m.tif"
        for feature_set in FEATURE_ORDER
    }
    missing = [str(path) for path in raster_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LODO AoA rasters: " + ", ".join(missing))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.125, top=0.94, wspace=0.19, hspace=0.24)
    map_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    support_cmap = ListedColormap(["#D9766B", "#2A9D8F"])

    for panel_number, (feature_set, ax) in enumerate(zip(FEATURE_ORDER, map_axes)):
        inside, extent = raster_inside(raster_paths[feature_set])
        ax.imshow(
            inside,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            cmap=support_cmap,
            vmin=0,
            vmax=1,
            zorder=1,
        )
        draw_domain_boundaries(ax, linewidth=0.65, color="#263238")
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude (deg E)")
        ax.set_ylabel("Latitude (deg N)")
        ax.set_title(FEATURE_LABEL[feature_set], pad=6, fontweight="bold" if feature_set.endswith("embeddings") and feature_set.startswith("conventional_") else "normal")
        panel_label(ax, f"({chr(97 + panel_number)})")

    ax = axes[1, 1]
    y_base = np.arange(len(DOMAIN_ORDER))[::-1]
    offsets = {FEATURE_ORDER[0]: 0.20, FEATURE_ORDER[1]: 0.0, FEATURE_ORDER[2]: -0.20}
    for feature_set in FEATURE_ORDER:
        sub = sensitivity[sensitivity.feature_set == feature_set].set_index("held_out_domain")
        y = y_base + offsets[feature_set]
        standard = np.array([sub.loc[domain, "standard_aoa_coverage"] for domain in DOMAIN_ORDER])
        lower = np.array([sub.loc[domain, "min"] for domain in DOMAIN_ORDER])
        upper = np.array([sub.loc[domain, "max"] for domain in DOMAIN_ORDER])
        ax.errorbar(
            100 * standard,
            y,
            xerr=np.vstack([100 * (standard - lower), 100 * (upper - standard)]),
            fmt=FEATURE_MARKER[feature_set],
            markersize=5,
            color=FEATURE_COLOR[feature_set],
            markerfacecolor=FEATURE_COLOR[feature_set],
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=1.25,
            capsize=2.4,
            label=FEATURE_LABEL[feature_set],
        )
    ax.set_yticks(y_base, DOMAIN_ORDER)
    ax.set_xlim(0, 102)
    ax.set_xlabel("LODO AoA coverage (%)")
    ax.set_title("Sensitivity to latent dimension and threshold percentile", pad=6)
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.55)
    ax.set_axisbelow(True)
    feature_handles, feature_labels = ax.get_legend_handles_labels()
    panel_label(ax, "(d)")

    fig.legend(
        feature_handles,
        feature_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    fig.legend(
        handles=[
            Patch(facecolor="#2A9D8F", edgecolor="#263238", label="Inside transfer AoA (DI <= 1)"),
            Patch(facecolor="#D9766B", edgecolor="#263238", label="Outside transfer AoA (DI > 1)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.29, 0.035),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.76,
        0.044,
        "Points: 15 dimensions / P95; whiskers: range across 16 AoA settings",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#3F454A",
    )
    path = OUT / "Figure_8_transfer_aoa_pixels_and_sensitivity_v6"
    fig.savefig(path.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_figure_7()
    make_figure_8()
    print(OUT)


if __name__ == "__main__":
    main()
