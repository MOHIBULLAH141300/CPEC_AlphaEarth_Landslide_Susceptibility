"""Generate a journal-refined Figure 7 without changing underlying results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_manuscript_v32_figures as base  # noqa: E402


OUTPUT_DIR = (
    base.PROJECT_ROOT
    / "FINAL PAPER"
    / "Manuscript_single_file"
    / "v4_1_2026-07-29"
    / "figures"
)
OUTPUT_STEM = OUTPUT_DIR / "Figure_7_subdomain_scores_and_transferability_refined"

FEATURE_SET = "conventional_alphaearth_embeddings"
DOMAIN_ORDER = [
    "Kashgar",
    "Gilgit-Baltistan",
    "KPK-AJK",
    "Balochistan",
    "Punjab-Sindh",
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def style_domain_map(
    ax: plt.Axes,
    domain: str,
    panel_letter: str,
    array: np.ndarray,
    extent: tuple[float, float, float, float],
    shape: object,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    image = ax.imshow(
        array,
        extent=extent,
        origin="upper",
        cmap="magma",
        norm=norm,
        interpolation="bilinear",
    )
    for part in base.shape_parts(shape):
        ax.plot(part[:, 0], part[:, 1], color="#202020", linewidth=1.05, zorder=10)

    xmin, xmax, ymin, ymax = base.shape_extent(shape)
    dx = max(xmax - xmin, 0.5)
    dy = max(ymax - ymin, 0.5)
    ax.set_xlim(xmin - 0.018 * dx, xmax + 0.018 * dx)
    ax.set_ylim(ymin - 0.018 * dy, ymax + 0.018 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"({panel_letter})  {domain}",
        loc="left",
        fontsize=10.8,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("Longitude (deg E)", labelpad=3)
    ax.set_ylabel("Latitude (deg N)", labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(direction="out", length=3.2, width=0.8)
    return image


def metric_arrays(
    fused_metrics: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    domains: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for domain in domains:
        estimate = float(
            fused_metrics.loc[fused_metrics.held_out_domain == domain, metric].iloc[0]
        )
        row = confidence_intervals[
            (confidence_intervals.feature_set == FEATURE_SET)
            & (confidence_intervals.held_out_domain == domain)
            & (confidence_intervals.metric == metric)
        ].iloc[0]
        estimates.append(estimate)
        lower_errors.append(estimate - float(row.ci95_low))
        upper_errors.append(float(row.ci95_high) - estimate)
    return (
        np.asarray(estimates),
        np.asarray(lower_errors),
        np.asarray(upper_errors),
    )


def build() -> None:
    configure()

    score_path = (
        base.MAP_DIR
        / "cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif"
    )
    score, extent, transform = base.downsample_raster(
        score_path, scale=6, return_transform=True
    )
    shapes = base.domain_shape_map()
    metrics = pd.read_csv(base.LODO_DIR / "leave_one_domain_out_metrics.csv")
    confidence_intervals = pd.read_csv(
        base.LODO_DIR / "leave_one_domain_out_block_bootstrap_ci.csv"
    )
    fused_metrics = metrics[metrics.feature_set == FEATURE_SET].copy()

    figure = plt.figure(figsize=(12.7, 7.6), facecolor="white")
    outer = GridSpec(
        1,
        2,
        figure=figure,
        width_ratios=[1.72, 1.0],
        left=0.06,
        right=0.975,
        bottom=0.12,
        top=0.93,
        wspace=0.22,
    )

    maps_grid = outer[0, 0].subgridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 0.085],
        hspace=0.37,
    )
    top_maps = maps_grid[0].subgridspec(1, 3, wspace=0.31)
    bottom_maps = maps_grid[1].subgridspec(1, 2, wspace=0.25)
    map_axes = [
        figure.add_subplot(top_maps[0, 0]),
        figure.add_subplot(top_maps[0, 1]),
        figure.add_subplot(top_maps[0, 2]),
        figure.add_subplot(bottom_maps[0, 0]),
        figure.add_subplot(bottom_maps[0, 1]),
    ]
    colorbar_axis = figure.add_subplot(maps_grid[2])
    transfer_axis = figure.add_subplot(outer[0, 1])

    norm = PowerNorm(gamma=0.45, vmin=0, vmax=1)
    image = None
    for index, (axis, domain) in enumerate(zip(map_axes, DOMAIN_ORDER)):
        shape = shapes[domain]
        clipped = base.clipped_domain_array(score, transform, shape)
        image = style_domain_map(
            axis,
            domain,
            chr(97 + index),
            clipped,
            extent,
            shape,
            norm,
        )

    assert image is not None
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label(
        "Fused case-control susceptibility score (0\u20131)",
        labelpad=5,
    )
    colorbar.ax.tick_params(labelsize=9, length=3)

    plot_domains = [
        "Punjab-Sindh",
        "Gilgit-Baltistan",
        "KPK-AJK",
        "Kashgar",
        "Balochistan",
    ]
    y_positions = np.arange(len(plot_domains))[::-1]
    metric_specs = [
        ("roc_auc", "ROC-AUC", "#0072B2", "o", 0.105),
        ("pr_auc", "PR-AUC", "#D55E00", "D", -0.105),
    ]
    for metric, label, color, marker, offset in metric_specs:
        estimates, lower, upper = metric_arrays(
            fused_metrics,
            confidence_intervals,
            plot_domains,
            metric,
        )
        transfer_axis.errorbar(
            estimates,
            y_positions + offset,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.2,
            capthick=1.25,
            label=label,
            zorder=4,
        )

    transfer_axis.set_title(
        "(f)  Leave-one-domain-out transfer performance",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )
    transfer_axis.set_yticks(y_positions, plot_domains)
    transfer_axis.set_ylim(-0.55, len(plot_domains) - 0.35)
    transfer_axis.set_xlim(0.50, 1.005)
    transfer_axis.xaxis.set_major_locator(MultipleLocator(0.10))
    transfer_axis.set_xlabel(
        "Held-out-domain score (95% block-bootstrap CI)",
        labelpad=7,
    )
    transfer_axis.grid(
        axis="x",
        color="#D9D9D9",
        linewidth=0.65,
        alpha=0.9,
        zorder=0,
    )
    transfer_axis.tick_params(direction="out", length=3.5, width=0.8)
    transfer_axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.018, 0.99),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        borderaxespad=0.2,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    plt.close(figure)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    build()
