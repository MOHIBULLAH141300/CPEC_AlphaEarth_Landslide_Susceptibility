"""Generate the refined manuscript Figure 4 without changing underlying results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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
OUTPUT_STEM = OUTPUT_DIR / "Figure_4_baseline_susceptibility_maps_refined"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "#333333",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=50,
    )


def style_map_axis(ax: plt.Axes, title: str, *, compact: bool = False) -> None:
    base.plot_boundary(ax, lw=1.15 if not compact else 1.0)
    ax.set_xlim(59.8, 80.6)
    ax.set_ylim(22.9, 42.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontweight="semibold", pad=7)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(direction="out", length=3.5, width=0.8)


def compact_north_arrow(ax: plt.Axes) -> None:
    x = 0.948
    ax.annotate(
        "",
        xy=(x, 0.925),
        xytext=(x, 0.845),
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "facecolor": "#111111",
            "edgecolor": "#111111",
            "linewidth": 1.1,
            "mutation_scale": 15,
        },
        zorder=40,
    )
    ax.text(
        x,
        0.938,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        zorder=40,
    )


def build() -> None:
    configure()
    paths = [
        base.MAP_DIR / "cpec_baseline_conventional_stacked_susceptibility_score_250m.tif",
        base.MAP_DIR
        / "cpec_baseline_alphaearth_embeddings_stacked_susceptibility_score_250m.tif",
        base.MAP_DIR
        / "cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif",
    ]
    arrays = []
    for path in paths:
        array, extent = base.downsample_raster(path)
        arrays.append(array)

    fig = plt.figure(figsize=(12.4, 7.3), facecolor="white")
    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.08, 1.0],
        left=0.055,
        right=0.975,
        bottom=0.135,
        top=0.94,
        wspace=0.19,
    )
    ax_fused = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(
        3,
        2,
        height_ratios=[1.0, 0.105, 0.72],
        hspace=0.42,
        wspace=0.28,
    )
    ax_conventional = fig.add_subplot(right[0, 0])
    ax_ae = fig.add_subplot(right[0, 1])
    cax = fig.add_subplot(right[1, :])
    ax_distribution = fig.add_subplot(right[2, :])

    score_norm = PowerNorm(gamma=0.45, vmin=0, vmax=1)
    map_axes = [ax_fused, ax_conventional, ax_ae]
    map_data = [arrays[2], arrays[0], arrays[1]]
    map_titles = [
        "Conventional + AlphaEarth Embeddings",
        "Conventional",
        "AlphaEarth Embeddings",
    ]

    image = None
    for index, (ax, data, title) in enumerate(zip(map_axes, map_data, map_titles)):
        image = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap="magma",
            norm=score_norm,
            interpolation="bilinear",
        )
        style_map_axis(ax, title, compact=index > 0)
        panel_label(ax, f"({chr(97 + index)})")

    base.plot_lines(ax_fused, base.KKH, "#00C8D7", lw=1.65, zorder=20)
    compact_north_arrow(ax_fused)
    base.scale_bar(ax_fused, 500, x=0.57, y=0.055)

    route_handle = Line2D(
        [0],
        [0],
        color="#00C8D7",
        lw=2.4,
        label="KKH proxy route (N35/314)",
    )
    fig.legend(
        handles=[route_handle],
        loc="lower center",
        bbox_to_anchor=(0.292, 0.035),
        frameon=False,
        handlelength=2.8,
    )

    assert image is not None
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label("Case-control susceptibility score (0–1)", labelpad=5)
    colorbar.ax.tick_params(labelsize=9, length=3)

    rng = np.random.default_rng(base.SEED)
    histogram_series = [
        ("Conventional", "Conventional", "-", arrays[0]),
        ("AlphaEarth Embeddings", "AlphaEarth", "--", arrays[1]),
        ("Conventional + AlphaEarth Embeddings", "Fused", "-.", arrays[2]),
    ]
    for name, label, linestyle, data in histogram_series:
        values = data[np.isfinite(data)]
        if len(values) > 120_000:
            values = rng.choice(values, size=120_000, replace=False)
        ax_distribution.hist(
            values,
            bins=np.linspace(0, 1, 61),
            density=True,
            histtype="step",
            linewidth=2.2,
            linestyle=linestyle,
            color=base.COLORS[name],
            label=label,
        )

    ax_distribution.set_yscale("log")
    ax_distribution.set_xlim(0, 1)
    ax_distribution.set_xlabel("Case-control susceptibility score")
    ax_distribution.set_ylabel("Density (log scale)")
    ax_distribution.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax_distribution.legend(
        frameon=False,
        loc="upper right",
        ncol=3,
        columnspacing=1.2,
        handlelength=2.3,
    )
    ax_distribution.tick_params(direction="out", length=3.5, width=0.8)
    panel_label(ax_distribution, "(d)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    plt.close(fig)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    build()
