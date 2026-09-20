"""Generate a compact, journal-readable refinement of manuscript Figure 8."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


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
OUTPUT_STEM = OUTPUT_DIR / "Figure_8_harmonised_area_of_applicability_refined"

DOMAINS = [
    "Balochistan",
    "Gilgit-Baltistan",
    "KPK-AJK",
    "Kashgar",
    "Punjab-Sindh",
]
FEATURE_SETS = [
    "Conventional",
    "AlphaEarth Embeddings",
    "Conventional + AlphaEarth Embeddings",
]
PANEL_TITLES = [
    "(a)  Conventional",
    "(b)  AlphaEarth Embeddings",
    "(c)  Conventional + AlphaEarth Embeddings",
]
LABEL_POSITIONS = {
    "Balochistan": (64.2, 26.10),
    "Gilgit-Baltistan": (75.3, 35.75),
    "KPK-AJK": (72.0, 34.0),
    "Kashgar": (76.8, 39.55),
    "Punjab-Sindh": (69.2, 25.25),
}
LABEL_NAMES = {
    "Balochistan": "Balochistan",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "KPK-AJK": "KPK-AJK",
    "Kashgar": "Kashgar",
    "Punjab-Sindh": "Punjab-Sindh",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.labelsize": 9.7,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def contrasting_text(value: float) -> tuple[str, list[pe.AbstractPathEffect]]:
    if value < 0.58:
        return "white", [pe.withStroke(linewidth=1.35, foreground="#111111")]
    return "#111111", [pe.withStroke(linewidth=1.35, foreground="white")]


def style_map_axis(
    axis: plt.Axes,
    title_text: str,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    axis.set_xlim(59.8, 80.6)
    axis.set_ylim(22.9, 42.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title_text, loc="left", fontweight="bold", pad=5)
    axis.set_xlabel("Longitude (deg E)" if show_xlabel else "", labelpad=3)
    axis.set_ylabel("Latitude (deg N)" if show_ylabel else "", labelpad=3)
    axis.tick_params(direction="out", length=3.2, width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.95)


def build() -> None:
    configure()
    aoa = pd.read_csv(base.AOA_DIR / "harmonised_aoa_domain_summary.csv")
    shapes = base.domain_shape_map()
    color_map = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    figure = plt.figure(figsize=(8.2, 7.2), facecolor="white")
    outer = GridSpec(
        2,
        1,
        figure=figure,
        height_ratios=[1.0, 0.055],
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.955,
        hspace=0.22,
    )
    panels = outer[0].subgridspec(2, 2, wspace=0.17, hspace=0.24)
    map_axes = [
        figure.add_subplot(panels[0, 0]),
        figure.add_subplot(panels[0, 1]),
        figure.add_subplot(panels[1, 0]),
    ]
    heatmap_axis = figure.add_subplot(panels[1, 1])
    colorbar_axis = figure.add_subplot(outer[1])

    for map_index, (axis, feature_set, title_text) in enumerate(zip(
        map_axes,
        FEATURE_SETS,
        PANEL_TITLES,
    )):
        subset = aoa[aoa.feature_set_name == feature_set].set_index(
            "held_out_domain"
        )
        for domain in DOMAINS:
            value = float(subset.loc[domain, "aoa_coverage"])
            base.polygon_fill(
                axis,
                shapes[domain],
                color_map(norm(value)),
                edgecolor="white",
                alpha=1.0,
                lw=1.0,
                zorder=3,
            )
            x_position, y_position = LABEL_POSITIONS[domain]
            text_color, effects = contrasting_text(value)
            axis.text(
                x_position,
                y_position,
                f"{LABEL_NAMES[domain]}\n{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.25,
                fontweight="bold",
                linespacing=0.90,
                color=text_color,
                path_effects=effects,
                zorder=12,
            )
        base.plot_boundary(axis, lw=1.05)
        style_map_axis(
            axis,
            title_text,
            show_xlabel=map_index == 2,
            show_ylabel=map_index in (0, 2),
        )

    matrix = (
        aoa.pivot(
            index="held_out_domain",
            columns="feature_set_name",
            values="aoa_coverage",
        )
        .reindex(index=DOMAINS, columns=FEATURE_SETS)
        .to_numpy()
    )
    heatmap = heatmap_axis.imshow(
        matrix,
        cmap=color_map,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            text_color, effects = contrasting_text(value)
            heatmap_axis.text(
                column_index,
                row_index,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight="bold",
                color=text_color,
                path_effects=effects,
            )

    heatmap_axis.set_title(
        "(d)  Held-out-domain comparison",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    heatmap_axis.set_yticks(np.arange(len(DOMAINS)), DOMAINS)
    heatmap_axis.set_xticks(
        np.arange(len(FEATURE_SETS)),
        [
            "Conventional",
            "AlphaEarth\nEmbeddings",
            "Conventional +\nAlphaEarth\nEmbeddings",
        ],
    )
    heatmap_axis.set_xlabel("")
    heatmap_axis.set_ylabel("Held-out domain", labelpad=4)
    heatmap_axis.set_xticks(
        np.arange(-0.5, len(FEATURE_SETS), 1),
        minor=True,
    )
    heatmap_axis.set_yticks(
        np.arange(-0.5, len(DOMAINS), 1),
        minor=True,
    )
    heatmap_axis.grid(which="minor", color="white", linewidth=1.6)
    heatmap_axis.tick_params(which="minor", bottom=False, left=False)
    heatmap_axis.tick_params(axis="x", pad=4)
    for spine in heatmap_axis.spines.values():
        spine.set_linewidth(0.95)

    colorbar = figure.colorbar(heatmap, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label(
        "Harmonised area-of-applicability coverage",
        labelpad=4,
    )
    colorbar.ax.tick_params(labelsize=9.5, length=3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(figure)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    build()
