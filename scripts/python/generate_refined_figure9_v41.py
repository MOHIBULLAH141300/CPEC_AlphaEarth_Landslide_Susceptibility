"""Generate a decision-focused refined Figure 9 from verified project outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
OUTPUT_STEM = OUTPUT_DIR / "Figure_9_road_exposure_and_ablation_refined"

PRIORITY_COLORS = {
    "below_p90": "#4D4D4D",
    "supported_high_score": "#0072B2",
    "verification_priority_high_score": "#E64B00",
}
PRIORITY_LABELS = {
    "below_p90": "Below map P90",
    "supported_high_score": "Supported high score",
    "verification_priority_high_score": "Verification priority",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def title(axis: plt.Axes, panel: str, text: str) -> None:
    axis.set_title(
        f"({panel})  {text}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=7,
    )


def road_segments(data: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            data[["start_lon", "start_lat"]].to_numpy(),
            data[["end_lon", "end_lat"]].to_numpy(),
        ],
        axis=1,
    )


def route_display_label(value: object) -> str:
    label = str(value)
    if label == "N35":
        return "N35 (KKH)"
    if label == "314":
        return "314 (KKH proxy)"
    return label


def build() -> None:
    configure()
    segments = pd.read_csv(base.ROAD_DIR / "road_segment_scores_250m.csv")
    summary = pd.read_csv(base.ROAD_DIR / "road_network_exposure_summary.csv")
    routes = pd.read_csv(base.ROAD_DIR / "named_route_exposure_ranking.csv").head(10)
    routes = routes.sort_values("actual_length_ge_p90_km", ascending=False).reset_index(
        drop=True
    )

    dem, dem_extent = base.read_dem()
    all_line_segments = road_segments(segments)
    score_norm = PowerNorm(gamma=0.45, vmin=0, vmax=1)

    figure = plt.figure(figsize=(12.7, 8.2), facecolor="white")
    grid = GridSpec(
        3,
        2,
        figure=figure,
        width_ratios=[1.52, 1.0],
        height_ratios=[1.08, 0.76, 0.92],
        left=0.06,
        right=0.975,
        bottom=0.105,
        top=0.94,
        wspace=0.29,
        hspace=0.38,
    )
    network_axis = figure.add_subplot(grid[0:2, 0])
    kkh_axis = figure.add_subplot(grid[0, 1])
    share_axis = figure.add_subplot(grid[1, 1])
    bottom = grid[2, :].subgridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.17)
    route_length_axis = figure.add_subplot(bottom[0, 0])
    ablation_axis = figure.add_subplot(bottom[0, 1], sharey=route_length_axis)

    # (a) Road-segment susceptibility over a subdued terrain context.
    base.terrain_background(network_axis, dem, dem_extent)
    network_axis.add_patch(
        Rectangle(
            (dem_extent[0], dem_extent[2]),
            dem_extent[1] - dem_extent[0],
            dem_extent[3] - dem_extent[2],
            facecolor="white",
            edgecolor="none",
            alpha=0.43,
            zorder=2,
        )
    )
    base.plot_domains(network_axis, alpha=0.055)
    network_axis.add_collection(
        LineCollection(
            all_line_segments,
            colors="white",
            linewidths=0.95,
            alpha=0.30,
            zorder=9,
        )
    )
    score_collection = LineCollection(
        all_line_segments,
        cmap="magma",
        norm=score_norm,
        linewidths=0.62,
        alpha=0.82,
        zorder=10,
    )
    score_collection.set_array(segments.fused_score.to_numpy())
    network_axis.add_collection(score_collection)
    kkh_segments = segments[segments.is_kkh_proxy.astype(bool)]
    kkh_lines = road_segments(kkh_segments)
    network_axis.add_collection(
        LineCollection(
            kkh_lines,
            colors="#202020",
            linewidths=3.0,
            alpha=0.88,
            zorder=12,
        )
    )
    network_axis.add_collection(
        LineCollection(
            kkh_lines,
            colors="#00D8FF",
            linewidths=1.75,
            alpha=1.0,
            zorder=13,
        )
    )
    base.plot_boundary(network_axis, lw=0.9)
    network_axis.set_xlim(59.8, 80.6)
    network_axis.set_ylim(22.9, 42.0)
    network_axis.set_aspect("equal")
    network_axis.set_anchor("N")
    network_axis.set_xlabel("Longitude (deg E)")
    network_axis.set_ylabel("Latitude (deg N)")
    network_axis.tick_params(direction="out", length=3.2, width=0.8)
    title(network_axis, "a", "CPEC road-segment susceptibility")
    network_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#00D8FF",
                linewidth=3.0,
                path_effects=[],
                label="KKH proxy route (N35/314)",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.018, 0.982),
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.94,
        borderpad=0.5,
        handlelength=2.7,
    )
    colorbar = figure.colorbar(
        score_collection,
        ax=network_axis,
        fraction=0.041,
        pad=0.024,
    )
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label("Case-control road score (0\u20131)", labelpad=6)
    colorbar.ax.tick_params(labelsize=9, length=3)

    # (b) Decision classes along the KKH proxy.
    base.terrain_background(kkh_axis, dem, dem_extent)
    kkh_axis.add_patch(
        Rectangle(
            (dem_extent[0], dem_extent[2]),
            dem_extent[1] - dem_extent[0],
            dem_extent[3] - dem_extent[2],
            facecolor="white",
            edgecolor="none",
            alpha=0.47,
            zorder=2,
        )
    )
    kkh_axis.add_collection(
        LineCollection(
            kkh_lines,
            colors="white",
            linewidths=4.6,
            alpha=0.92,
            zorder=7,
        )
    )
    draw_order = [
        "below_p90",
        "verification_priority_high_score",
        "supported_high_score",
    ]
    for priority in draw_order:
        subset = kkh_segments[kkh_segments.priority_class == priority]
        if subset.empty:
            continue
        kkh_axis.add_collection(
            LineCollection(
                road_segments(subset),
                colors=PRIORITY_COLORS[priority],
                linewidths=2.0 if priority == "below_p90" else 3.15,
                alpha=0.95,
                zorder=8 if priority == "below_p90" else 12,
            )
        )
    kkh_axis.set_xlim(72.3, 86.4)
    kkh_axis.set_ylim(33.2, 40.7)
    kkh_axis.set_aspect("equal")
    kkh_axis.set_anchor("C")
    kkh_axis.set_xlabel("Longitude (deg E)")
    kkh_axis.set_ylabel("Latitude (deg N)")
    kkh_axis.set_xticks(np.arange(74, 87, 2))
    kkh_axis.tick_params(direction="out", length=3.2, width=0.8)
    title(kkh_axis, "b", "KKH segment priority")
    priority_handles = [
        Line2D(
            [0],
            [0],
            color=PRIORITY_COLORS[key],
            linewidth=2.4,
            label=PRIORITY_LABELS[key],
        )
        for key in [
            "below_p90",
            "supported_high_score",
            "verification_priority_high_score",
        ]
    ]
    kkh_axis.legend(
        handles=priority_handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.50),
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.92,
        borderpad=0.55,
        labelspacing=0.5,
        handlelength=2.4,
    )

    # (c) Relative exposure of the full network and KKH proxy.
    groups = ["Full CPEC road network", "KKH proxy route (N35/314)"]
    exposure = summary.set_index("road_group").reindex(groups)
    total_length = exposure.total_length_km.to_numpy()
    p80_share = 100 * exposure.actual_length_ge_p80_km.to_numpy() / total_length
    p90_share = 100 * exposure.actual_length_ge_p90_km.to_numpy() / total_length
    y_share = np.arange(2)[::-1]
    for y_value, p80, p90 in zip(y_share, p80_share, p90_share):
        share_axis.plot(
            [p90, p80],
            [y_value, y_value],
            color="#A9A9A9",
            linewidth=3,
            solid_capstyle="round",
            zorder=1,
        )
    share_axis.scatter(
        p80_share,
        y_share,
        s=72,
        color="#E69F00",
        edgecolor="white",
        linewidth=0.7,
        label="\u2265 P80",
        zorder=3,
    )
    share_axis.scatter(
        p90_share,
        y_share,
        s=72,
        color="#D55E00",
        marker="D",
        edgecolor="white",
        linewidth=0.7,
        label="\u2265 P90",
        zorder=3,
    )
    for x_value, y_value in zip(p80_share, y_share):
        share_axis.text(
            x_value + 1.8,
            y_value + 0.12,
            f"{x_value:.1f}%",
            color="#A06600",
            fontsize=8.3,
            ha="left",
            va="bottom",
        )
    for x_value, y_value in zip(p90_share, y_share):
        share_axis.text(
            x_value + 1.8,
            y_value - 0.12,
            f"{x_value:.1f}%",
            color="#A13E00",
            fontsize=8.3,
            ha="left",
            va="top",
        )
    share_axis.set_yticks(y_share, ["Full CPEC network", "KKH proxy"])
    share_axis.set_ylabel("")
    share_axis.set_xlim(0, 108)
    share_axis.set_ylim(-0.45, 1.45)
    share_axis.set_xlabel(
        "Road length exceeding susceptibility threshold (%)",
        labelpad=5,
    )
    share_axis.xaxis.set_major_locator(MultipleLocator(20))
    share_axis.grid(axis="x", color="#DEDEDE", linewidth=0.6, zorder=0)
    share_axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    share_axis.tick_params(direction="out", length=3.2, width=0.8)
    title(share_axis, "c", "Road exposure above score thresholds")

    # (d) Route-code ranking by exposed length.
    y_routes = np.arange(len(routes))
    exposed_length = routes.actual_length_ge_p90_km.to_numpy()
    for y_value, length in zip(y_routes, exposed_length):
        route_length_axis.plot(
            [0, length],
            [y_value, y_value],
            color="#B5C8DA",
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
    route_length_axis.scatter(
        exposed_length,
        y_routes,
        s=55,
        color="#3F78A8",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    route_length_axis.set_yticks(
        y_routes,
        [route_display_label(value) for value in routes.route_label],
    )
    route_length_axis.invert_yaxis()
    route_length_axis.set_ylim(len(routes) - 0.5, -1.0)
    route_length_axis.set_xlim(0, float(exposed_length.max()) * 1.12)
    route_length_axis.set_xlabel("Road length at or above map P90 (km)")
    route_length_axis.set_ylabel("Named route code")
    route_length_axis.grid(axis="x", color="#DEDEDE", linewidth=0.6, zorder=0)
    route_length_axis.tick_params(direction="out", length=3.2, width=0.8)
    title(route_length_axis, "d", "Named-route exposure above P90")

    # (e) Paired full-model and no-road-distance scores.
    full_score = routes.length_weighted_mean_score.to_numpy()
    no_road_score = routes.length_weighted_mean_score_no_road_distance.to_numpy()
    for y_value, full_value, ablated_value in zip(
        y_routes,
        full_score,
        no_road_score,
    ):
        ablation_axis.plot(
            [ablated_value, full_value],
            [y_value, y_value],
            color="#B8B8B8",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
    ablation_axis.scatter(
        full_score,
        y_routes,
        s=55,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.7,
        label="Full model",
        zorder=3,
    )
    ablation_axis.scatter(
        no_road_score,
        y_routes,
        s=55,
        color="#D55E00",
        marker="D",
        edgecolor="white",
        linewidth=0.7,
        label="Without road distance",
        zorder=3,
    )
    ablation_axis.set_xlim(0, 1.0)
    ablation_axis.set_xlabel("Length-weighted mean susceptibility score")
    ablation_axis.grid(axis="x", color="#DEDEDE", linewidth=0.6, zorder=0)
    ablation_axis.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )
    ablation_axis.tick_params(axis="x", direction="out", length=3.2, width=0.8)
    ablation_axis.spines["left"].set_visible(True)
    ablation_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.45,
    )
    title(ablation_axis, "e", "Effect of removing road-distance predictor")

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
