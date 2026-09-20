"""Generate the refined, publication-ready study-area Figure 1 for manuscript V4."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import ee
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_manuscript_v32_figures as base  # noqa: E402


OUTPUT_DIR = (
    base.PROJECT_ROOT
    / "FINAL PAPER"
    / "Manuscript_single_file"
    / "v4_2026-07-28"
    / "figures"
)
OUTPUT_STEM = OUTPUT_DIR / "Figure_1_study_area_refined"
SATELLITE_CACHE = (
    base.PROJECT_ROOT
    / "01_clean_data"
    / "cartographic_reference"
    / "modis_2018_cpec_context"
    / "modis_2018_multidate_truecolour_context.png"
)

DOMAIN_COLORS = {
    "Balochistan": "#B07AA1",
    "Gilgit-Baltistan": "#59A14F",
    "KP-AJK": "#F28E2B",
    "Kashgar (Xinjiang, China)": "#4E79A7",
    "Punjab-Sindh lowland corridor": "#EDC948",
}

DOMAIN_DISPLAY = {
    "Balochistan": "Balochistan",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "KP-AJK": "KPK-AJK",
    "Kashgar (Xinjiang, China)": "Kashgar",
    "Punjab-Sindh lowland corridor": "Punjab-Sindh",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 500,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.13",
            "facecolor": "white",
            "edgecolor": "#444444",
            "linewidth": 0.75,
            "alpha": 0.94,
        },
        zorder=80,
    )


def neutral_relief(ax: plt.Axes, data: np.ndarray, extent: tuple[float, ...]):
    valid_mask = np.isfinite(data)
    valid = np.where(valid_mask, data, np.nanmedian(data))
    hillshade = LightSource(azdeg=315, altdeg=42).hillshade(
        valid,
        vert_exag=1.25,
        dx=1,
        dy=1,
    )
    hillshade = np.where(valid_mask, hillshade, np.nan)
    elevation = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="Greys",
        vmin=0,
        vmax=8000,
        alpha=0.72,
        interpolation="bilinear",
        zorder=0,
    )
    ax.imshow(
        hillshade,
        extent=extent,
        origin="upper",
        cmap="gray",
        alpha=0.25,
        interpolation="bilinear",
        zorder=1,
    )
    return elevation


def satellite_context(extent: tuple[float, ...]) -> np.ndarray:
    """Return a subdued, cloud-screened 2018 true-colour Earth Engine context."""
    if not SATELLITE_CACHE.exists():
        SATELLITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ee.Initialize(project="ee-mohibullah141300")
        west, east, south, north = extent
        region = ee.Geometry.Rectangle(
            [float(west), float(south), float(east), float(north)],
            geodesic=False,
        )

        images: list[ee.Image] = []
        for image_id in [
            "2018_05_25",
            "2018_07_12",
            "2018_09_06",
            "2018_10_16",
        ]:
            image = ee.Image(f"MODIS/061/MOD09A1/{image_id}")
            state_qa = image.select("StateQA")
            cloud_free = state_qa.bitwiseAnd(3).eq(0)
            shadow_free = state_qa.rightShift(2).bitwiseAnd(1).eq(0)
            images.append(
                image.updateMask(cloud_free.And(shadow_free))
                .select(["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"])
                .multiply(0.0001)
            )

        composite = ee.ImageCollection.fromImages(images).median()
        visual = composite.visualize(
            bands=["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
            min=0.02,
            max=0.35,
            gamma=1.15,
        )
        url = visual.getThumbURL(
            {
                "region": region,
                "dimensions": "1800x1653",
                "crs": "EPSG:4326",
                "format": "png",
            }
        )
        urllib.request.urlretrieve(url, SATELLITE_CACHE)

    image = plt.imread(SATELLITE_CACHE)[..., :3]
    luminance = np.sum(image * np.array([0.2126, 0.7152, 0.0722]), axis=2)
    subdued = 0.82 * image + 0.18 * luminance[..., np.newaxis]
    subdued[luminance < 0.035] = np.array([0.18, 0.27, 0.31])
    return np.clip(subdued, 0, 1)


def plot_subdomains(ax: plt.Axes) -> None:
    boundary_segments: list[np.ndarray] = []
    for shape, attributes in base.shapefile_records(base.DOMAINS):
        domain = attributes["domain"]
        for part in base.shape_parts(shape):
            if len(part) < 3:
                continue
            ax.fill(
                part[:, 0],
                part[:, 1],
                facecolor=DOMAIN_COLORS[domain],
                edgecolor="none",
                alpha=0.44,
                zorder=3,
            )
            boundary_segments.append(part)

    ax.add_collection(
        LineCollection(
            boundary_segments,
            colors="white",
            linewidths=2.4,
            alpha=0.95,
            zorder=4,
        )
    )
    ax.add_collection(
        LineCollection(
            boundary_segments,
            colors="#252525",
            linewidths=0.85,
            alpha=0.95,
            zorder=5,
        )
    )


def north_arrow_in_clear_space(ax: plt.Axes) -> None:
    x, y = 0.068, 0.855
    ax.annotate(
        "",
        xy=(x, y + 0.060),
        xytext=(x, y - 0.045),
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "facecolor": "#111111",
            "edgecolor": "#111111",
            "linewidth": 1.2,
            "mutation_scale": 16,
        },
        zorder=70,
    )
    ax.text(
        x,
        y + 0.076,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
        zorder=70,
    )


def add_legends(fig: plt.Figure) -> None:
    subdomain_handles = [
        Patch(
            facecolor=DOMAIN_COLORS[domain],
            edgecolor="#252525",
            linewidth=0.8,
            alpha=0.72,
            label=DOMAIN_DISPLAY[domain],
        )
        for domain in DOMAIN_COLORS
    ]
    feature_handles = [
        Line2D([0], [0], color="#111111", lw=1.7, label="Official CPEC boundary"),
        Line2D([0], [0], color="#FFD21F", lw=3.0, label="KKH proxy route (N35/314)"),
        Line2D([0], [0], color="#9C2F12", lw=1.8, label="Active faults"),
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            ms=5.5,
            mfc="#D73027",
            mec="white",
            mew=0.75,
            label="Unified slope-failure inventory",
        ),
    ]

    first = fig.legend(
        handles=subdomain_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.075),
        handlelength=2.0,
        columnspacing=1.5,
        handletextpad=0.55,
        borderpad=0.45,
    )
    first.get_frame().set_edgecolor("#A8A8A8")
    first.get_frame().set_linewidth(0.75)
    first.get_frame().set_facecolor("white")
    first.get_frame().set_alpha(0.97)

    second = fig.legend(
        handles=feature_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=2.6,
        columnspacing=1.45,
        handletextpad=0.55,
        borderpad=0.45,
    )
    second.get_frame().set_edgecolor("#A8A8A8")
    second.get_frame().set_linewidth(0.75)
    second.get_frame().set_facecolor("white")
    second.get_frame().set_alpha(0.97)


def build() -> None:
    configure()
    dem, extent = base.read_dem()
    satellite = satellite_context(extent)
    inventory_x, inventory_y = base.inventory_points()

    fig = plt.figure(figsize=(13.2, 7.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.94, 1.43],
        height_ratios=[0.67, 1.0],
        left=0.045,
        right=0.925,
        bottom=0.175,
        top=0.975,
        hspace=0.022,
        wspace=0.085,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[:, 1])

    world_extent = (-180, 180, -58, 88)
    ax_a.set_facecolor("#DDECF4")
    base.plot_countries(
        ax_a,
        world_extent,
        highlight={"Pakistan": "#D95F59", "China": "#F2A65A"},
    )
    ax_a.add_patch(
        Rectangle(
            (45, 20),
            50,
            30,
            fill=False,
            ec="#B2182B",
            lw=1.7,
            zorder=10,
        )
    )
    ax_a.set_xlim(world_extent[0], world_extent[1])
    ax_a.set_ylim(world_extent[2], world_extent[3])
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_anchor("S")
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    regional_extent = (44, 96, 19, 51)
    ax_b.set_facecolor("#DDECF4")
    base.plot_countries(
        ax_b,
        regional_extent,
        highlight={"Pakistan": "#F6D7D5", "China": "#F9E2C3"},
        labels=True,
    )
    base.plot_boundary(ax_b, lw=1.8, color="#B2182B", zorder=12)
    ax_b.add_patch(
        Rectangle(
            (59.8, 22.9),
            20.8,
            19.1,
            fill=False,
            ec="#222222",
            lw=1.0,
            ls="--",
            zorder=11,
        )
    )
    ax_b.set_xlim(regional_extent[0], regional_extent[1])
    ax_b.set_ylim(regional_extent[2], regional_extent[3])
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_anchor("N")
    ax_b.set_xlabel("Longitude (°E)")
    ax_b.set_ylabel("Latitude (°N)")
    ax_b.grid(color="white", lw=0.5, alpha=0.75)

    ax_c.imshow(
        satellite,
        extent=extent,
        origin="upper",
        alpha=0.76,
        interpolation="bilinear",
        zorder=-2,
    )
    elevation = neutral_relief(ax_c, dem, extent)
    plot_subdomains(ax_c)
    base.plot_lines(ax_c, base.FAULTS, "#9C2F12", lw=1.05, alpha=0.96, zorder=9)
    base.plot_lines(ax_c, base.KKH, "#FFD21F", lw=2.2, alpha=1.0, zorder=12)
    ax_c.scatter(
        inventory_x,
        inventory_y,
        s=10,
        color="#D73027",
        edgecolor="white",
        linewidth=0.42,
        alpha=0.93,
        zorder=13,
    )
    base.plot_boundary(ax_c, lw=2.0, color="#111111", zorder=15)
    ax_c.set_xlim(59.8, 80.6)
    ax_c.set_ylim(22.9, 42.0)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel("Longitude (°E)")
    ax_c.set_ylabel("Latitude (°N)")
    ax_c.grid(color="white", lw=0.38, alpha=0.55)

    panel_label(ax_a, "(a)")
    panel_label(ax_b, "(b)")
    panel_label(ax_c, "(c)")
    north_arrow_in_clear_space(ax_c)
    base.scale_bar(ax_c, 500, x=0.60, y=0.050)

    city_style = {
        "fontsize": 9.4,
        "fontweight": "bold",
        "color": "#111111",
        "path_effects": [pe.withStroke(linewidth=3.2, foreground="white")],
        "zorder": 30,
    }
    cities = [
        (75.99, 39.47, "Kashgar", 0.12, 0.10),
        (74.31, 35.92, "Gilgit", 0.12, 0.10),
        (73.05, 33.68, "Islamabad", 0.12, 0.10),
        (66.99, 30.18, "Quetta", 0.12, 0.10),
        (67.01, 24.86, "Karachi", 0.12, 0.12),
    ]
    for x, y, name, dx, dy in cities:
        ax_c.plot(
            x,
            y,
            marker="s",
            ms=4.6,
            mfc="white",
            mec="black",
            mew=0.65,
            zorder=29,
        )
        ax_c.text(x + dx, y + dy, name, **city_style)

    color_axis = fig.add_axes([0.938, 0.305, 0.015, 0.390])
    colorbar = fig.colorbar(elevation, cax=color_axis, orientation="vertical")
    colorbar.set_label("Elevation (m)", labelpad=7)
    colorbar.ax.tick_params(labelsize=8.8)

    add_legends(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )
    plt.close(fig)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    build()
