from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_NETWORK", "OFF")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import datadir
from matplotlib.colors import TwoSlopeNorm
from rasterio.enums import Resampling

datadir.set_data_dir(os.environ["PROJ_DATA"])


PROJECT_ROOT = Path(r"D:\DING PROJECT")
SCENARIO_DIR = (
    PROJECT_ROOT
    / "04_maps"
    / "annual_dynamic_2017_2024"
    / "2018"
    / "03_scenario_maps_250m"
    / "seismic_rainfall"
)
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures" / "seismic_rainfall_scenarios_2018"
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)

PROBABILITY_MAPS = [
    (
        "Baseline",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_baseline_probability_250m.tif",
    ),
    (
        "Heavy rainfall",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_heavy_rainfall_stress_probability_250m.tif",
    ),
    (
        "Strong earthquake",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_strong_earthquake_stress_probability_250m.tif",
    ),
    (
        "Compound",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_compound_heavy_rainfall_strong_earthquake_probability_250m.tif",
    ),
]

CHANGE_MAPS = [
    (
        "Heavy rainfall minus baseline",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_heavy_rainfall_minus_baseline_probability_250m.tif",
    ),
    (
        "Strong earthquake minus baseline",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_strong_earthquake_minus_baseline_probability_250m.tif",
    ),
    (
        "Compound minus baseline",
        SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_compound_minus_baseline_probability_250m.tif",
    ),
]


def load_boundary() -> gpd.GeoDataFrame:
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:32642")
    return boundary.to_crs("EPSG:4326")


def read_display_raster(path: Path, max_side: int = 1800) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        factor = max(1, int(np.ceil(max(src.width, src.height) / max_side)))
        height = int(np.ceil(src.height / factor))
        width = int(np.ceil(src.width / factor))
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average,
        )
        if src.nodata is not None:
            data = np.ma.masked_where(np.isclose(data, src.nodata), data)
        bounds = src.bounds
    return data, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def finish_map_axis(ax, boundary: gpd.GeoDataFrame) -> None:
    boundary.boundary.plot(ax=ax, color="#111827", linewidth=0.6)
    ax.set_xlim(60.6, 80.2)
    ax.set_ylim(23.4, 41.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#d1d5db", linewidth=0.35, alpha=0.65)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#374151")


def make_probability_figure(boundary: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.2), constrained_layout=True)
    image = None
    for ax, (title, path) in zip(axes.ravel(), PROBABILITY_MAPS):
        data, extent = read_display_raster(path)
        image = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        finish_map_axis(ax, boundary)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Susceptibility probability")
    fig.suptitle(
        "2018 fused seismic-rainfall scenario susceptibility maps",
        fontsize=14,
        fontweight="bold",
    )
    out_png = FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_probability_maps.png"
    out_pdf = FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_probability_maps.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def robust_change_limit() -> float:
    values = []
    for _, path in CHANGE_MAPS:
        data, _ = read_display_raster(path, max_side=1400)
        valid = data.compressed()
        if valid.size:
            values.append(valid)
    all_values = np.concatenate(values)
    q = np.nanpercentile(np.abs(all_values), 99)
    return float(max(0.05, min(0.25, q)))


def make_change_figure(boundary: gpd.GeoDataFrame) -> None:
    limit = robust_change_limit()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), constrained_layout=True)
    image = None
    for ax, (title, path) in zip(axes.ravel(), CHANGE_MAPS):
        data, extent = read_display_raster(path)
        image = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        finish_map_axis(ax, boundary)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, pad=0.02)
    cbar.set_label(f"Probability change (clipped to +/-{limit:.2f})")
    fig.suptitle(
        "Scenario-driven change in 2018 fused susceptibility",
        fontsize=14,
        fontweight="bold",
    )
    out_png = FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_change_maps.png"
    out_pdf = FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_change_maps.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    make_probability_figure(boundary)
    make_change_figure(boundary)
    print(FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_probability_maps.png")
    print(FIG_DIR / "figure_2018_fused_seismic_rainfall_scenario_change_maps.png")


if __name__ == "__main__":
    main()
