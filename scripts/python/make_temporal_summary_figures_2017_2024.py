"""Create visual QA/publication draft figures for temporal summary outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling


PROJECT_ROOT = Path(r"D:\DING PROJECT")
ANNUAL = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
OUT = ANNUAL / "00_multi_year_summary_outputs"
FIG = OUT / "02_figures"
YEARS = list(range(2017, 2025))
NODATA = -9999.0


def read_raster(path: Path, factor: int = 8, resampling=Resampling.average):
    with rasterio.open(path) as src:
        h = max(1, src.height // factor)
        w = max(1, src.width // factor)
        arr = src.read(1, out_shape=(h, w), resampling=resampling).astype("float32")
        nodata = src.nodata if src.nodata is not None else NODATA
        arr[(arr == nodata) | ~np.isfinite(arr)] = np.nan
        bounds = src.bounds
    return arr, [bounds.left, bounds.right, bounds.bottom, bounds.top]


def fused_path(year: int) -> Path:
    return (
        ANNUAL
        / str(year)
        / "02_probability_maps_250m"
        / f"cpec_{year}_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"
    )


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#666666")


def annual_probability_figure() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    ims = []
    for ax, year in zip(axes.ravel(), YEARS):
        arr, extent = read_raster(fused_path(year), factor=8)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap="magma", vmin=0, vmax=1)
        ims.append(im)
        ax.set_title(str(year), fontsize=11, fontweight="bold")
        clean_axes(ax)
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    cbar.set_label("Fused susceptibility probability")
    fig.suptitle(
        "Annual Dynamic Susceptibility, 2017-2024\n"
        "Fixed 2018 Spatial-CV Stacked Ensemble with annual predictors",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(FIG / "figure_annual_fused_probability_2017_2024.png", dpi=300)
    plt.close(fig)


def temporal_summary_figure() -> None:
    items = [
        (
            OUT / "cpec_2017_2024_fused_mean_probability_250m.tif",
            "Mean probability",
            "magma",
            0,
            1,
            Resampling.average,
        ),
        (
            OUT / "cpec_2017_2024_fused_temporal_std_probability_250m.tif",
            "Temporal variability (std)",
            "viridis",
            0,
            None,
            Resampling.average,
        ),
        (
            OUT / "cpec_2017_2024_fused_theilsen_trend_slope_per_year_250m.tif",
            "Theil-Sen trend slope / year",
            "RdBu_r",
            -0.03,
            0.03,
            Resampling.average,
        ),
        (
            OUT / "cpec_2017_2024_fused_persistent_high_probability_mask_p80_count_ge6_250m.tif",
            "Persistent high susceptibility",
            "Greys",
            0,
            1,
            Resampling.nearest,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (path, title, cmap, vmin, vmax, resampling) in zip(axes.ravel(), items):
        arr, extent = read_raster(path, factor=8, resampling=resampling)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        clean_axes(ax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.01)
        if "Persistent" in title:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["No", "Yes"])
    fig.suptitle("Temporal Summary of Fused Susceptibility, 2017-2024", fontsize=14, fontweight="bold")
    fig.savefig(FIG / "figure_temporal_summary_fused_2017_2024.png", dpi=300)
    plt.close(fig)


def change_figure() -> None:
    pairs = list(zip(YEARS[:-1], YEARS[1:]))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    last_im = None
    for ax, (prev, curr) in zip(axes, pairs):
        path = OUT / "01_year_to_year_change_maps" / f"cpec_{curr}_minus_{prev}_fused_probability_change_250m.tif"
        arr, extent = read_raster(path, factor=8)
        last_im = ax.imshow(arr, extent=extent, origin="upper", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        ax.set_title(f"{curr} - {prev}", fontsize=11, fontweight="bold")
        clean_axes(ax)
    axes[-1].axis("off")
    cbar = fig.colorbar(last_im, ax=axes[:-1].tolist(), shrink=0.82, pad=0.012)
    cbar.set_label("Probability change")
    fig.suptitle("Year-to-Year Fused Susceptibility Change", fontsize=14, fontweight="bold")
    fig.savefig(FIG / "figure_year_to_year_fused_probability_change_2017_2024.png", dpi=300)
    plt.close(fig)


def main() -> None:
    annual_probability_figure()
    temporal_summary_figure()
    change_figure()
    print(FIG)


if __name__ == "__main__":
    main()
