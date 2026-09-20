"""Recompute slope and aspect from the exported Copernicus GLO-30 elevation band.

The earlier GEE mosaic lost the DEM tile projection before ``ee.Terrain`` was
called, producing a binary slope band and a coarsely quantised aspect band.
This script derives both variables on the 250 m analysis grid, samples them at
all modelling locations, and writes corrected copies without overwriting any
historical input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(r"D:\DING PROJECT")
BASE_STACK = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_gee_inputs_250m"
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
SOURCE_TABLE = (
    PROJECT_ROOT
    / "03_models"
    / "v3_admin_transferability_domain_tests"
    / "cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv"
)
OUT_DIR = PROJECT_ROOT / "01_clean_data" / "manuscript_v3_corrected_terrain"
OUT_RASTER = OUT_DIR / "cpec_copernicus_glo30_corrected_slope_aspect_250m.tif"
OUT_TABLE = OUT_DIR / "cpec_baseline_samples_corrected_terrain.csv"
QA_TABLE = OUT_DIR / "corrected_terrain_qa.csv"
WINDOW_SIZE = 1024


def fill_nearest(array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(array)
    if valid.all():
        return array
    if not valid.any():
        return array
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return array[tuple(nearest)]


def horn_slope_aspect(
    elevation: np.ndarray,
    latitude_degrees: np.ndarray,
    xres_degrees: float,
    yres_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = fill_nearest(elevation.astype("float64", copy=False))
    if not np.isfinite(z).any():
        shape = (max(z.shape[0] - 2, 0), max(z.shape[1] - 2, 0))
        return np.full(shape, np.nan, dtype="float32"), np.full(shape, np.nan, dtype="float32")

    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z6 = z[1:-1, :-2], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    metres_per_degree_lon = 111320.0 * np.clip(np.cos(np.deg2rad(latitude_degrees)), 0.2, None)
    dx = xres_degrees * metres_per_degree_lon[:, None]
    dy = yres_degrees * 110574.0
    dz_dx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * dx)
    dz_dy_north = ((z1 + 2.0 * z2 + z3) - (z7 + 2.0 * z8 + z9)) / (8.0 * dy)
    gradient = np.hypot(dz_dx, dz_dy_north)
    slope = np.rad2deg(np.arctan(gradient))
    aspect = np.mod(np.rad2deg(np.arctan2(-dz_dx, -dz_dy_north)), 360.0)
    aspect[gradient < 1e-8] = 0.0
    return slope.astype("float32"), aspect.astype("float32")


def expanded_window(window: Window, width: int, height: int) -> tuple[Window, tuple[int, int, int, int]]:
    row0 = max(int(window.row_off) - 1, 0)
    col0 = max(int(window.col_off) - 1, 0)
    row1 = min(int(window.row_off + window.height) + 1, height)
    col1 = min(int(window.col_off + window.width) + 1, width)
    expanded = Window(col0, row0, col1 - col0, row1 - row0)
    top = int(window.row_off) - row0
    left = int(window.col_off) - col0
    return expanded, (top, left, int(window.height), int(window.width))


def derive_raster() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(BASE_STACK) as src:
        if src.descriptions[0] != "elevation_m":
            raise ValueError("The first source band is not elevation_m.")
        profile = src.profile.copy()
        profile.update(
            count=2,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(OUT_RASTER, "w", **profile) as dst:
            dst.set_band_description(1, "slope_deg_corrected")
            dst.set_band_description(2, "aspect_deg_corrected")
            for row_off in range(0, src.height, WINDOW_SIZE):
                for col_off in range(0, src.width, WINDOW_SIZE):
                    height = min(WINDOW_SIZE, src.height - row_off)
                    width = min(WINDOW_SIZE, src.width - col_off)
                    window = Window(col_off, row_off, width, height)
                    expanded, (top, left, out_h, out_w) = expanded_window(window, src.width, src.height)
                    elevation = src.read(1, window=expanded, out_dtype="float32")
                    source_valid = np.isfinite(src.read(1, window=window, out_dtype="float32"))

                    # Add edge padding so every output cell has a 3 x 3 neighbourhood.
                    pad_top = 1 - top
                    pad_left = 1 - left
                    pad_bottom = 1 - (int(expanded.height) - (top + out_h))
                    pad_right = 1 - (int(expanded.width) - (left + out_w))
                    if any(v > 0 for v in [pad_top, pad_bottom, pad_left, pad_right]):
                        elevation = np.pad(
                            elevation,
                            (
                                (max(pad_top, 0), max(pad_bottom, 0)),
                                (max(pad_left, 0), max(pad_right, 0)),
                            ),
                            mode="edge",
                        )
                    lat_rows = np.arange(row_off, row_off + out_h, dtype=float) + 0.5
                    latitudes = src.transform.f + lat_rows * src.transform.e
                    slope, aspect = horn_slope_aspect(
                        elevation,
                        latitudes,
                        abs(src.transform.a),
                        abs(src.transform.e),
                    )
                    slope = slope[:out_h, :out_w]
                    aspect = aspect[:out_h, :out_w]
                    slope[~source_valid] = np.nan
                    aspect[~source_valid] = np.nan
                    dst.write(slope, 1, window=window)
                    dst.write(aspect, 2, window=window)
            dst.update_tags(
                source_dem="COPERNICUS/DEM/GLO30 elevation exported on the 250 m analysis grid",
                method="Horn 3x3 finite difference with latitude-adjusted east-west spacing",
                purpose="manuscript v3 correction of invalid binary slope and quantised aspect",
            )


def sample_corrected_table() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_TABLE, encoding="utf-8-sig")
    coords = list(zip(df["longitude"].astype(float), df["latitude"].astype(float)))
    with rasterio.open(OUT_RASTER) as src:
        sampled = np.asarray(list(src.sample(coords)), dtype=float)
    df["slope_deg_original_invalid"] = df["slope_deg"]
    df["aspect_deg_original_quantised"] = df["aspect_deg"]
    df["slope_deg"] = sampled[:, 0]
    df["aspect_deg"] = sampled[:, 1]
    missing = ~np.isfinite(df["slope_deg"]) | ~np.isfinite(df["aspect_deg"])
    if missing.any():
        raise RuntimeError(f"Corrected terrain is missing at {int(missing.sum())} sample locations.")
    df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    return df


def write_qa(df: pd.DataFrame) -> None:
    rows = []
    for factor in [
        "slope_deg_original_invalid",
        "slope_deg",
        "aspect_deg_original_quantised",
        "aspect_deg",
    ]:
        s = pd.to_numeric(df[factor], errors="coerce")
        rows.append(
            {
                "factor": factor,
                "n": s.notna().sum(),
                "unique_values": s.nunique(),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "median": s.median(),
                "p99": s.quantile(0.99),
                "max": s.max(),
            }
        )
    qa = pd.DataFrame(rows)
    qa.to_csv(QA_TABLE, index=False)
    manifest = {
        "source_stack": str(BASE_STACK),
        "source_table": str(SOURCE_TABLE),
        "corrected_raster": str(OUT_RASTER),
        "corrected_table": str(OUT_TABLE),
        "method": "Horn 3x3 finite difference on exported Copernicus GLO-30 elevation",
        "source_problem": "GEE terrain was computed after mosaicking without restoring native DEM projection",
        "historical_files_overwritten": False,
    }
    (OUT_DIR / "correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(qa.to_string(index=False))


def main() -> None:
    derive_raster()
    df = sample_corrected_table()
    write_qa(df)
    print(OUT_RASTER)
    print(OUT_TABLE)


if __name__ == "__main__":
    main()
