from __future__ import annotations

import csv
import math
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(r"D:\DING PROJECT")
YEARS = list(range(2017, 2025))
NODATA = -9999.0

TEMPLATE_RASTER = PROJECT_ROOT / (
    r"04_maps\annual_dynamic_2017_2024\2018\02_probability_maps_250m"
    r"\cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"
)

SEISMIC_ROOT = PROJECT_ROOT / r"01_clean_data\new_step_required_data\B3_annual_seismicity_2017_2024"
CATALOG_CSV = SEISMIC_ROOT / r"catalog_usgs_comcat\usgs_comcat_cpec_buffer_M4plus_2017_2024_combined.csv"
SHAKEMAP_TABLE = (
    SEISMIC_ROOT
    / r"shakemap_metadata\usgs_shakemap_product_urls_and_normalized_grid_files_2017_2024.csv"
)
SUMMARY_DIR = SEISMIC_ROOT / "derived_summaries"


def log(message: str) -> None:
    print(message, flush=True)


def output_dir_for_year(year: int) -> Path:
    out_dir = (
        PROJECT_ROOT
        / "04_maps"
        / "annual_dynamic_2017_2024"
        / str(year)
        / "01_predictor_rasters_250m"
        / "annual_seismicity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def raster_profile(template: rasterio.io.DatasetReader) -> dict:
    profile = template.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def write_full_raster(path: Path, data: np.ndarray, valid_mask: np.ndarray, profile: dict) -> None:
    arr = np.asarray(data, dtype=np.float32)
    out = np.where(valid_mask, arr, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_CSV)
    required = {"year", "longitude", "latitude", "mag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required fields: {sorted(missing)}")
    df = df.dropna(subset=["year", "longitude", "latitude", "mag"]).copy()
    df["year"] = df["year"].astype(int)
    df["mag"] = df["mag"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    return df


def wgs84_to_utm42(lon_deg, lat_deg) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS84 geographic to EPSG:32642 forward transform.

    This avoids a local pyproj/PROJ database issue while preserving the project
    convention of measuring distances in UTM Zone 42N.
    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lon0 = math.radians(69.0)

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)

    n = a / np.sqrt(1.0 - e2 * sin_phi**2)
    t = tan_phi**2
    c = ep2 * cos_phi**2
    aa = cos_phi * (lam - lon0)
    e4 = e2**2
    e6 = e2**3
    m = a * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * np.sin(2.0 * phi)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * np.sin(4.0 * phi)
        - (35.0 * e6 / 3072.0) * np.sin(6.0 * phi)
    )
    x = 500000.0 + k0 * n * (
        aa
        + (1.0 - t + c) * aa**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * ep2) * aa**5 / 120.0
    )
    y = k0 * (
        m
        + n
        * tan_phi
        * (
            aa**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * aa**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * ep2) * aa**6 / 720.0
        )
    )
    return x, y


def build_event_trees(df: pd.DataFrame) -> tuple[dict[int, cKDTree], dict[int, cKDTree]]:
    trees_all: dict[int, cKDTree] = {}
    trees_m5: dict[int, cKDTree] = {}

    for year in YEARS:
        yearly = df[df["year"] == year]
        if yearly.empty:
            continue
        x, y = wgs84_to_utm42(yearly["longitude"].to_numpy(), yearly["latitude"].to_numpy())
        trees_all[year] = cKDTree(np.column_stack([x, y]))

        yearly_m5 = yearly[yearly["mag"] >= 5.0]
        if not yearly_m5.empty:
            x5, y5 = wgs84_to_utm42(yearly_m5["longitude"].to_numpy(), yearly_m5["latitude"].to_numpy())
            trees_m5[year] = cKDTree(np.column_stack([x5, y5]))

    return trees_all, trees_m5


def cell_area_km2_by_row(transform: Affine, height: int, width: int) -> np.ndarray:
    # Existing annual rasters are geographic WGS84. This area approximation is sufficient
    # for a relative earthquake-kernel density index at the corridor scale.
    rows = np.arange(height, dtype=np.float64)
    lat = transform.f + (rows + 0.5) * transform.e
    dx_km = abs(transform.a) * 111.320 * np.cos(np.deg2rad(lat))
    dy_km = abs(transform.e) * 110.574
    area = dx_km * dy_km
    area[area <= 0] = np.nan
    return area.astype(np.float32)


def generate_kernel_density_rasters(
    df: pd.DataFrame,
    template: rasterio.io.DatasetReader,
    valid_mask: np.ndarray,
    profile: dict,
    bandwidth_km: float = 25.0,
    internal_scale_factor: int = 10,
) -> list[Path]:
    height, width = template.height, template.width
    transform = template.transform
    coarse_height = int(math.ceil(height / internal_scale_factor))
    coarse_width = int(math.ceil(width / internal_scale_factor))
    coarse_transform = transform * Affine.scale(internal_scale_factor, internal_scale_factor)
    res_x = abs(coarse_transform.a)
    res_y = abs(coarse_transform.e)
    mean_lat = (template.bounds.top + template.bounds.bottom) / 2.0
    pixel_m_y = res_y * 110_574.0
    pixel_m_x = res_x * 111_320.0 * math.cos(math.radians(mean_lat))
    sigma_y = bandwidth_km * 1000.0 / pixel_m_y
    sigma_x = bandwidth_km * 1000.0 / pixel_m_x
    pad = int(math.ceil(max(sigma_x, sigma_y) * 4.0)) + 8
    expanded_transform = coarse_transform * Affine.translation(-pad, -pad)
    exp_height = coarse_height + 2 * pad
    exp_width = coarse_width + 2 * pad
    row_area_km2 = cell_area_km2_by_row(coarse_transform, coarse_height, coarse_width)
    written: list[Path] = []

    log(
        f"Kernel density: bandwidth={bandwidth_km:g} km, "
        f"internal grid={internal_scale_factor}x template pixel, "
        f"sigma_y={sigma_y:.1f}px, sigma_x={sigma_x:.1f}px, pad={pad}px"
    )

    for year in YEARS:
        yearly = df[df["year"] == year]
        count_grid = np.zeros((exp_height, exp_width), dtype=np.float32)
        mag_grid = np.zeros((exp_height, exp_width), dtype=np.float32)

        for row in yearly.itertuples(index=False):
            r, c = rowcol(expanded_transform, float(row.longitude), float(row.latitude))
            if 0 <= r < exp_height and 0 <= c < exp_width:
                count_grid[r, c] += 1.0
                mag_grid[r, c] += float(row.mag)

        count_kde = gaussian_filter(
            count_grid, sigma=(sigma_y, sigma_x), mode="constant", cval=0.0, truncate=4.0
        )
        mag_kde = gaussian_filter(
            mag_grid, sigma=(sigma_y, sigma_x), mode="constant", cval=0.0, truncate=4.0
        )
        del count_grid, mag_grid

        count_crop = count_kde[pad : pad + coarse_height, pad : pad + coarse_width]
        mag_crop = mag_kde[pad : pad + coarse_height, pad : pad + coarse_width]
        del count_kde, mag_kde

        # Convert smoothed event mass per pixel to an approximate density per 1000 km2.
        count_density = (count_crop / row_area_km2[:, None]) * 1000.0
        mag_density = (mag_crop / row_area_km2[:, None]) * 1000.0
        count_250m = np.zeros((height, width), dtype=np.float32)
        mag_250m = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=count_density.astype(np.float32),
            destination=count_250m,
            src_transform=coarse_transform,
            src_crs=template.crs,
            dst_transform=template.transform,
            dst_crs=template.crs,
            src_nodata=None,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=mag_density.astype(np.float32),
            destination=mag_250m,
            src_transform=coarse_transform,
            src_crs=template.crs,
            dst_transform=template.transform,
            dst_crs=template.crs,
            src_nodata=None,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )

        out_dir = output_dir_for_year(year)
        count_path = out_dir / f"cpec_{year}_annual_earthquake_kernel_density_m4plus_per_1000km2_250m.tif"
        mag_path = out_dir / (
            f"cpec_{year}_annual_magnitude_weighted_earthquake_kernel_density_"
            "m4plus_per_1000km2_250m.tif"
        )
        write_full_raster(count_path, count_250m, valid_mask, profile)
        write_full_raster(mag_path, mag_250m, valid_mask, profile)
        written.extend([count_path, mag_path])
        log(f"{year}: wrote earthquake kernel density rasters")

    return written


def write_distance_rasters(
    template: rasterio.io.DatasetReader,
    valid_mask: np.ndarray,
    profile: dict,
    trees_all: dict[int, cKDTree],
    trees_m5: dict[int, cKDTree],
) -> list[Path]:
    height, width = template.height, template.width
    transform = template.transform
    block_rows = 256

    outputs: dict[tuple[int, str], rasterio.io.DatasetWriter] = {}
    paths: list[Path] = []
    for year in YEARS:
        out_dir = output_dir_for_year(year)
        for key, suffix in [
            ("all", "distance_to_nearest_annual_earthquake_m4plus_km_250m"),
            ("m5", "distance_to_nearest_annual_earthquake_m5plus_km_250m"),
        ]:
            path = out_dir / f"cpec_{year}_{suffix}.tif"
            outputs[(year, key)] = rasterio.open(path, "w", **profile)
            paths.append(path)

    try:
        col_idx = np.arange(width, dtype=np.float64)
        lon_row = transform.c + (col_idx + 0.5) * transform.a
        for r0 in range(0, height, block_rows):
            h = min(block_rows, height - r0)
            rows = np.arange(r0, r0 + h, dtype=np.float64)
            lat_col = transform.f + (rows + 0.5) * transform.e
            lon_grid = np.broadcast_to(lon_row, (h, width))
            lat_grid = np.broadcast_to(lat_col[:, None], (h, width))
            x, y = wgs84_to_utm42(lon_grid, lat_grid)
            points = np.column_stack([x.ravel(), y.ravel()])
            block_valid = valid_mask[r0 : r0 + h, :]
            valid_flat = block_valid.ravel()

            for year in YEARS:
                for key, tree_dict in [("all", trees_all), ("m5", trees_m5)]:
                    arr = np.full((h, width), NODATA, dtype=np.float32)
                    tree = tree_dict.get(year)
                    if tree is not None and valid_flat.any():
                        distances_m, _ = tree.query(points[valid_flat], k=1)
                        flat = arr.ravel()
                        flat[valid_flat] = (distances_m / 1000.0).astype(np.float32)
                    outputs[(year, key)].write(arr, 1, window=Window(0, r0, width, h))

            if r0 % (block_rows * 8) == 0:
                log(f"Distance rasters: processed rows {r0}-{r0 + h - 1} of {height}")
    finally:
        for dst in outputs.values():
            dst.close()

    log("Distance rasters complete")
    return paths


def read_shakemap_grid(path: Path) -> tuple[np.ndarray, np.ndarray, Affine, str, dict[str, float]]:
    if path.read_bytes()[:2] == b"PK":
        with zipfile.ZipFile(path) as zf:
            xml_name = next((name for name in zf.namelist() if name.endswith("grid.xml")), zf.namelist()[0])
            text = zf.read(xml_name).decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    spec_match = re.search(r"<grid_specification\s+([^>]+)/?>", text)
    if not spec_match:
        raise ValueError(f"No grid_specification in {path}")
    attrs = dict(re.findall(r'(\w+)="([^"]+)"', spec_match.group(1)))
    lon_min = float(attrs["lon_min"])
    lat_min = float(attrs["lat_min"])
    lon_max = float(attrs["lon_max"])
    lat_max = float(attrs["lat_max"])
    dx = float(attrs["nominal_lon_spacing"])
    dy = float(attrs["nominal_lat_spacing"])
    nlon = int(attrs["nlon"])
    nlat = int(attrs["nlat"])

    fields = []
    for field_text in re.findall(r"<grid_field\s+([^>]+)/?>", text):
        field_attrs = dict(re.findall(r'(\w+)="([^"]+)"', field_text))
        fields.append((int(field_attrs["index"]), field_attrs["name"]))
    field_names = [name for _, name in sorted(fields)]
    pga_idx = field_names.index("PGA")
    mmi_idx = field_names.index("MMI")

    data_match = re.search(r"<grid_data>(.*?)</grid_data>", text, flags=re.S)
    if not data_match:
        raise ValueError(f"No grid_data in {path}")
    values = np.fromstring(data_match.group(1), sep=" ", dtype=np.float32)
    nfields = len(field_names)
    expected = nlat * nlon * nfields
    if values.size != expected:
        raise ValueError(f"{path.name}: expected {expected} grid values, got {values.size}")

    table = values.reshape((nlat * nlon, nfields))
    pga = table[:, pga_idx].reshape((nlat, nlon)).astype(np.float32)
    mmi = table[:, mmi_idx].reshape((nlat, nlon)).astype(np.float32)
    src_transform = rasterio.transform.from_origin(lon_min - dx / 2, lat_max + dy / 2, dx, dy)
    meta = {
        "lon_min": lon_min,
        "lat_min": lat_min,
        "lon_max": lon_max,
        "lat_max": lat_max,
        "dx": dx,
        "dy": dy,
        "nlon": nlon,
        "nlat": nlat,
    }
    return pga, mmi, src_transform, "EPSG:4326", meta


def clipped_window_from_bounds(bounds: tuple[float, float, float, float], template) -> Window | None:
    dst_window = from_bounds(*bounds, transform=template.transform)
    dst_window = dst_window.round_offsets().round_lengths()
    pad = 2
    dst_window = Window(
        dst_window.col_off - pad,
        dst_window.row_off - pad,
        dst_window.width + 2 * pad,
        dst_window.height + 2 * pad,
    )
    full = Window(0, 0, template.width, template.height)
    try:
        inter = dst_window.intersection(full)
    except Exception:
        return None
    if inter.width <= 0 or inter.height <= 0:
        return None
    return Window(int(inter.col_off), int(inter.row_off), int(inter.width), int(inter.height))


def generate_shakemap_max_rasters(
    template: rasterio.io.DatasetReader,
    valid_mask: np.ndarray,
    profile: dict,
) -> list[Path]:
    table = pd.read_csv(SHAKEMAP_TABLE)
    table = table.dropna(subset=["year", "normalized_grid_file"]).copy()
    table["year"] = table["year"].astype(int)
    paths: list[Path] = []

    for year in YEARS:
        yearly = table[table["year"] == year]
        max_pga = np.zeros((template.height, template.width), dtype=np.float32)
        max_mmi = np.zeros((template.height, template.width), dtype=np.float32)

        for row in yearly.itertuples(index=False):
            grid_path = Path(str(row.normalized_grid_file))
            if not grid_path.exists():
                log(f"{year}: missing ShakeMap grid file {grid_path}")
                continue
            try:
                pga, mmi, src_transform, src_crs, meta = read_shakemap_grid(grid_path)
            except Exception as exc:
                log(f"{year}: skipped {grid_path.name}: {exc}")
                continue

            bounds = (
                meta["lon_min"] - meta["dx"] / 2,
                meta["lat_min"] - meta["dy"] / 2,
                meta["lon_max"] + meta["dx"] / 2,
                meta["lat_max"] + meta["dy"] / 2,
            )
            win = clipped_window_from_bounds(bounds, template)
            if win is None:
                continue
            h, w = int(win.height), int(win.width)
            dst_transform = window_transform(win, template.transform)
            dst_pga = np.zeros((h, w), dtype=np.float32)
            dst_mmi = np.zeros((h, w), dtype=np.float32)

            reproject(
                source=pga,
                destination=dst_pga,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=template.crs,
                src_nodata=None,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=mmi,
                destination=dst_mmi,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=template.crs,
                src_nodata=None,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
            )
            r0, c0 = int(win.row_off), int(win.col_off)
            max_pga[r0 : r0 + h, c0 : c0 + w] = np.maximum(
                max_pga[r0 : r0 + h, c0 : c0 + w], dst_pga
            )
            max_mmi[r0 : r0 + h, c0 : c0 + w] = np.maximum(
                max_mmi[r0 : r0 + h, c0 : c0 + w], dst_mmi
            )

        out_dir = output_dir_for_year(year)
        pga_path = out_dir / f"cpec_{year}_annual_max_shakemap_pga_percent_g_250m.tif"
        mmi_path = out_dir / f"cpec_{year}_annual_max_shakemap_mmi_250m.tif"
        write_full_raster(pga_path, max_pga, valid_mask, profile)
        write_full_raster(mmi_path, max_mmi, valid_mask, profile)
        paths.extend([pga_path, mmi_path])
        log(f"{year}: wrote annual maximum ShakeMap PGA/MMI rasters from {len(yearly)} events")

    return paths


def summarize_raster(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        valid = data.compressed()
        if valid.size == 0:
            return {
                "raster": str(path),
                "valid_pixels": 0,
                "min": "",
                "max": "",
                "mean": "",
                "p90": "",
            }
        return {
            "raster": str(path),
            "valid_pixels": int(valid.size),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "p90": float(np.percentile(valid, 90)),
        }


def write_outputs_manifest(paths: list[Path], template_path: Path) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = [summarize_raster(path) for path in paths]
    summary_csv = SUMMARY_DIR / "annual_seismic_factor_raster_qa_2017_2024.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["raster", "valid_pixels", "min", "max", "mean", "p90"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = SUMMARY_DIR / "annual_seismic_factor_raster_manifest_2017_2024.md"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("# Annual Seismic Factor Rasters, 2017-2024\n\n")
        handle.write("Created: 2026-06-05\n\n")
        handle.write(f"Template raster: `{template_path}`\n\n")
        handle.write("All rasters are clipped/masked to the existing 250 m CPEC annual-map grid.\n\n")
        handle.write("## Factors\n\n")
        handle.write("- Annual earthquake kernel density, M4.0+, per 1000 km2.\n")
        handle.write("- Annual magnitude-weighted earthquake kernel density, M4.0+, per 1000 km2.\n")
        handle.write("- Distance to nearest annual M4.0+ earthquake epicentre, km.\n")
        handle.write("- Distance to nearest annual M5.0+ earthquake epicentre, km.\n")
        handle.write("- Annual maximum USGS ShakeMap PGA, percent g.\n")
        handle.write("- Annual maximum USGS ShakeMap MMI.\n\n")
        handle.write("## Output Rasters\n\n")
        for path in sorted(paths):
            handle.write(f"- `{path}`\n")
        handle.write("\n## QA Table\n\n")
        handle.write(f"`{summary_csv}`\n")

    log(f"Wrote QA summary: {summary_csv}")
    log(f"Wrote manifest: {manifest}")


def main() -> int:
    if not TEMPLATE_RASTER.exists():
        raise FileNotFoundError(TEMPLATE_RASTER)
    if not CATALOG_CSV.exists():
        raise FileNotFoundError(CATALOG_CSV)
    if not SHAKEMAP_TABLE.exists():
        raise FileNotFoundError(SHAKEMAP_TABLE)

    df = load_catalog()
    log(f"Loaded USGS catalog records: {len(df)}")

    written_paths: list[Path] = []
    with rasterio.open(TEMPLATE_RASTER) as template:
        valid_mask = template.read(1) != template.nodata
        profile = raster_profile(template)
        trees_all, trees_m5 = build_event_trees(df)

        written_paths.extend(generate_kernel_density_rasters(df, template, valid_mask, profile))
        written_paths.extend(write_distance_rasters(template, valid_mask, profile, trees_all, trees_m5))
        written_paths.extend(generate_shakemap_max_rasters(template, valid_mask, profile))

    write_outputs_manifest(written_paths, TEMPLATE_RASTER)
    log("Annual seismic factor raster generation complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
