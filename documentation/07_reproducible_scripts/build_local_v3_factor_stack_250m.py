"""Build the 11 local v3 predictor bands on the GEE 250 m reference grid.

Outputs a single aligned GeoTIFF:
    D:\DING PROJECT\04_maps\stacked_ensemble_local_v3_factors_250m\cpec_2018_local_v3_factors_250m.tif

The stack contains:
- log1p road distance
- log1p river/stream distance
- log1p active fault distance
- lithology code
- profile curvature
- plan curvature
- TRI
- TWI
- valley depth
- soil type
- earthquake density > Ms5
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT_ROOT = Path(r"D:\DING PROJECT")
CPEC_DATA = Path(r"D:\CPEC data")
REF = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_gee_inputs_250m"
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
OUT_DIR = PROJECT_ROOT / "04_maps" / "stacked_ensemble_local_v3_factors_250m"
OUT = OUT_DIR / "cpec_2018_local_v3_factors_250m.tif"
REPORT = PROJECT_ROOT / "05_reports" / "local_v3_factor_stack_250m_report_2026-05-03.md"
SAMPLE_TABLE = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"

BANDS = [
    "log1p_dist_road_m",
    "log1p_dist_river_m",
    "log1p_dist_fault_m",
    "lithology_code",
    "profile_curvature",
    "plan_curvature",
    "tri",
    "twi",
    "valley_depth",
    "soil_type",
    "eq_density_ms5",
]


def find_one(pattern: str, *, contains: str | None = None) -> Path:
    matches = sorted(CPEC_DATA.rglob(pattern), key=lambda p: (len(str(p)), str(p)))
    if contains:
        matches = [p for p in matches if contains in str(p)]
    if not matches:
        raise FileNotFoundError(f"No match for {pattern!r} contains={contains!r}")
    return matches[0]


def sources() -> dict[str, Path]:
    return {
        "roads": find_one("路网.shp"),
        "rivers": find_one("河网.shp"),
        "faults": CPEC_DATA / "中巴经济走廊1：25万活动断裂带（1964年）" / "FAULTS.shp",
        "geology_200m": find_one("地层岩性.shp"),
        "soil_type": find_one("土壤类型.tif"),
        "profile_curvature": find_one("剖面曲率.tif"),
        "plan_curvature": find_one("平面曲率.tif"),
        "tri": find_one("TRI.tif"),
        "twi": find_one("Topographic Wetness Index1.tif"),
        "valley_depth": find_one("Valley Depth.tif"),
        "eq_density_ms5": find_one("大于5Ms地震密度.tif"),
    }


def read_vector(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def rasterize_lines_distance(path: Path, out_shape: tuple[int, int], transform, crs, pixel_size_m: float) -> np.ndarray:
    gdf = read_vector(path).to_crs(crs)
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    dist = distance_transform_edt(mask == 0) * pixel_size_m
    return np.log1p(dist).astype("float32")


def lithology_codes(path: Path, out_shape: tuple[int, int], transform, crs, fill_value: float) -> np.ndarray:
    gdf = read_vector(path).to_crs(crs)
    candidates = [c for c in gdf.columns if c != "geometry"]
    chosen = None
    for col in candidates:
        if gdf[col].notna().sum() > 0 and gdf[col].nunique(dropna=True) > 1:
            chosen = col
            break
    if chosen is None:
        return np.full(out_shape, fill_value, dtype="float32")
    codes, _ = pd.factorize(gdf[chosen], sort=True)
    shapes = [(geom, int(code)) for geom, code in zip(gdf.geometry, codes) if code >= 0 and geom is not None]
    arr = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        all_touched=True,
        dtype="float32",
    )
    arr = np.where(np.isfinite(arr), arr, fill_value).astype("float32")
    return arr


def reproject_raster(path: Path, out_shape: tuple[int, int], transform, crs, fill_value: float, categorical: bool = False) -> np.ndarray:
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    dst = np.full(out_shape, fill_value, dtype="float32")
    with rasterio.open(path) as src:
        src_arr = src.read(1, out_dtype="float32")
        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr = np.where(np.isclose(src_arr, float(src_nodata)), np.nan, src_arr)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=fill_value,
            resampling=resampling,
        )
    dst = np.where(np.isfinite(dst), dst, fill_value).astype("float32")
    return dst


def sample_means() -> dict[str, float]:
    df = pd.read_csv(SAMPLE_TABLE, encoding="utf-8-sig")
    return {band: float(pd.to_numeric(df[band], errors="coerce").mean()) for band in BANDS}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    srcs = sources()
    fills = sample_means()
    rows = []
    with rasterio.open(REF) as ref:
        shape = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs
        # Reference is EPSG:4326 at about 250 m; use nominal 250 m for distance-transform conversion.
        pixel_size_m = 250.0
        profile = ref.profile.copy()
        profile.update(
            {
                "count": len(BANDS),
                "dtype": "float32",
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
                "nodata": None,
            }
        )
        arrays = {
            "log1p_dist_road_m": rasterize_lines_distance(srcs["roads"], shape, transform, crs, pixel_size_m),
            "log1p_dist_river_m": rasterize_lines_distance(srcs["rivers"], shape, transform, crs, pixel_size_m),
            "log1p_dist_fault_m": rasterize_lines_distance(srcs["faults"], shape, transform, crs, pixel_size_m),
            "lithology_code": lithology_codes(srcs["geology_200m"], shape, transform, crs, fills["lithology_code"]),
            "profile_curvature": reproject_raster(srcs["profile_curvature"], shape, transform, crs, fills["profile_curvature"]),
            "plan_curvature": reproject_raster(srcs["plan_curvature"], shape, transform, crs, fills["plan_curvature"]),
            "tri": reproject_raster(srcs["tri"], shape, transform, crs, fills["tri"]),
            "twi": reproject_raster(srcs["twi"], shape, transform, crs, fills["twi"]),
            "valley_depth": reproject_raster(srcs["valley_depth"], shape, transform, crs, fills["valley_depth"]),
            "soil_type": reproject_raster(srcs["soil_type"], shape, transform, crs, fills["soil_type"], categorical=True),
            "eq_density_ms5": reproject_raster(srcs["eq_density_ms5"], shape, transform, crs, fills["eq_density_ms5"]),
        }
        with rasterio.open(OUT, "w", **profile) as dst:
            for idx, band in enumerate(BANDS, start=1):
                arr = arrays[band].astype("float32")
                arr = np.where(np.isfinite(arr), arr, fills[band]).astype("float32")
                dst.write(arr, idx)
                dst.set_band_description(idx, band)
                rows.append(
                    {
                        "band": band,
                        "fill_value": fills[band],
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                        "source": str(srcs.get(band.replace("log1p_dist_", "").replace("_m", "s"), "")),
                    }
                )
            dst.update_tags(
                year="2018",
                resolution_m="250",
                output_type="local_v3_predictor_stack_for_spatial_cv_stacked_ensemble",
                band_count=str(len(BANDS)),
                reference_grid=str(REF),
            )

    qa = pd.DataFrame(rows)
    qa.to_csv(OUT_DIR / "local_v3_factor_stack_250m_qa.csv", index=False)
    lines = [
        "# Local v3 Factor Stack 250 m",
        "",
        f"Output: `{OUT}`",
        "",
        "The stack is aligned to the GEE 250 m reference grid and contains the 11 local v3 factors missing from the GEE public factor image.",
        "",
        "| Band | Fill value | Min | Max | Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['band']} | {row['fill_value']:.6g} | {row['min']:.6g} | {row['max']:.6g} | {row['mean']:.6g} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
