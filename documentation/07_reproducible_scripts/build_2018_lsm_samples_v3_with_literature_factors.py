"""Build corrected 2018 v3 sample table with literature-supported factors.

This extends cpec_2018_lsm_samples_v2 with full-CPEC factors that were missing:
- distance to roads
- distance to rivers/streams
- distance to active faults
- lithology/geology class
- selected terrain morphology and seismic/soil/human context rasters where available

The script samples only from full-CPEC data in D:\CPEC data. It does not use
the archived KKH-subset folders.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Some Windows Conda sessions expose PROJ_DATA but pyproj still fails to find
# proj.db. Set both before importing geopandas/pyproj.
os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import pyproj.datadir
from pyproj import CRS
from shapely.geometry import Point

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")

PROJECT_ROOT = Path(r"D:\DING PROJECT")
CPEC_DATA = Path(r"D:\CPEC data")
INPUT = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
OUTPUT = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "sample_v3_factor_audit"
REPORT = PROJECT_ROOT / "05_reports" / "sample_v3_build_report_2026-05-03.md"

WGS84 = CRS.from_epsg(4326)
DISTANCE_CRS = CRS.from_proj4("+proj=aeqd +lat_0=32.55 +lon_0=70.39 +datum=WGS84 +units=m +no_defs")


def find_one(pattern: str, *, contains: str | None = None) -> Path:
    matches = sorted(CPEC_DATA.rglob(pattern), key=lambda p: (len(str(p)), str(p)))
    if contains:
        matches = [p for p in matches if contains in str(p)]
    if not matches:
        raise FileNotFoundError(f"No match for {pattern!r} contains={contains!r}")
    return matches[0]


def find_sources() -> dict[str, str]:
    sources = {
        "roads": str(find_one("路网.shp")),
        "rivers": str(find_one("河网.shp")),
        "faults": str(CPEC_DATA / "中巴经济走廊1：25万活动断裂带（1964年）" / "FAULTS.shp"),
        "geology_200m": str(find_one("地层岩性.shp")),
        "soil_type": str(find_one("土壤类型.tif")),
        "river_distance_raster": str(find_one("river_buffer.tif")),
        "profile_curvature": str(find_one("剖面曲率.tif")),
        "plan_curvature": str(find_one("平面曲率.tif")),
        "tri": str(find_one("TRI.tif")),
        "twi": str(find_one("Topographic Wetness Index1.tif")),
        "valley_depth": str(find_one("Valley Depth.tif")),
        "relief": str(find_one("地形起伏度.tif")),
        "ls_factor": str(find_one("LS-Factor1.tif")),
        "earthquake_density": str(find_one("大于5Ms地震密度.tif")),
        "population_density": str(find_one("人口密度.tif")),
        "night_lights": str(find_one("夜间灯光指数.tif")),
    }
    return sources


def samples_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df[["longitude", "latitude"]].copy(), geometry=geom, crs=WGS84)


def read_vector(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    return gdf


def distance_to_vector(points: gpd.GeoDataFrame, path: str) -> np.ndarray:
    features = read_vector(path)
    p = points.to_crs(DISTANCE_CRS)
    f = features.to_crs(DISTANCE_CRS)
    # union_all is fast enough for these regional 1:250k line layers and 3316 samples.
    target = f.geometry.union_all()
    return p.geometry.distance(target).to_numpy(dtype="float64")


def nearest_polygon_code(points: gpd.GeoDataFrame, path: str) -> tuple[np.ndarray, str]:
    poly = read_vector(path)
    pts = points.to_crs(poly.crs)
    joined = gpd.sjoin(pts[["geometry"]], poly, how="left", predicate="within")
    candidates = [c for c in joined.columns if c not in {"geometry", "index_right"}]
    chosen = None
    for c in candidates:
        if joined[c].notna().sum() > 0 and joined[c].nunique(dropna=True) > 1:
            chosen = c
            break
    if chosen is None:
        return np.full(len(points), np.nan), "none"
    codes, _ = pd.factorize(joined[chosen], sort=True)
    codes = codes.astype("float64")
    codes[codes < 0] = np.nan
    return codes, chosen


def sample_raster(points: gpd.GeoDataFrame, path: str) -> np.ndarray:
    values = np.full(len(points), np.nan, dtype="float64")
    with rasterio.open(path) as src:
        pts = points.to_crs(src.crs) if src.crs else points
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        nodata = src.nodata
        for i, val in enumerate(src.sample(coords, indexes=1)):
            x = float(val[0])
            if nodata is not None and math.isclose(x, float(nodata), rel_tol=0, abs_tol=1e-12):
                values[i] = np.nan
            elif np.isfinite(x):
                values[i] = x
    return values


def add_log1p(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        out = f"log1p_{col}"
        vals = pd.to_numeric(df[col], errors="coerce")
        df[out] = np.log1p(np.clip(vals, 0, None))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    pts = samples_gdf(df)
    sources = find_sources()

    additions: dict[str, dict[str, object]] = {}

    distance_layers = {
        "dist_road_m": sources["roads"],
        "dist_river_m": sources["rivers"],
        "dist_fault_m": sources["faults"],
    }
    for col, path in distance_layers.items():
        df[col] = distance_to_vector(pts, path)
        additions[col] = {"source": path, "type": "vector_distance_m", "missing": int(pd.isna(df[col]).sum())}

    lith_codes, lith_field = nearest_polygon_code(pts, sources["geology_200m"])
    df["lithology_code"] = lith_codes
    additions["lithology_code"] = {
        "source": sources["geology_200m"],
        "type": "polygon_class_factorized",
        "source_field": lith_field,
        "missing": int(pd.isna(df["lithology_code"]).sum()),
    }

    raster_layers = {
        "profile_curvature": sources["profile_curvature"],
        "plan_curvature": sources["plan_curvature"],
        "tri": sources["tri"],
        "twi": sources["twi"],
        "valley_depth": sources["valley_depth"],
        "relief": sources["relief"],
        "ls_factor": sources["ls_factor"],
        "soil_type": sources["soil_type"],
        "eq_density_ms5": sources["earthquake_density"],
        "population_density": sources["population_density"],
        "night_lights": sources["night_lights"],
        "dist_river_raster_m": sources["river_distance_raster"],
    }
    for col, path in raster_layers.items():
        df[col] = sample_raster(pts, path)
        additions[col] = {"source": path, "type": "raster_sample", "missing": int(pd.isna(df[col]).sum())}

    add_log1p(df, ["dist_road_m", "dist_river_m", "dist_fault_m"])

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    (OUT_DIR / "sample_v3_sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = pd.DataFrame(
        [
            {
                "factor": col,
                "source": info["source"],
                "type": info["type"],
                "source_field": info.get("source_field", ""),
                "missing_count": info["missing"],
                "missing_percent": 100 * info["missing"] / len(df),
                "min": float(pd.to_numeric(df[col], errors="coerce").min(skipna=True)),
                "max": float(pd.to_numeric(df[col], errors="coerce").max(skipna=True)),
                "mean": float(pd.to_numeric(df[col], errors="coerce").mean(skipna=True)),
            }
            for col, info in additions.items()
        ]
    )
    audit.to_csv(OUT_DIR / "sample_v3_added_factor_qa.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# cpec_2018_lsm_samples_v3 Build Report",
        "",
        "## Literature Gate",
        "",
        "The added factors pass the landslide-susceptibility literature gate: road proximity, river/stream proximity, fault proximity, lithology, terrain morphology, hydrological terrain indices, and seismic background are standard conditioning variables in recent ML/ensemble LSM studies.",
        "",
        "## Output",
        "",
        f"- Input: `{INPUT}`",
        f"- Output: `{OUTPUT}`",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        "",
        "## Added Factors",
        "",
        "| Factor | Type | Missing % | Min | Max | Mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['factor']} | {row['type']} | {row['missing_percent']:.2f} | "
            f"{row['min']:.3f} | {row['max']:.3f} | {row['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Saved QA",
            "",
            f"- Sources: `{OUT_DIR / 'sample_v3_sources.json'}`",
            f"- Added-factor QA: `{OUT_DIR / 'sample_v3_added_factor_qa.csv'}`",
            "",
            "## Next Step",
            "",
            "Run v3 multicollinearity diagnostics before modelling. Factors with excessive missingness, extreme redundancy, or weak justification should be dropped before final model training.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUTPUT))
    print(audit.drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()
