"""Create 250 m high-impact outputs from the 2018 stacked LSM rasters.

Outputs are derived from the three true 250 m stacked-ensemble probability
rasters:
- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

The script creates uncertainty/reliability rasters, AlphaEarth added-value
diagnostics, optional quantile priority zones, and CPEC/KKH road exposure
hotspot tables. It uses the official study-area boundary as an analysis mask
so statistics do not include the rectangular raster background.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from pyproj import CRS
from rasterio import features
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT_ROOT = Path(r"D:\DING PROJECT")
SOURCE_RASTER_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
OUT_DIR = PROJECT_ROOT / "04_maps" / "stacked_ensemble_250m_high_impact_outputs"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "stacked_250m_uncertainty_added_value_road_exposure_2026-05-03.md"
SCRIPT_PATH = PROJECT_ROOT / "06_scripts" / "python" / Path(__file__).name

BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)
CPEC_DATA = Path(r"D:\CPEC data")

RASTERS = {
    "Conventional": SOURCE_RASTER_DIR
    / "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
    "AlphaEarth Embeddings": SOURCE_RASTER_DIR
    / "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    "Conventional + AlphaEarth Embeddings": SOURCE_RASTER_DIR
    / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
}

OUTPUT_RASTERS = {
    "mean_probability": "cpec_2018_stacked_mean_probability_250m.tif",
    "disagreement_std": "cpec_2018_stacked_probability_disagreement_std_250m.tif",
    "disagreement_range": "cpec_2018_stacked_probability_disagreement_range_250m.tif",
    "fused_entropy": "cpec_2018_fused_probability_entropy_250m.tif",
    "reliability_score": "cpec_2018_fused_reliability_score_250m.tif",
    "fused_minus_conventional": "cpec_2018_fused_minus_conventional_added_value_250m.tif",
    "fused_minus_alphaearth": "cpec_2018_fused_minus_alphaearth_probability_250m.tif",
    "alphaearth_minus_conventional": "cpec_2018_alphaearth_minus_conventional_probability_250m.tif",
    "alphaearth_conventional_abs_disagreement": (
        "cpec_2018_alphaearth_conventional_disagreement_abs_250m.tif"
    ),
    "reliable_high_probability_mask": "cpec_2018_reliable_high_probability_mask_250m.tif",
    "uncertain_high_probability_mask": "cpec_2018_uncertain_high_probability_mask_250m.tif",
    "fused_quantile_priority_zones": "cpec_2018_fused_probability_quantile_priority_zones_250m.tif",
}

NODATA_FLOAT = -9999.0
NODATA_UINT8 = 255
WGS84 = CRS.from_epsg(4326)
DISTANCE_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=32.55 +lon_0=70.39 +datum=WGS84 +units=m +no_defs"
)
MAX_THREE_MODEL_STD = math.sqrt(2.0 / 9.0)


@dataclass
class RasterPackage:
    profile: dict
    transform: object
    crs: CRS
    mask: np.ndarray
    conventional: np.ndarray
    alphaearth: np.ndarray
    fused: np.ndarray


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_road_source() -> Path:
    """Find the official 2018 1:250k road shapefile without hardcoding Chinese text."""
    candidates = []
    for shp in CPEC_DATA.rglob("*.shp"):
        text = str(shp)
        if "2018" in text and shp.stat().st_size > 100_000:
            try:
                gdf = gpd.read_file(shp, rows=1)
                cols = {c.lower() for c in gdf.columns}
                if {"type", "rank"}.issubset(cols) and "shape_le_2" in cols:
                    candidates.append(shp)
            except Exception:
                continue
    if not candidates:
        # The road layer is 451176 bytes in the current CPEC data package.
        candidates = [p for p in CPEC_DATA.rglob("*.shp") if p.stat().st_size == 451176]
    if not candidates:
        raise FileNotFoundError("Could not find the 2018 CPEC road network shapefile.")
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def read_boundary(crs: CRS | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.from_epsg(32642), allow_override=True)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def rasterize_study_mask(profile: dict) -> np.ndarray:
    boundary = read_boundary(profile["crs"])
    shapes = [(geom, 1) for geom in boundary.geometry]
    mask = features.rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    return mask


def read_rasters() -> RasterPackage:
    with rasterio.open(RASTERS["Conventional"]) as src:
        profile = src.profile.copy()
        conventional = src.read(1).astype("float32")
    arrays = {"Conventional": conventional}
    for label in ["AlphaEarth Embeddings", "Conventional + AlphaEarth Embeddings"]:
        with rasterio.open(RASTERS[label]) as src:
            if src.width != profile["width"] or src.height != profile["height"]:
                raise ValueError(f"Raster grid mismatch: {RASTERS[label]}")
            arrays[label] = src.read(1).astype("float32")

    mask = rasterize_study_mask(profile)
    return RasterPackage(
        profile=profile,
        transform=profile["transform"],
        crs=profile["crs"],
        mask=mask,
        conventional=arrays["Conventional"],
        alphaearth=arrays["AlphaEarth Embeddings"],
        fused=arrays["Conventional + AlphaEarth Embeddings"],
    )


def stats_for(name: str, arr: np.ndarray, mask: np.ndarray) -> dict:
    valid = arr[mask & np.isfinite(arr)]
    return {
        "name": name,
        "count": int(valid.size),
        "min": float(np.min(valid)),
        "p05": float(np.percentile(valid, 5)),
        "p20": float(np.percentile(valid, 20)),
        "p50": float(np.percentile(valid, 50)),
        "p80": float(np.percentile(valid, 80)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "nonfinite_inside_mask": int(np.size(arr[mask]) - np.isfinite(arr[mask]).sum()),
    }


def write_float_raster(path: Path, arr: np.ndarray, profile: dict, mask: np.ndarray, desc: str) -> None:
    out = arr.astype("float32", copy=True)
    out[~mask | ~np.isfinite(out)] = NODATA_FLOAT
    prof = profile.copy()
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA_FLOAT,
        compress="deflate",
        predictor=2,
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, desc)


def write_uint8_raster(path: Path, arr: np.ndarray, profile: dict, mask: np.ndarray, desc: str) -> None:
    out = arr.astype("uint8", copy=True)
    out[~mask] = NODATA_UINT8
    prof = profile.copy()
    prof.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=NODATA_UINT8,
        compress="deflate",
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, desc)


def create_raster_outputs(pkg: RasterPackage) -> tuple[dict[str, Path], dict, pd.DataFrame]:
    stack = np.stack([pkg.conventional, pkg.alphaearth, pkg.fused], axis=0).astype("float32")
    mean_prob = stack.mean(axis=0)
    disagreement_std = stack.std(axis=0)
    disagreement_range = stack.max(axis=0) - stack.min(axis=0)

    eps = np.float32(1e-6)
    p = np.clip(pkg.fused, eps, 1.0 - eps)
    entropy = (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)).astype("float32")
    uncertainty_norm = np.clip(disagreement_std / MAX_THREE_MODEL_STD, 0.0, 1.0)
    reliability = (pkg.fused * (1.0 - uncertainty_norm)).astype("float32")

    fused_minus_conv = (pkg.fused - pkg.conventional).astype("float32")
    fused_minus_alpha = (pkg.fused - pkg.alphaearth).astype("float32")
    alpha_minus_conv = (pkg.alphaearth - pkg.conventional).astype("float32")
    alpha_conv_abs = np.abs(alpha_minus_conv).astype("float32")

    fused_valid = pkg.fused[pkg.mask & np.isfinite(pkg.fused)]
    std_valid = disagreement_std[pkg.mask & np.isfinite(disagreement_std)]
    thresholds = {
        "fused_probability_p20": float(np.percentile(fused_valid, 20)),
        "fused_probability_p40": float(np.percentile(fused_valid, 40)),
        "fused_probability_p60": float(np.percentile(fused_valid, 60)),
        "fused_probability_p80": float(np.percentile(fused_valid, 80)),
        "fused_probability_p90": float(np.percentile(fused_valid, 90)),
        "uncertainty_std_p25": float(np.percentile(std_valid, 25)),
        "uncertainty_std_p50": float(np.percentile(std_valid, 50)),
        "uncertainty_std_p75": float(np.percentile(std_valid, 75)),
    }

    reliable_high = (
        (pkg.fused >= thresholds["fused_probability_p80"])
        & (disagreement_std <= thresholds["uncertainty_std_p50"])
        & pkg.mask
    ).astype("uint8")
    uncertain_high = (
        (pkg.fused >= thresholds["fused_probability_p80"])
        & (disagreement_std >= thresholds["uncertainty_std_p75"])
        & pkg.mask
    ).astype("uint8")

    priority = np.zeros(pkg.fused.shape, dtype="uint8")
    qvals = [
        thresholds["fused_probability_p20"],
        thresholds["fused_probability_p40"],
        thresholds["fused_probability_p60"],
        thresholds["fused_probability_p80"],
    ]
    priority[pkg.fused <= qvals[0]] = 1
    priority[(pkg.fused > qvals[0]) & (pkg.fused <= qvals[1])] = 2
    priority[(pkg.fused > qvals[1]) & (pkg.fused <= qvals[2])] = 3
    priority[(pkg.fused > qvals[2]) & (pkg.fused <= qvals[3])] = 4
    priority[pkg.fused > qvals[3]] = 5

    arrays = {
        "mean_probability": mean_prob,
        "disagreement_std": disagreement_std,
        "disagreement_range": disagreement_range,
        "fused_entropy": entropy,
        "reliability_score": reliability,
        "fused_minus_conventional": fused_minus_conv,
        "fused_minus_alphaearth": fused_minus_alpha,
        "alphaearth_minus_conventional": alpha_minus_conv,
        "alphaearth_conventional_abs_disagreement": alpha_conv_abs,
    }
    paths = {}
    for key, arr in arrays.items():
        path = OUT_DIR / OUTPUT_RASTERS[key]
        write_float_raster(path, arr, pkg.profile, pkg.mask, key)
        paths[key] = path

    uint_arrays = {
        "reliable_high_probability_mask": reliable_high,
        "uncertain_high_probability_mask": uncertain_high,
        "fused_quantile_priority_zones": priority,
    }
    for key, arr in uint_arrays.items():
        path = OUT_DIR / OUTPUT_RASTERS[key]
        write_uint8_raster(path, arr, pkg.profile, pkg.mask, key)
        paths[key] = path

    stats = [
        stats_for("Conventional stacked probability", pkg.conventional, pkg.mask),
        stats_for("AlphaEarth Embeddings stacked probability", pkg.alphaearth, pkg.mask),
        stats_for("Conventional + AlphaEarth Embeddings stacked probability", pkg.fused, pkg.mask),
        stats_for("Mean probability", mean_prob, pkg.mask),
        stats_for("Model disagreement std", disagreement_std, pkg.mask),
        stats_for("Model disagreement range", disagreement_range, pkg.mask),
        stats_for("Fused reliability score", reliability, pkg.mask),
        stats_for("Fused minus Conventional", fused_minus_conv, pkg.mask),
        stats_for("AlphaEarth minus Conventional", alpha_minus_conv, pkg.mask),
    ]
    stats_df = pd.DataFrame(stats)

    thresholds_path = OUT_DIR / "stacked_250m_thresholds_and_diagnostics.json"
    thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    stats_df.to_csv(OUT_DIR / "stacked_250m_raster_diagnostics.csv", index=False)
    paths["thresholds_json"] = thresholds_path
    paths["raster_diagnostics_csv"] = OUT_DIR / "stacked_250m_raster_diagnostics.csv"

    derived = {
        "mean_probability": mean_prob,
        "disagreement_std": disagreement_std,
        "reliability": reliability,
        "fused_minus_conventional": fused_minus_conv,
        "alphaearth_minus_conventional": alpha_minus_conv,
        "reliable_high": reliable_high,
        "uncertain_high": uncertain_high,
        "priority": priority,
    }
    return paths, {"thresholds": thresholds, "derived": derived}, stats_df


def downsample_for_plot(arr: np.ndarray, mask: np.ndarray, factor: int = 6) -> np.ndarray:
    out = arr[::factor, ::factor].astype("float32", copy=True)
    m = mask[::factor, ::factor]
    out[~m] = np.nan
    return out


def plot_raster_figures(pkg: RasterPackage, derived: dict) -> list[Path]:
    paths = []
    extent = [
        pkg.profile["transform"].c,
        pkg.profile["transform"].c + pkg.profile["transform"].a * pkg.profile["width"],
        pkg.profile["transform"].f + pkg.profile["transform"].e * pkg.profile["height"],
        pkg.profile["transform"].f,
    ]
    bnd = read_boundary(WGS84)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        ("Fused Stacked Probability", pkg.fused, "viridis", 0, 1),
        ("Model Disagreement (Std.)", derived["disagreement_std"], "magma", 0, None),
        ("Reliability Score", derived["reliability"], "YlGnBu", 0, 1),
        ("Mean Probability", derived["mean_probability"], "viridis", 0, 1),
    ]
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(
            downsample_for_plot(arr, pkg.mask),
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        bnd.boundary.plot(ax=ax, color="black", linewidth=0.35)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig_path = FIG_DIR / "fig_cpec_2018_stacked_uncertainty_reliability_250m.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(fig_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    max_abs = float(
        np.nanpercentile(
            np.abs(derived["fused_minus_conventional"][pkg.mask & np.isfinite(derived["fused_minus_conventional"])]),
            98,
        )
    )
    im0 = axes[0].imshow(
        downsample_for_plot(derived["fused_minus_conventional"], pkg.mask),
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
    )
    axes[0].set_title("Fused minus Conventional")
    fig.colorbar(im0, ax=axes[0], shrink=0.82)
    im1 = axes[1].imshow(
        downsample_for_plot(derived["alphaearth_minus_conventional"], pkg.mask),
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
    )
    axes[1].set_title("AlphaEarth Embeddings minus Conventional")
    fig.colorbar(im1, ax=axes[1], shrink=0.82)
    for ax in axes:
        bnd.boundary.plot(ax=ax, color="black", linewidth=0.35)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig_path = FIG_DIR / "fig_cpec_2018_alphaearth_added_value_250m.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(fig_path)

    cmap = ListedColormap(["#4575b4", "#91bfdb", "#ffffbf", "#fdae61", "#d73027"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)
    im = ax.imshow(
        downsample_for_plot(derived["priority"].astype("float32"), pkg.mask),
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
    )
    bnd.boundary.plot(ax=ax, color="black", linewidth=0.35)
    ax.set_title("Fused Probability Quantile Priority Zones")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5], shrink=0.78)
    cb.ax.set_yticklabels(["Very low", "Low", "Moderate", "High", "Very high"])
    fig_path = FIG_DIR / "fig_cpec_2018_fused_quantile_priority_zones_250m.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(fig_path)

    return paths


def flatten_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty and isinstance(g, LineString)]
    return []


def route_code(row: pd.Series) -> str:
    for col in ["shield", "name1", "name2", "name3", "name4", "name5"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "none":
            return str(value).strip()
    return "unknown"


def is_kkh_proxy(code: str) -> bool:
    cleaned = code.upper().replace("-", "").replace(" ", "")
    return cleaned in {"N35", "314", "G314"}


def build_road_segments(segment_length_m: float = 10_000.0) -> gpd.GeoDataFrame:
    roads_path = find_road_source()
    roads = gpd.read_file(roads_path)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    if roads.crs is None:
        roads = roads.set_crs(CRS.from_epsg(32642), allow_override=True)
    boundary = read_boundary(roads.crs)
    roads = gpd.clip(roads, boundary)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads = roads.to_crs(DISTANCE_CRS)

    rows = []
    seg_id = 1
    for _, row in roads.iterrows():
        code = route_code(row)
        attrs = {
            "route_code": code,
            "is_kkh_proxy": bool(is_kkh_proxy(code)),
            "iso_cc": row.get("iso_cc", None),
            "road_type": row.get("type", None),
            "road_rank": row.get("rank", None),
        }
        for line in flatten_lines(row.geometry):
            length = float(line.length)
            if length <= 0:
                continue
            starts = np.arange(0.0, length, segment_length_m)
            for start in starts:
                stop = min(start + segment_length_m, length)
                if stop - start < 1.0:
                    continue
                geom = substring(line, start, stop)
                if geom.is_empty:
                    continue
                rows.append(
                    {
                        "segment_id": seg_id,
                        "length_km": float(geom.length / 1000.0),
                        **attrs,
                        "geometry": geom,
                    }
                )
                seg_id += 1
    if not rows:
        raise RuntimeError("No road segments remained after clipping to the study boundary.")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=DISTANCE_CRS)
    gdf.attrs["road_source"] = str(roads_path)
    return gdf


def sample_array_at_points(src: rasterio.DatasetReader, points_wgs84: list) -> np.ndarray:
    coords = [(pt.x, pt.y) for pt in points_wgs84]
    vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
    vals[~np.isfinite(vals)] = np.nan
    return vals


def create_road_exposure(
    pkg: RasterPackage, derived: dict, thresholds: dict
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[Path]]:
    segments = build_road_segments()
    sample_points = []
    sample_segment_ids = []
    sample_spacing_m = 1000.0
    for _, row in segments.iterrows():
        geom = row.geometry
        length = float(geom.length)
        distances = list(np.arange(0.0, length, sample_spacing_m))
        distances.append(min(length, max(0.0, length * 0.5)))
        distances.append(length)
        for d in sorted(set(round(x, 3) for x in distances)):
            if d < 0 or d > length:
                continue
            sample_points.append(geom.interpolate(d))
            sample_segment_ids.append(row.segment_id)

    pts = gpd.GeoDataFrame(
        {"segment_id": sample_segment_ids}, geometry=sample_points, crs=DISTANCE_CRS
    ).to_crs(WGS84)

    # Use derived raster files so road exposure is traceable to archived map products.
    raster_paths = {
        "probability": RASTERS["Conventional + AlphaEarth Embeddings"],
        "uncertainty_std": OUT_DIR / OUTPUT_RASTERS["disagreement_std"],
        "reliability": OUT_DIR / OUTPUT_RASTERS["reliability_score"],
        "added_value": OUT_DIR / OUTPUT_RASTERS["fused_minus_conventional"],
        "priority_zone": OUT_DIR / OUTPUT_RASTERS["fused_quantile_priority_zones"],
    }
    sampled = pd.DataFrame({"segment_id": sample_segment_ids})
    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            sampled[name] = sample_array_at_points(src, list(pts.geometry))

    high80 = thresholds["fused_probability_p80"]
    high90 = thresholds["fused_probability_p90"]
    sampled["is_high80"] = sampled["probability"] >= high80
    sampled["is_high90"] = sampled["probability"] >= high90
    sampled["is_very_high_zone"] = sampled["priority_zone"] == 5
    sampled["positive_added_value"] = sampled["added_value"].clip(lower=0)

    agg = sampled.groupby("segment_id").agg(
        sample_n=("probability", "size"),
        mean_probability=("probability", "mean"),
        max_probability=("probability", "max"),
        mean_uncertainty_std=("uncertainty_std", "mean"),
        mean_reliability_score=("reliability", "mean"),
        mean_added_value_fused_minus_conventional=("added_value", "mean"),
        mean_positive_added_value=("positive_added_value", "mean"),
        high80_share=("is_high80", "mean"),
        high90_share=("is_high90", "mean"),
        very_high_zone_share=("is_very_high_zone", "mean"),
    )
    segments = segments.merge(agg, left_on="segment_id", right_index=True, how="left")
    segments["uncertainty_norm"] = (
        segments["mean_uncertainty_std"] / MAX_THREE_MODEL_STD
    ).clip(lower=0, upper=1)
    segments["priority_score"] = (
        segments["mean_probability"].fillna(0)
        * (1.0 - segments["uncertainty_norm"].fillna(1))
        + 0.25 * segments["mean_positive_added_value"].fillna(0)
        + 0.10 * segments["high90_share"].fillna(0)
    )
    segments = segments.sort_values("priority_score", ascending=False).reset_index(drop=True)
    segments["hotspot_rank"] = np.arange(1, len(segments) + 1)

    exposure_rows = []
    for group_name, group_df in [
        ("All clipped roads", segments),
        ("KKH proxy routes (N35 / 314)", segments[segments["is_kkh_proxy"]]),
    ]:
        if group_df.empty:
            continue
        total_len = float(group_df["length_km"].sum())
        exposure_rows.append(
            {
                "road_group": group_name,
                "segment_count": int(len(group_df)),
                "total_length_km": total_len,
                "weighted_length_probability_p80_km": float(
                    (group_df["length_km"] * group_df["high80_share"].fillna(0)).sum()
                ),
                "weighted_length_probability_p90_km": float(
                    (group_df["length_km"] * group_df["high90_share"].fillna(0)).sum()
                ),
                "length_majority_p80_km": float(
                    group_df.loc[group_df["high80_share"] >= 0.5, "length_km"].sum()
                ),
                "length_majority_p90_km": float(
                    group_df.loc[group_df["high90_share"] >= 0.5, "length_km"].sum()
                ),
                "mean_probability_length_weighted": float(
                    np.average(
                        group_df["mean_probability"].fillna(0),
                        weights=group_df["length_km"],
                    )
                ),
                "mean_uncertainty_length_weighted": float(
                    np.average(
                        group_df["mean_uncertainty_std"].fillna(0),
                        weights=group_df["length_km"],
                    )
                ),
                "mean_reliability_length_weighted": float(
                    np.average(
                        group_df["mean_reliability_score"].fillna(0),
                        weights=group_df["length_km"],
                    )
                ),
            }
        )
    summary = pd.DataFrame(exposure_rows)

    out_paths = []
    all_csv = OUT_DIR / "road_segment_exposure_all_250m.csv"
    top_csv = OUT_DIR / "road_hotspot_segments_top50_250m.csv"
    summary_csv = OUT_DIR / "road_exposure_summary_250m.csv"
    sampled_csv = OUT_DIR / "road_exposure_sample_points_250m.csv"
    geojson = OUT_DIR / "road_hotspot_segments_top50_250m.geojson"
    gpkg = OUT_DIR / "road_segment_exposure_all_250m.gpkg"

    segments.drop(columns="geometry").to_csv(all_csv, index=False)
    segments.head(50).drop(columns="geometry").to_csv(top_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    sampled.to_csv(sampled_csv, index=False)
    segments.to_file(gpkg, layer="road_segment_exposure_250m", driver="GPKG")
    segments.head(50).to_crs(WGS84).to_file(geojson, driver="GeoJSON")
    out_paths += [all_csv, top_csv, summary_csv, sampled_csv, geojson, gpkg]

    fig_path = plot_road_hotspot_figure(pkg, segments)
    out_paths.append(fig_path)

    return segments, summary, out_paths


def plot_road_hotspot_figure(pkg: RasterPackage, segments: gpd.GeoDataFrame) -> Path:
    extent = [
        pkg.profile["transform"].c,
        pkg.profile["transform"].c + pkg.profile["transform"].a * pkg.profile["width"],
        pkg.profile["transform"].f + pkg.profile["transform"].e * pkg.profile["height"],
        pkg.profile["transform"].f,
    ]
    bnd = read_boundary(WGS84)
    top = segments.head(50).to_crs(WGS84)
    kkh = segments[segments["is_kkh_proxy"]].to_crs(WGS84)
    all_roads = segments.to_crs(WGS84)

    fig, ax = plt.subplots(figsize=(8.6, 8.0), constrained_layout=True)
    im = ax.imshow(
        downsample_for_plot(pkg.fused, pkg.mask),
        extent=extent,
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    all_roads.plot(ax=ax, color="#2b2b2b", linewidth=0.12, alpha=0.35)
    if not kkh.empty:
        kkh.plot(ax=ax, color="#0072B2", linewidth=0.65, alpha=0.85)
    top.plot(
        ax=ax,
        column="priority_score",
        cmap="plasma",
        linewidth=1.15,
        legend=True,
        legend_kwds={"label": "Hotspot priority score", "shrink": 0.65},
    )
    bnd.boundary.plot(ax=ax, color="black", linewidth=0.45)
    fig.colorbar(im, ax=ax, label="Fused stacked probability", shrink=0.72)
    ax.set_title("CPEC/KKH Road Hotspot Segments from 250 m Fused Stacked Probability")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig_path = FIG_DIR / "fig_cpec_2018_road_exposure_hotspots_250m.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def markdown_table(df: pd.DataFrame, cols: list[str], decimals: int = 4, max_rows: int | None = None) -> str:
    view = df[cols].copy()
    if max_rows:
        view = view.head(max_rows)
    for c in view.select_dtypes(include=[np.floating]).columns:
        view[c] = view[c].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    view = view.fillna("")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = []
    for i, header in enumerate(headers):
        max_row_width = max([len(row[i]) for row in rows], default=0)
        widths.append(max(len(header), max_row_width))

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep, *[fmt(row) for row in rows]])


def write_report(
    paths: dict[str, Path],
    fig_paths: list[Path],
    raster_stats: pd.DataFrame,
    thresholds: dict,
    road_segments: gpd.GeoDataFrame,
    road_summary: pd.DataFrame,
    road_paths: list[Path],
) -> None:
    top_cols = [
        "hotspot_rank",
        "route_code",
        "is_kkh_proxy",
        "length_km",
        "mean_probability",
        "mean_uncertainty_std",
        "mean_reliability_score",
        "mean_added_value_fused_minus_conventional",
        "high80_share",
        "high90_share",
        "priority_score",
    ]
    stats_cols = ["name", "count", "min", "p50", "p80", "p90", "max", "mean", "std"]
    summary_cols = [
        "road_group",
        "segment_count",
        "total_length_km",
        "weighted_length_probability_p80_km",
        "weighted_length_probability_p90_km",
        "length_majority_p80_km",
        "length_majority_p90_km",
        "mean_probability_length_weighted",
        "mean_uncertainty_length_weighted",
        "mean_reliability_length_weighted",
    ]

    text = f"""# 250 m Stacked Ensemble Uncertainty, Added-Value, And Road Exposure Outputs

Date: 2026-05-03

## Purpose

This package converts the three 2018 **Spatial-CV Stacked Ensemble** probability rasters into publishable diagnostic products at the true 250 m modelling grid.

The core probability maps remain:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

The fused map, **Conventional + AlphaEarth Embeddings**, is used as the primary planning raster.

## Method Summary

1. Rasterized the official CPEC study-area boundary to the 250 m probability grid.
2. Computed mean probability, three-model disagreement, entropy, and a continuous reliability score.
3. Computed AlphaEarth added-value diagnostics using signed probability differences.
4. Created optional fused-probability quantile priority zones for exposure summaries.
5. Clipped the official 2018 CPEC road network to the study area.
6. Split roads into 10 km segments and sampled raster values every 1 km along each segment.
7. Ranked road hotspots by high fused probability, low model disagreement, and positive AlphaEarth added value.

The uncertainty map here is a **feature-set/model-disagreement uncertainty proxy** from the three stacked outputs. It is not yet a full bootstrap or repeated non-landslide sampling uncertainty product.

## Key Thresholds

- Fused probability P80: `{thresholds["fused_probability_p80"]:.6f}`
- Fused probability P90: `{thresholds["fused_probability_p90"]:.6f}`
- Uncertainty std P50: `{thresholds["uncertainty_std_p50"]:.6f}`
- Uncertainty std P75: `{thresholds["uncertainty_std_p75"]:.6f}`

Reliable high-probability mask: fused probability >= P80 and disagreement std <= P50.

Uncertain high-probability mask: fused probability >= P80 and disagreement std >= P75.

## Raster Diagnostics

{markdown_table(raster_stats, stats_cols)}

## Road Exposure Summary

{markdown_table(road_summary, summary_cols, decimals=3)}

## Top 15 Road Hotspot Segments

{markdown_table(road_segments, top_cols, decimals=4, max_rows=15)}

## Raster Outputs

""" + "\n".join(f"- `{path}`" for path in paths.values() if path.suffix.lower() in {".tif", ".json", ".csv"}) + f"""

## Road Outputs

""" + "\n".join(f"- `{path}`" for path in road_paths) + f"""

## Figure Outputs

""" + "\n".join(f"- `{path}`" for path in fig_paths) + f"""

## Interpretation For Paper

- The fused stacked model is treated as the best planning product because it combines physically interpretable conditioning factors with AlphaEarth embedding information.
- The disagreement map shows where the three feature sets give inconsistent susceptibility probabilities; these zones should be described as lower-confidence prediction areas.
- The reliable high-probability mask highlights locations where the fused model is high and the feature-set disagreement is low.
- The uncertain high-probability mask is useful for field verification because the model suggests possible hazard but the feature sets disagree.
- The fused-minus-Conventional map directly supports the AlphaEarth added-value argument. Positive values show where the fused model raises susceptibility relative to conventional predictors.
- The road hotspot output makes the work CPEC/KKH-applied by translating susceptibility into ranked infrastructure segments.

## Notes

- The optional five quantile zones are included only for map communication and road exposure summaries. They are not used as model classes and should not replace continuous probability maps.
- KKH proxy routes are identified from route codes `N35`, `314`, or `G314` where present in the official 2018 road layer.
- Outside-boundary raster cells are written as nodata in derived products. The no-nodata requirement is preserved inside the official study-area mask.
- Because distance to roads is also an input conditioning factor, the road exposure analysis should be described as a planning-priority overlay rather than a fully independent validation layer. A later road-distance ablation can quantify how sensitive the road hotspots are to that predictor.

## Literature Support

- Uncertainty and confidence maps are supported by recent uncertainty-focused LSM work, including Monte Carlo/confidence-map approaches for sampling randomness: [Quantifying uncertainty in landslide susceptibility mapping due to sampling randomness](https://www.sciencedirect.com/science/article/abs/pii/S2212420924007283).
- Sampling and non-landslide/control point choices are a recognized source of LSM uncertainty: [Scientific Reports 2024 sampling-resolution study](https://www.nature.com/articles/s41598-024-52145-w) and [ScienceDirect 2024 non-landslide sampling comparison](https://www.sciencedirect.com/science/article/pii/S1574954124001250).
- Stacking/heterogeneous ensembles and optimized sampling are consistent with recent LSM model-design literature: [Remote Sensing 2024 optimized sampling and heterogeneous ensemble ML](https://www.mdpi.com/2072-4292/16/19/3663).
- The AlphaEarth/Satellite Embedding novelty is supported by the official Earth Engine dataset and tutorial: [Satellite Embedding V1 Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) and [Earth Engine Satellite Embedding tutorial](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction).
- The foundation-model framing is supported by the AlphaEarth Foundations preprint: [AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291).

## Reproducibility

- Script: `{SCRIPT_PATH}`
- Source rasters: `{SOURCE_RASTER_DIR}`
- Output folder: `{OUT_DIR}`
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    pkg = read_rasters()
    paths, diag, raster_stats = create_raster_outputs(pkg)
    fig_paths = plot_raster_figures(pkg, diag["derived"])
    road_segments, road_summary, road_paths = create_road_exposure(
        pkg, diag["derived"], diag["thresholds"]
    )
    write_report(
        paths,
        fig_paths,
        raster_stats,
        diag["thresholds"],
        road_segments,
        road_summary,
        road_paths,
    )
    print(f"Wrote raster outputs to: {OUT_DIR}")
    print(f"Wrote figures to: {FIG_DIR}")
    print(f"Wrote report to: {REPORT}")


if __name__ == "__main__":
    main()
