"""Create final 2018 reliability-aware susceptibility and exposure products.

Inputs:
- 2018 Spatial-CV Stacked Ensemble probability rasters.
- 2018 area-of-applicability / transfer-confidence rasters.
- Three-feature-set disagreement raster from the previous high-impact module.
- Official CPEC study boundary, Pakistan admin polygons, Kashgar polygon, and
  CPEC 2018 road network.

Outputs:
- Reliability-weighted probability rasters for all three feature sets.
- Final fused reliable/uncertain high-susceptibility masks.
- Administrative-domain area summaries using corrected public labels.
- CPEC/KKH road exposure summaries and top hotspot segments.
"""

from __future__ import annotations

import json
import math
import os
import shutil
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import CRS, Geod
from rasterio import features
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
PROB_DIR = PROJECT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
AOA_DIR = PROJECT / "04_maps" / "area_of_applicability_transfer_confidence_250m"
HIGH_IMPACT_DIR = PROJECT / "04_maps" / "stacked_ensemble_250m_high_impact_outputs"
OUT_DIR = PROJECT / "04_maps" / "reliability_aware_susceptibility_2018_250m"
FIG_DIR = PROJECT / "05_reports" / "figures" / "reliability_aware_susceptibility_2018_250m"
REPORT = PROJECT / "05_reports" / "cpec_2018_reliability_aware_susceptibility_and_exposure_report.md"
PACKAGE_MAPS = PACKAGE / "04_probability_maps_250m" / "04_reliability_aware_final_outputs"
PACKAGE_TABLES = PACKAGE / "02_model_performance_tables"
PACKAGE_FIGS = PACKAGE / "03_figures" / "07_reliability_aware_final_outputs"
PACKAGE_METHODS = PACKAGE / "01_methods_and_decisions"
PACKAGE_SCRIPTS = PACKAGE / "07_reproducible_scripts"
PACKAGE_ROADS = PACKAGE / "05_road_exposure_outputs" / "reliability_aware_2018"

BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)
PAK_ADMIN1 = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\gadm41_PAK_shp\gadm41_PAK_1.shp"
)
KASHGAR = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\New Folder\cpecpart.shp"
)
CPEC_DATA = Path(r"D:\CPEC data")

NODATA_FLOAT = np.float32(-9999.0)
NODATA_UINT8 = np.uint8(255)
WGS84 = CRS.from_epsg(4326)
DISTANCE_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=32.55 +lon_0=70.39 +datum=WGS84 +units=m +no_defs"
)
GEOD = Geod(ellps="WGS84")
MAX_THREE_MODEL_STD = math.sqrt(2.0 / 9.0)

PUBLIC_DOMAINS = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]


@dataclass(frozen=True)
class FeatureSet:
    key: str
    label: str
    prob_path: Path
    confidence_path: Path
    aoa_path: Path


FEATURE_SETS = [
    FeatureSet(
        key="conventional",
        label="Conventional",
        prob_path=PROB_DIR / "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
        confidence_path=AOA_DIR / "cpec_2018_conventional_transfer_confidence_250m.tif",
        aoa_path=AOA_DIR / "cpec_2018_conventional_in_area_of_applicability_250m.tif",
    ),
    FeatureSet(
        key="alphaearth_embeddings",
        label="AlphaEarth Embeddings",
        prob_path=PROB_DIR / "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        confidence_path=AOA_DIR / "cpec_2018_alphaearth_embeddings_transfer_confidence_250m.tif",
        aoa_path=AOA_DIR / "cpec_2018_alphaearth_embeddings_in_area_of_applicability_250m.tif",
    ),
    FeatureSet(
        key="conventional_alphaearth_embeddings",
        label="Conventional + AlphaEarth Embeddings",
        prob_path=PROB_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        confidence_path=AOA_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_transfer_confidence_250m.tif",
        aoa_path=AOA_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_in_area_of_applicability_250m.tif",
    ),
]

FUSED = FEATURE_SETS[2]
DISAGREEMENT_STD = HIGH_IMPACT_DIR / "cpec_2018_stacked_probability_disagreement_std_250m.tif"
THRESHOLDS_JSON = HIGH_IMPACT_DIR / "stacked_250m_thresholds_and_diagnostics.json"


def ensure_dirs() -> None:
    for path in [
        OUT_DIR,
        FIG_DIR,
        PACKAGE_MAPS,
        PACKAGE_TABLES,
        PACKAGE_FIGS,
        PACKAGE_METHODS,
        PACKAGE_SCRIPTS,
        PACKAGE_ROADS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_boundary(crs: CRS | str | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.from_epsg(32642), allow_override=True)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def rasterize_mask(profile: dict, geoms, dtype: str = "uint8") -> np.ndarray:
    return features.rasterize(
        [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        default_value=1,
        all_touched=True,
        dtype=dtype,
    )


def study_mask(profile: dict) -> np.ndarray:
    return rasterize_mask(profile, read_boundary(profile["crs"]).geometry).astype(bool)


def domain_polygons(crs: CRS | str) -> gpd.GeoDataFrame:
    pak = gpd.read_file(PAK_ADMIN1).to_crs(crs)
    kas = gpd.read_file(KASHGAR).to_crs(crs)
    mapping = {
        "Kashgar (Xinjiang, China)": "Kashgar (Xinjiang, China)",
        "Gilgit-Baltistan": "Gilgit-Baltistan",
        "Balochistan": "Balochistan",
        "Khyber-Pakhtunkhwa": "KP-AJK",
        "Federally Administered Tribal Ar": "KP-AJK",
        "Azad Kashmir": "KP-AJK",
        "Punjab": "Punjab-Sindh lowland corridor",
        "Islamabad": "Punjab-Sindh lowland corridor",
        "Sindh": "Punjab-Sindh lowland corridor",
    }
    rows = []
    for _, row in pak.iterrows():
        domain = mapping.get(row["NAME_1"])
        if domain:
            rows.append({"domain": domain, "admin_unit": row["NAME_1"], "geometry": row.geometry})
    for geom in kas.geometry:
        rows.append(
            {
                "domain": "Kashgar (Xinjiang, China)",
                "admin_unit": "Kashgar",
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=pak.crs)


def domain_masks(profile: dict) -> dict[str, np.ndarray]:
    domains = domain_polygons(profile["crs"])
    return {
        domain: rasterize_mask(profile, domains.loc[domains["domain"] == domain, "geometry"]).astype(bool)
        for domain in PUBLIC_DOMAINS
    }


def read_float(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile.copy()


def read_uint(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
        if src.nodata is not None:
            arr[arr == src.nodata] = 0
        return arr.astype("uint8")


def write_float(path: Path, arr: np.ndarray, profile: dict, mask: np.ndarray, desc: str) -> None:
    out = arr.astype("float32", copy=True)
    out[~mask | ~np.isfinite(out)] = NODATA_FLOAT
    prof = profile.copy()
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=float(NODATA_FLOAT),
        compress="deflate",
        predictor=2,
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, desc)


def write_uint(path: Path, arr: np.ndarray, profile: dict, mask: np.ndarray, desc: str) -> None:
    out = arr.astype("uint8", copy=True)
    out[~mask] = NODATA_UINT8
    prof = profile.copy()
    prof.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=int(NODATA_UINT8),
        compress="deflate",
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, desc)


def pixel_area_by_row_km2(profile: dict) -> np.ndarray:
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    col = max(0, min(width - 1, width // 2))
    areas = np.zeros(height, dtype="float64")
    for row in range(height):
        lon0, lat0 = transform * (col, row)
        lon1, lat1 = transform * (col + 1, row + 1)
        xs = [lon0, lon1, lon1, lon0]
        ys = [lat0, lat0, lat1, lat1]
        area_m2, _ = GEOD.polygon_area_perimeter(xs, ys)
        areas[row] = abs(area_m2) / 1_000_000.0
    return areas


def area_km2(mask: np.ndarray, row_areas: np.ndarray) -> float:
    return float((mask.sum(axis=1).astype("float64") * row_areas).sum())


def stats_values(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return {"mean": np.nan, "median": np.nan, "p80": np.nan, "p90": np.nan}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p80": float(np.percentile(vals, 80)),
        "p90": float(np.percentile(vals, 90)),
    }


def create_raster_products() -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame, dict]:
    thresholds = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
    fused_prob, profile = read_float(FUSED.prob_path)
    mask = study_mask(profile)
    domains = domain_masks(profile)
    row_areas = pixel_area_by_row_km2(profile)
    uncertainty, _ = read_float(DISAGREEMENT_STD)
    uncertainty_norm = np.clip(uncertainty / MAX_THREE_MODEL_STD, 0.0, 1.0)

    paths: dict[str, Path] = {}
    summary_rows = []
    feature_arrays = {}
    for fs in FEATURE_SETS:
        prob, _ = read_float(fs.prob_path)
        confidence, _ = read_float(fs.confidence_path)
        aoa = read_uint(fs.aoa_path)
        weighted = np.clip(prob * confidence, 0.0, 1.0).astype("float32")
        weighted_path = OUT_DIR / f"cpec_2018_{fs.key}_reliability_weighted_probability_250m.tif"
        write_float(weighted_path, weighted, profile, mask, f"{fs.key}_probability_times_transfer_confidence")
        paths[f"{fs.key}_weighted_probability"] = weighted_path
        feature_arrays[fs.key] = {"prob": prob, "confidence": confidence, "aoa": aoa, "weighted": weighted}
        valid_mask = mask & np.isfinite(prob) & np.isfinite(confidence)
        s_prob = stats_values(prob, valid_mask)
        s_conf = stats_values(confidence, valid_mask)
        s_weighted = stats_values(weighted, valid_mask)
        summary_rows.append(
            {
                "feature_set": fs.label,
                "study_area_valid_pixels": int(valid_mask.sum()),
                "mean_probability": s_prob["mean"],
                "mean_transfer_confidence": s_conf["mean"],
                "mean_reliability_weighted_probability": s_weighted["mean"],
                "p80_reliability_weighted_probability": s_weighted["p80"],
                "p90_reliability_weighted_probability": s_weighted["p90"],
                "inside_aoa_percent": float(100.0 * ((aoa == 1) & mask).sum() / mask.sum()),
            }
        )

    fused = feature_arrays[FUSED.key]
    fused_conf = fused["confidence"]
    fused_aoa = fused["aoa"]
    fused_weighted = fused["weighted"]
    conf_vals = fused_conf[mask & np.isfinite(fused_conf)]
    confidence_p25 = float(np.percentile(conf_vals, 25))
    confidence_p50 = float(np.percentile(conf_vals, 50))
    probability_p80 = float(thresholds["fused_probability_p80"])
    probability_p90 = float(thresholds["fused_probability_p90"])
    uncertainty_p50 = float(thresholds["uncertainty_std_p50"])
    uncertainty_p75 = float(thresholds["uncertainty_std_p75"])

    reliability_aware_score = np.clip(fused_prob * fused_conf * (1.0 - uncertainty_norm), 0.0, 1.0)
    score_path = OUT_DIR / "cpec_2018_final_fused_reliability_aware_susceptibility_score_250m.tif"
    write_float(score_path, reliability_aware_score, profile, mask, "final_fused_reliability_aware_susceptibility_score")
    paths["final_reliability_aware_score"] = score_path

    reliable_high = (
        (fused_prob >= probability_p80)
        & (fused_conf >= confidence_p50)
        & (fused_aoa == 1)
        & (uncertainty <= uncertainty_p50)
        & mask
    )
    reliable_very_high = (
        (fused_prob >= probability_p90)
        & (fused_conf >= confidence_p50)
        & (fused_aoa == 1)
        & mask
    )
    uncertain_high = (
        (fused_prob >= probability_p80)
        & (((fused_aoa == 0) | (fused_conf < confidence_p25)) | (uncertainty >= uncertainty_p75))
        & mask
    )
    field_verification_priority = np.clip(
        fused_prob * (1.0 - fused_conf) + 0.35 * uncertainty_norm * (fused_prob >= probability_p80),
        0.0,
        1.0,
    ).astype("float32")

    uint_outputs = {
        "final_reliable_high_mask": (
            OUT_DIR / "cpec_2018_final_reliable_high_susceptibility_mask_250m.tif",
            reliable_high.astype("uint8"),
            "final_reliable_high_susceptibility_mask",
        ),
        "final_reliable_very_high_mask": (
            OUT_DIR / "cpec_2018_final_reliable_very_high_susceptibility_mask_250m.tif",
            reliable_very_high.astype("uint8"),
            "final_reliable_very_high_susceptibility_mask",
        ),
        "final_uncertain_high_mask": (
            OUT_DIR / "cpec_2018_final_uncertain_high_susceptibility_mask_250m.tif",
            uncertain_high.astype("uint8"),
            "final_uncertain_high_susceptibility_mask",
        ),
    }
    for key, (path, arr, desc) in uint_outputs.items():
        write_uint(path, arr, profile, mask, desc)
        paths[key] = path

    field_path = OUT_DIR / "cpec_2018_final_field_verification_priority_score_250m.tif"
    write_float(field_path, field_verification_priority, profile, mask, "final_field_verification_priority_score")
    paths["field_verification_priority"] = field_path

    area_rows = []
    for domain, dmask in domains.items():
        zone = mask & dmask
        if not zone.any():
            continue
        zone_area = area_km2(zone, row_areas)
        rel_area = area_km2(zone & reliable_high, row_areas)
        very_rel_area = area_km2(zone & reliable_very_high, row_areas)
        uncertain_area = area_km2(zone & uncertain_high, row_areas)
        outside_aoa_area = area_km2(zone & (fused_aoa == 0), row_areas)
        area_rows.append(
            {
                "domain": domain,
                "area_km2": zone_area,
                "mean_fused_probability": float(np.nanmean(fused_prob[zone])),
                "mean_fused_transfer_confidence": float(np.nanmean(fused_conf[zone])),
                "mean_reliability_aware_score": float(np.nanmean(reliability_aware_score[zone])),
                "reliable_high_area_km2": rel_area,
                "reliable_high_area_percent": 100.0 * rel_area / zone_area,
                "reliable_very_high_area_km2": very_rel_area,
                "uncertain_high_area_km2": uncertain_area,
                "uncertain_high_area_percent": 100.0 * uncertain_area / zone_area,
                "outside_fused_aoa_area_km2": outside_aoa_area,
                "outside_fused_aoa_percent": 100.0 * outside_aoa_area / zone_area,
            }
        )

    overall_df = pd.DataFrame(summary_rows)
    area_df = pd.DataFrame(area_rows)
    diagnostics = {
        "probability_p80_threshold": probability_p80,
        "probability_p90_threshold": probability_p90,
        "fused_transfer_confidence_p25": confidence_p25,
        "fused_transfer_confidence_p50": confidence_p50,
        "uncertainty_std_p50": uncertainty_p50,
        "uncertainty_std_p75": uncertainty_p75,
        "reliable_high_rule": "probability>=P80, confidence>=P50, inside AOA, disagreement<=P50",
        "reliable_very_high_rule": "probability>=P90, confidence>=P50, inside AOA",
        "uncertain_high_rule": "probability>=P80 and (outside AOA or confidence<P25 or disagreement>=P75)",
    }
    (OUT_DIR / "cpec_2018_reliability_aware_thresholds.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    paths["thresholds_json"] = OUT_DIR / "cpec_2018_reliability_aware_thresholds.json"
    overall_df.to_csv(OUT_DIR / "cpec_2018_reliability_aware_feature_set_summary.csv", index=False)
    area_df.to_csv(OUT_DIR / "cpec_2018_reliability_aware_subdomain_area_summary.csv", index=False)
    return paths, overall_df, area_df, diagnostics


def find_road_source() -> Path:
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
        candidates = [p for p in CPEC_DATA.rglob("*.shp") if p.stat().st_size == 451176]
    if not candidates:
        raise FileNotFoundError("Could not find the 2018 CPEC road network shapefile.")
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


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
    roads = gpd.clip(roads, read_boundary(roads.crs))
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy().to_crs(DISTANCE_CRS)
    rows = []
    seg_id = 1
    for _, row in roads.iterrows():
        code = route_code(row)
        attrs = {
            "route_code": code,
            "is_kkh_proxy": bool(is_kkh_proxy(code)),
            "road_type": row.get("type", None),
            "road_rank": row.get("rank", None),
        }
        for line in flatten_lines(row.geometry):
            length = float(line.length)
            if length <= 0:
                continue
            for start in np.arange(0.0, length, segment_length_m):
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
    segments = gpd.GeoDataFrame(rows, geometry="geometry", crs=DISTANCE_CRS)
    segments.attrs["road_source"] = str(roads_path)
    return segments


def assign_domains_to_segments(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids = segments.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs(WGS84)
    domains = domain_polygons(WGS84)
    joined = gpd.sjoin(centroids, domains[["domain", "geometry"]], how="left", predicate="within")
    out = segments.copy()
    out["domain"] = joined["domain"].to_numpy()
    out["domain"] = out["domain"].fillna("Unassigned")
    return out


def sample_array_at_points(src: rasterio.DatasetReader, points_wgs84: list) -> np.ndarray:
    coords = [(pt.x, pt.y) for pt in points_wgs84]
    vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
    if src.nodata is not None:
        vals[vals == src.nodata] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return vals


def create_road_exposure(paths: dict[str, Path]) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    segments = assign_domains_to_segments(build_road_segments())
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
            if 0 <= d <= length:
                sample_points.append(geom.interpolate(d))
                sample_segment_ids.append(row.segment_id)
    pts = gpd.GeoDataFrame({"segment_id": sample_segment_ids}, geometry=sample_points, crs=DISTANCE_CRS).to_crs(WGS84)
    raster_paths = {
        "fused_probability": FUSED.prob_path,
        "fused_transfer_confidence": FUSED.confidence_path,
        "reliability_weighted_probability": paths[f"{FUSED.key}_weighted_probability"],
        "final_reliability_aware_score": paths["final_reliability_aware_score"],
        "field_verification_priority": paths["field_verification_priority"],
        "reliable_high_mask": paths["final_reliable_high_mask"],
        "reliable_very_high_mask": paths["final_reliable_very_high_mask"],
        "uncertain_high_mask": paths["final_uncertain_high_mask"],
        "fused_aoa_mask": FUSED.aoa_path,
        "model_disagreement_std": DISAGREEMENT_STD,
    }
    sampled = pd.DataFrame({"segment_id": sample_segment_ids})
    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            sampled[name] = sample_array_at_points(src, list(pts.geometry))
    for col in ["reliable_high_mask", "reliable_very_high_mask", "uncertain_high_mask", "fused_aoa_mask"]:
        sampled[col] = sampled[col].fillna(0).astype(float)

    agg = sampled.groupby("segment_id").agg(
        sample_n=("fused_probability", "size"),
        mean_fused_probability=("fused_probability", "mean"),
        max_fused_probability=("fused_probability", "max"),
        mean_fused_transfer_confidence=("fused_transfer_confidence", "mean"),
        mean_reliability_weighted_probability=("reliability_weighted_probability", "mean"),
        mean_final_reliability_aware_score=("final_reliability_aware_score", "mean"),
        mean_field_verification_priority=("field_verification_priority", "mean"),
        reliable_high_share=("reliable_high_mask", "mean"),
        reliable_very_high_share=("reliable_very_high_mask", "mean"),
        uncertain_high_share=("uncertain_high_mask", "mean"),
        fused_aoa_share=("fused_aoa_mask", "mean"),
        mean_model_disagreement_std=("model_disagreement_std", "mean"),
    )
    segments = segments.merge(agg, left_on="segment_id", right_index=True, how="left")
    segments["outside_aoa_share"] = 1.0 - segments["fused_aoa_share"].fillna(0)
    segments["reliability_aware_road_priority_score"] = (
        segments["mean_final_reliability_aware_score"].fillna(0)
        + 0.35 * segments["reliable_high_share"].fillna(0)
        + 0.15 * segments["reliable_very_high_share"].fillna(0)
    )
    segments["field_verification_road_priority_score"] = (
        segments["mean_field_verification_priority"].fillna(0)
        + 0.25 * segments["uncertain_high_share"].fillna(0)
        + 0.10 * segments["outside_aoa_share"].fillna(0)
    )
    segments = segments.sort_values("reliability_aware_road_priority_score", ascending=False).reset_index(drop=True)
    segments["hotspot_rank"] = np.arange(1, len(segments) + 1)

    summary_rows = []
    for road_group, gdf in [
        ("All clipped roads", segments),
        ("KKH proxy routes (N35 / 314)", segments[segments["is_kkh_proxy"]]),
    ]:
        if gdf.empty:
            continue
        for domain in ["All domains", *PUBLIC_DOMAINS]:
            sub = gdf if domain == "All domains" else gdf[gdf["domain"] == domain]
            if sub.empty:
                continue
            length = sub["length_km"].fillna(0)
            total = float(length.sum())
            summary_rows.append(
                {
                    "road_group": road_group,
                    "domain": domain,
                    "segment_count": int(len(sub)),
                    "total_length_km": total,
                    "reliable_high_weighted_length_km": float((length * sub["reliable_high_share"].fillna(0)).sum()),
                    "reliable_very_high_weighted_length_km": float(
                        (length * sub["reliable_very_high_share"].fillna(0)).sum()
                    ),
                    "uncertain_high_weighted_length_km": float((length * sub["uncertain_high_share"].fillna(0)).sum()),
                    "outside_aoa_weighted_length_km": float((length * sub["outside_aoa_share"].fillna(0)).sum()),
                    "mean_probability_length_weighted": float(
                        np.average(sub["mean_fused_probability"].fillna(0), weights=length)
                    ),
                    "mean_transfer_confidence_length_weighted": float(
                        np.average(sub["mean_fused_transfer_confidence"].fillna(0), weights=length)
                    ),
                    "mean_reliability_aware_score_length_weighted": float(
                        np.average(sub["mean_final_reliability_aware_score"].fillna(0), weights=length)
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    sample_csv = OUT_DIR / "reliability_aware_road_sample_points_250m.csv"
    all_csv = OUT_DIR / "reliability_aware_road_segment_exposure_all_250m.csv"
    top_csv = OUT_DIR / "reliability_aware_road_hotspot_segments_top50_250m.csv"
    field_csv = OUT_DIR / "field_verification_priority_road_segments_top50_250m.csv"
    summary_csv = OUT_DIR / "reliability_aware_road_exposure_summary_by_domain_250m.csv"
    gpkg = OUT_DIR / "reliability_aware_road_segment_exposure_all_250m.gpkg"
    geojson = OUT_DIR / "reliability_aware_road_hotspot_segments_top50_250m.geojson"
    sampled.to_csv(sample_csv, index=False)
    segments.drop(columns="geometry").to_csv(all_csv, index=False)
    segments.head(50).drop(columns="geometry").to_csv(top_csv, index=False)
    segments.sort_values("field_verification_road_priority_score", ascending=False).head(50).drop(columns="geometry").to_csv(
        field_csv, index=False
    )
    summary.to_csv(summary_csv, index=False)
    segments.to_file(gpkg, layer="reliability_aware_road_exposure", driver="GPKG")
    segments.head(50).to_crs(WGS84).to_file(geojson, driver="GeoJSON")
    road_paths = [sample_csv, all_csv, top_csv, field_csv, summary_csv, gpkg, geojson]
    return segments, summary, sampled, road_paths


def downsample(arr: np.ndarray, mask: np.ndarray, factor: int = 6) -> np.ndarray:
    out = arr[::factor, ::factor].astype("float32", copy=True)
    submask = mask[::factor, ::factor]
    out[~submask] = np.nan
    return out


def extent_from_profile(profile: dict) -> list[float]:
    transform = profile["transform"]
    return [
        transform.c,
        transform.c + transform.a * profile["width"],
        transform.f + transform.e * profile["height"],
        transform.f,
    ]


def save_fig(fig: plt.Figure, name: str) -> Path:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    shutil.copy2(png, PACKAGE_FIGS / png.name)
    shutil.copy2(pdf, PACKAGE_FIGS / pdf.name)
    return png


def make_figures(paths: dict[str, Path], area_df: pd.DataFrame, road_segments: gpd.GeoDataFrame) -> list[Path]:
    fused_prob, profile = read_float(FUSED.prob_path)
    mask = study_mask(profile)
    score, _ = read_float(paths["final_reliability_aware_score"])
    reliable_high = read_uint(paths["final_reliable_high_mask"]).astype("float32")
    uncertain_high = read_uint(paths["final_uncertain_high_mask"]).astype("float32")
    bnd = read_boundary(WGS84)
    extent = extent_from_profile(profile)
    fig_paths = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8))
    panels = [
        ("Fused Probability", fused_prob, "viridis", 0, 1),
        ("Reliability-Aware Score", score, "YlGnBu", 0, 1),
        ("Uncertain High Mask", uncertain_high, "magma", 0, 1),
    ]
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(downsample(arr, mask), extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        bnd.boundary.plot(ax=ax, color="black", linewidth=0.25)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("CPEC 2018 Reliability-Aware Susceptibility Outputs", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig_paths.append(save_fig(fig, "figure_cpec_2018_reliability_aware_susceptibility_outputs"))
    plt.close(fig)

    plot_df = area_df.set_index("domain").reindex(PUBLIC_DOMAINS)
    metrics = [
        "reliable_high_area_percent",
        "uncertain_high_area_percent",
        "outside_fused_aoa_percent",
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    vals = plot_df[metrics].to_numpy()
    im = ax.imshow(vals, vmin=0, vmax=max(1.0, float(np.nanmax(vals))), cmap="YlGnBu")
    ax.set_yticks(
        np.arange(len(PUBLIC_DOMAINS)),
        [
            "Kashgar\n(Xinjiang, China)",
            "Gilgit-Baltistan",
            "KP-AJK",
            "Balochistan",
            "Punjab-Sindh\nlowland corridor",
        ],
    )
    ax.set_xticks(
        np.arange(len(metrics)),
        ["Reliable high\narea %", "Uncertain high\narea %", "Outside fused\nAOA %"],
    )
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.1f}%", ha="center", va="center", fontsize=8, color="#0f172a")
    ax.set_title("Reliability-Aware Area Summary by CPEC Subdomain", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Area percentage", shrink=0.82)
    fig.tight_layout()
    fig_paths.append(save_fig(fig, "figure_cpec_2018_reliability_aware_area_by_subdomain"))
    plt.close(fig)

    all_roads = road_segments.to_crs(WGS84)
    top = all_roads.head(50)
    kkh = all_roads[all_roads["is_kkh_proxy"]]
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    im = ax.imshow(downsample(score, mask), extent=extent, origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
    all_roads.plot(ax=ax, color="#333333", linewidth=0.10, alpha=0.25)
    if not kkh.empty:
        kkh.plot(ax=ax, color="#0072B2", linewidth=0.65, alpha=0.80)
    top.plot(
        ax=ax,
        column="reliability_aware_road_priority_score",
        cmap="plasma",
        linewidth=1.1,
        legend=True,
        legend_kwds={"label": "Road priority score", "shrink": 0.65},
    )
    bnd.boundary.plot(ax=ax, color="black", linewidth=0.35)
    fig.colorbar(im, ax=ax, label="Reliability-aware susceptibility score", shrink=0.72)
    ax.set_title("CPEC/KKH Reliability-Aware Road Hotspots", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig_paths.append(save_fig(fig, "figure_cpec_2018_reliability_aware_road_hotspots"))
    plt.close(fig)
    return fig_paths


def markdown_table(df: pd.DataFrame, decimals: int = 3, max_rows: int | None = None) -> str:
    view = df.head(max_rows).copy() if max_rows else df.copy()
    for c in view.select_dtypes(include=[np.floating]).columns:
        view[c] = view[c].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    view = view.fillna("")
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    overall_df: pd.DataFrame,
    area_df: pd.DataFrame,
    road_summary: pd.DataFrame,
    road_segments: gpd.GeoDataFrame,
    diagnostics: dict,
    paths: dict[str, Path],
    road_paths: list[Path],
    fig_paths: list[Path],
) -> None:
    top_cols = [
        "hotspot_rank",
        "route_code",
        "domain",
        "is_kkh_proxy",
        "length_km",
        "mean_fused_probability",
        "mean_fused_transfer_confidence",
        "mean_final_reliability_aware_score",
        "reliable_high_share",
        "uncertain_high_share",
        "reliability_aware_road_priority_score",
    ]
    report = f"""# CPEC 2018 Reliability-Aware Susceptibility and Exposure Outputs

Date: 2026-05-10

## Purpose

This step converts the 2018 stacked-ensemble probability maps and transfer-confidence maps into decision-ready products. The goal is to separate high susceptibility that is reliable from high susceptibility that should be treated as uncertain and prioritized for field verification.

## Corrected Public Domain Labels

- Khyber Pakhtunkhwa and Azad Jammu and Kashmir are displayed as `KP-AJK` for the project figures and tables.
- Former FATA is not shown as a separate current public domain because it was merged with Khyber Pakhtunkhwa through Pakistan's 25th Constitutional Amendment in 2018; legacy GADM polygons labelled FATA are assigned to the KP side of the `KP-AJK` analysis domain.
- AJK is grouped with KP only as an analysis-domain decision for the northern-western mountainous corridor and to avoid a statistically small held-out test region; it is not presented as the same administrative unit.
- Punjab, Islamabad, and Sindh are displayed as `Punjab-Sindh lowland corridor`. Islamabad is administratively a federal territory, but it is absorbed into this lowland corridor only for statistical grouping because there was only one Islamabad sample.
- Official sources commonly use `KP` for Khyber Pakhtunkhwa; this project therefore uses `KP-AJK` rather than a legacy FATA-separated label.

## Why These Subdomains Are Defensible

These subdomains are used as **analysis domains for model transferability**, not as a replacement for official political boundaries. The grouping was necessary because leave-one-domain-out testing requires both landslide and non-landslide samples in every held-out region. Small units such as Islamabad and legacy FATA cannot support stable independent testing alone.

The design follows spatial-validation and transferability literature: spatially structured data should be evaluated with block/group validation rather than ordinary random splits, and the area where model error is expected to transfer should be explicitly assessed.

Key support:

- Roberts et al. (2017), Ecography, cross-validation strategies for spatial/hierarchical data: https://doi.org/10.1111/ecog.02881
- Meyer and Pebesma (2021), area of applicability for spatial prediction models: https://doi.org/10.1111/2041-210X.13650
- Official KP government portal: https://kp.gov.pk/
- Pakistan Code / Constitution source for the 25th Amendment context: https://pakistancode.gov.pk/
- AJK official portal: https://ajk.gov.pk/

## Final Rules

```json
{json.dumps(diagnostics, indent=2)}
```

The final score is:

`reliability-aware susceptibility = fused probability x fused transfer confidence x (1 - normalized model disagreement)`

## Feature-Set Summary

{markdown_table(overall_df)}

## Subdomain Area Summary

{markdown_table(area_df)}

## Road Exposure Summary

{markdown_table(road_summary)}

## Top 15 Reliability-Aware Road Hotspots

{markdown_table(road_segments[top_cols], max_rows=15)}

## Main Raster Outputs

""" + "\n".join(f"- `{path}`" for path in paths.values()) + f"""

## Road Outputs

""" + "\n".join(f"- `{path}`" for path in road_paths) + f"""

## Figures

""" + "\n".join(f"- `{path}`" for path in fig_paths) + """

## Interpretation

- The fused reliability-aware score is the preferred final planning surface because it keeps probability continuous but penalizes low transfer confidence and high model disagreement.
- Reliable high-susceptibility zones are the strongest areas to report as robust hotspots.
- Uncertain high-susceptibility zones are not discarded; they become priority areas for field checking, inventory improvement, and local validation.
- Road exposure is a planning overlay, not an independent validation, because distance to roads is one of the conditioning factors.
"""
    REPORT.write_text(report, encoding="utf-8")
    (PACKAGE_METHODS / "cpec_2018_reliability_aware_susceptibility_and_exposure_report.md").write_text(
        report, encoding="utf-8"
    )


def copy_outputs(paths: dict[str, Path], road_paths: list[Path], overall_df: pd.DataFrame, area_df: pd.DataFrame, road_summary: pd.DataFrame) -> None:
    for path in paths.values():
        shutil.copy2(path, PACKAGE_MAPS / path.name)
    for path in road_paths:
        shutil.copy2(path, PACKAGE_ROADS / path.name)
    overall_df.to_csv(PACKAGE_TABLES / "cpec_2018_reliability_aware_feature_set_summary.csv", index=False)
    area_df.to_csv(PACKAGE_TABLES / "cpec_2018_reliability_aware_subdomain_area_summary.csv", index=False)
    road_summary.to_csv(PACKAGE_TABLES / "cpec_2018_reliability_aware_road_exposure_summary_by_domain.csv", index=False)
    shutil.copy2(Path(__file__), PACKAGE_SCRIPTS / Path(__file__).name)


def main() -> None:
    ensure_dirs()
    paths, overall_df, area_df, diagnostics = create_raster_products()
    road_segments, road_summary, sampled, road_paths = create_road_exposure(paths)
    fig_paths = make_figures(paths, area_df, road_segments)
    copy_outputs(paths, road_paths, overall_df, area_df, road_summary)
    write_report(overall_df, area_df, road_summary, road_segments, diagnostics, paths, road_paths, fig_paths)
    print(f"Saved reliability-aware outputs to: {OUT_DIR}")
    print(f"Saved package copies to: {PACKAGE_MAPS}")


if __name__ == "__main__":
    main()
