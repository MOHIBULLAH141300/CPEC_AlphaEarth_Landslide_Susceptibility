"""Build a dated landslide inventory branch for temporal validation.

This script standardizes dated landslide sources, clips them to the official
CPEC boundary, removes likely duplicate/training-overlap records, and checks
the annual 2017-2024 susceptibility maps against independent dated events.

The output is designed as a defensible validation/QA branch, not as a new model
training step.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", os.environ["PROJ_DATA"])
os.environ.setdefault("PROJ_NETWORK", "OFF")

try:
    from pyproj import datadir

    datadir.set_data_dir(os.environ["PROJ_DATA"])
except Exception:
    pass

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point


PROJECT_ROOT = Path(r"D:\DING PROJECT")
WORKSPACE_ROOT = Path(r"C:\Users\Administrator\Desktop\cpec landslides")

DATED_ROOT = PROJECT_ROOT / "01_clean_data" / "new_step_required_data" / "B4_dated_inventory"
OUT_ROOT = DATED_ROOT / "processed_cpec_temporal_validation_2026-06-05"
REPORT_DIR = PROJECT_ROOT / "05_reports"
FIG_DIR = REPORT_DIR / "figures" / "dated_temporal_validation_2026-06-05"
ANNUAL_DIR = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"

BOUNDARY = WORKSPACE_ROOT / "cpec boundary" / "cpec boundary" / "CPEC_BOUNDARY.shp"
HMA_DIR = WORKSPACE_ROOT / "HMA_LS_Cat_2-20260421_073021"
HMA_POINT = HMA_DIR / "HMA_LS_Cat_point_v02.0.shp"
HMA_POLY = HMA_DIR / "HMA_LS_Cat_poly_v02.0.shp"
GLC_CSV = DATED_ROOT / "NASA_Global_Landslide_Catalog_Export_rows.csv"
COOLR_REPORTS_CSV = DATED_ROOT / "NASA_COOLR_Reports_Points_CPEC_bbox_60_20_80_42.csv"
COOLR_EVENTS_CSV = DATED_ROOT / "NASA_COOLR_Events_Points_CPEC_bbox_60_20_80_42.csv"
SAMPLES_V3 = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"

YEARS = list(range(2017, 2025))
MODEL_SETS = {
    "Conventional": "conventional",
    "AlphaEarth Embeddings": "alphaearth_embeddings",
    "Conventional + AlphaEarth Embeddings": "conventional_alphaearth_embeddings",
}
PRIMARY_MODEL = "Conventional + AlphaEarth Embeddings"
RASTER_NODATA = -9999.0
BACKGROUND_SAMPLE_SIZE = 200_000
AUC_BACKGROUND_SAMPLE_SIZE = 50_000
DEDUP_DISTANCE_M = 1000.0
DEDUP_DATE_WINDOW_DAYS = 30
TRAINING_OVERLAP_STRICT_M = 1000.0
TRAINING_OVERLAP_REPORTED_M = 500.0
RANDOM_SEED = 20260605


@dataclass(frozen=True)
class OutputPaths:
    standardized_csv: Path
    standardized_gpkg: Path
    deduplicated_csv: Path
    deduplicated_gpkg: Path
    validation_csv: Path
    validation_gpkg: Path
    event_probability_csv: Path
    summary_csv: Path
    raster_stats_csv: Path
    dedup_log_csv: Path
    source_qa_csv: Path
    report_md: Path


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def outputs() -> OutputPaths:
    return OutputPaths(
        standardized_csv=OUT_ROOT / "cpec_dated_inventory_all_sources_clipped_standardized.csv",
        standardized_gpkg=OUT_ROOT / "cpec_dated_inventory_all_sources_clipped_standardized.gpkg",
        deduplicated_csv=OUT_ROOT / "cpec_dated_inventory_deduplicated.csv",
        deduplicated_gpkg=OUT_ROOT / "cpec_dated_inventory_deduplicated.gpkg",
        validation_csv=OUT_ROOT / "cpec_dated_inventory_temporal_validation_candidates_2017_2024.csv",
        validation_gpkg=OUT_ROOT / "cpec_dated_inventory_temporal_validation_candidates_2017_2024.gpkg",
        event_probability_csv=OUT_ROOT / "annual_temporal_validation_event_probability_table.csv",
        summary_csv=OUT_ROOT / "annual_temporal_validation_summary_by_year_and_model.csv",
        raster_stats_csv=OUT_ROOT / "annual_temporal_validation_raster_thresholds_by_year_and_model.csv",
        dedup_log_csv=OUT_ROOT / "dated_inventory_deduplication_log.csv",
        source_qa_csv=OUT_ROOT / "dated_inventory_source_qa_counts.csv",
        report_md=REPORT_DIR / "dated_inventory_temporal_validation_2026-06-05.md",
    )


def read_boundary() -> gpd.GeoDataFrame:
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:32642")
    return boundary.to_crs("EPSG:4326")


def boundary_geometry(boundary: gpd.GeoDataFrame):
    if hasattr(boundary.geometry, "union_all"):
        return boundary.geometry.union_all()
    return boundary.geometry.unary_union


def safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_date(series: pd.Series, *, epoch_ms: bool = False) -> pd.Series:
    if epoch_ms:
        return pd.to_datetime(series, unit="ms", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def location_quality(value) -> str:
    text = safe_str(value).lower().replace("_", " ").replace("-", " ")
    if not text:
        return "unknown"
    high_terms = ["exact", "known", "1km", "1 km", "5km", "5 km"]
    medium_terms = ["10km", "10 km", "25km", "25 km", "approx"]
    low_terms = ["50km", "50 km", "100km", "100 km", "unknown", "other"]
    if any(term in text for term in high_terms):
        return "high"
    if any(term in text for term in medium_terms):
        return "medium"
    if any(term in text for term in low_terms):
        return "low"
    return "unknown"


def source_rank(source_dataset: str) -> int:
    ranks = {
        "HMA point": 1,
        "HMA polygon representative point": 2,
        "NASA COOLR reports": 3,
        "NASA GLC legacy": 4,
        "NASA COOLR events": 5,
    }
    return ranks.get(source_dataset, 9)


def quality_rank(quality: str) -> int:
    ranks = {"high": 0, "medium": 1, "unknown": 2, "low": 3}
    return ranks.get(quality, 4)


def clean_lon_lat(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    out = df.copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    valid = (
        out[lon_col].notna()
        & out[lat_col].notna()
        & out[lon_col].between(-180, 180)
        & out[lat_col].between(-90, 90)
    )
    return out.loc[valid].copy()


def standardize_point_source(
    df: pd.DataFrame,
    source_dataset: str,
    *,
    lon_col: str,
    lat_col: str,
    date_col: str,
    epoch_ms: bool = False,
    event_id_cols: Iterable[str] = ("event_id", "ev_id"),
    title_cols: Iterable[str] = ("event_title", "ev_title"),
    source_name_cols: Iterable[str] = ("source_name", "src_name"),
    source_link_cols: Iterable[str] = ("source_link", "src_link"),
    accuracy_cols: Iterable[str] = ("location_accuracy", "loc_accu"),
    category_cols: Iterable[str] = ("landslide_category", "ls_cat"),
    trigger_cols: Iterable[str] = ("landslide_trigger", "ls_trig"),
    size_cols: Iterable[str] = ("landslide_size", "ls_size"),
    country_cols: Iterable[str] = ("country_name", "ctry_name"),
    country_code_cols: Iterable[str] = ("country_code", "ctry_code"),
) -> gpd.GeoDataFrame:
    df = clean_lon_lat(df, lon_col, lat_col)
    event_id_col = first_existing(df, event_id_cols)
    title_col = first_existing(df, title_cols)
    source_name_col = first_existing(df, source_name_cols)
    source_link_col = first_existing(df, source_link_cols)
    accuracy_col = first_existing(df, accuracy_cols)
    category_col = first_existing(df, category_cols)
    trigger_col = first_existing(df, trigger_cols)
    size_col = first_existing(df, size_cols)
    country_col = first_existing(df, country_cols)
    country_code_col = first_existing(df, country_code_cols)
    event_date = parse_date(df[date_col], epoch_ms=epoch_ms)

    out = pd.DataFrame(
        {
            "source_dataset": source_dataset,
            "source_event_id": df[event_id_col].map(safe_str) if event_id_col else "",
            "source_name": df[source_name_col].map(safe_str) if source_name_col else "",
            "source_link": df[source_link_col].map(safe_str) if source_link_col else "",
            "event_title": df[title_col].map(safe_str) if title_col else "",
            "event_date": event_date,
            "event_year": event_date.dt.year.astype("Int64"),
            "event_year_quality": np.where(event_date.notna(), "dated", "missing"),
            "location_accuracy": df[accuracy_col].map(safe_str) if accuracy_col else "",
            "location_quality": (
                df[accuracy_col].map(location_quality) if accuracy_col else "unknown"
            ),
            "landslide_category": df[category_col].map(safe_str) if category_col else "",
            "landslide_trigger": df[trigger_col].map(safe_str) if trigger_col else "",
            "landslide_size": df[size_col].map(safe_str) if size_col else "",
            "country_name": df[country_col].map(safe_str) if country_col else "",
            "country_code": df[country_code_col].map(safe_str) if country_code_col else "",
            "longitude": df[lon_col].astype(float),
            "latitude": df[lat_col].astype(float),
        }
    )
    out["raw_source_row"] = np.arange(len(out), dtype=int)
    out["source_rank"] = out["source_dataset"].map(source_rank)
    out["quality_rank"] = out["location_quality"].map(quality_rank)
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["longitude"], out["latitude"]),
        crs="EPSG:4326",
    )


def load_hma_point() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(HMA_POINT)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    return standardize_point_source(
        pd.DataFrame(gdf.drop(columns="geometry")),
        "HMA point",
        lon_col="longitude",
        lat_col="latitude",
        date_col="ev_date",
        event_id_cols=("ev_id",),
        title_cols=("ev_title",),
        source_name_cols=("src_name",),
        source_link_cols=("src_link",),
        accuracy_cols=("loc_accu",),
        category_cols=("ls_cat",),
        trigger_cols=("ls_trig",),
        size_cols=("ls_size",),
        country_cols=("ctry_name",),
        country_code_cols=("ctry_code",),
    )


def load_hma_polygon_representative() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(HMA_POLY)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    records = []
    geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry.representative_point()
        record = row.drop(labels=["geometry"]).to_dict()
        record["longitude"] = float(geom.x)
        record["latitude"] = float(geom.y)
        records.append(record)
        geoms.append(Point(geom.x, geom.y))
    tmp = pd.DataFrame(records)
    std = standardize_point_source(
        tmp,
        "HMA polygon representative point",
        lon_col="longitude",
        lat_col="latitude",
        date_col="ev_date",
        event_id_cols=("ev_id",),
        title_cols=("ev_title",),
        source_name_cols=("src_name",),
        source_link_cols=("src_link",),
        accuracy_cols=("loc_accu",),
        category_cols=("ls_cat",),
        trigger_cols=("ls_trig",),
        size_cols=("ls_size",),
        country_cols=("ctry_name",),
        country_code_cols=("ctry_code",),
    )
    std["geometry"] = geoms
    return std


def load_glc() -> gpd.GeoDataFrame:
    df = pd.read_csv(GLC_CSV, low_memory=False)
    return standardize_point_source(
        df,
        "NASA GLC legacy",
        lon_col="longitude",
        lat_col="latitude",
        date_col="event_date",
    )


def load_coolr_reports() -> gpd.GeoDataFrame:
    df = pd.read_csv(COOLR_REPORTS_CSV, low_memory=False)
    return standardize_point_source(
        df,
        "NASA COOLR reports",
        lon_col="longitude",
        lat_col="latitude",
        date_col="event_date",
        epoch_ms=True,
    )


def load_coolr_events() -> gpd.GeoDataFrame:
    df = pd.read_csv(COOLR_EVENTS_CSV, low_memory=False)
    return standardize_point_source(
        df,
        "NASA COOLR events",
        lon_col="longitude",
        lat_col="latitude",
        date_col="event_date",
        epoch_ms=True,
        category_cols=("landslide_category",),
        trigger_cols=("landslide_trigger",),
        size_cols=("landslide_size",),
    )


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = boundary_geometry(boundary)
    clipped = gdf.loc[gdf.geometry.within(geom) | gdf.geometry.intersects(geom)].copy()
    clipped["inside_official_cpec_boundary"] = True
    return clipped


def source_qa(unclipped: dict[str, gpd.GeoDataFrame], clipped: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    for source, gdf in unclipped.items():
        clipped_gdf = clipped[source]
        rows.append(
            {
                "source_dataset": source,
                "raw_records": int(len(gdf)),
                "clipped_to_cpec": int(len(clipped_gdf)),
                "dated_2017_2024_clipped": int(
                    clipped_gdf["event_year"].between(2017, 2024).sum()
                ),
                "records_with_event_year_clipped": int(clipped_gdf["event_year"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def concat_sources(clipped: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    cols = [
        "source_dataset",
        "source_event_id",
        "source_name",
        "source_link",
        "event_title",
        "event_date",
        "event_year",
        "event_year_quality",
        "location_accuracy",
        "location_quality",
        "landslide_category",
        "landslide_trigger",
        "landslide_size",
        "country_name",
        "country_code",
        "longitude",
        "latitude",
        "raw_source_row",
        "source_rank",
        "quality_rank",
        "inside_official_cpec_boundary",
        "geometry",
    ]
    frames = [gdf[cols].copy() for gdf in clipped.values()]
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["dated_record_id"] = [f"dated_{i:05d}" for i in range(1, len(out) + 1)]
    out["event_date_iso"] = out["event_date"].dt.strftime("%Y-%m-%d")
    out["event_date_numeric"] = out["event_date"].astype("int64")
    out.loc[out["event_date"].isna(), "event_date_numeric"] = np.nan
    return out


def best_record_indices(gdf: gpd.GeoDataFrame, group_indices: list[int]) -> int:
    subset = gdf.loc[group_indices].copy()
    subset["has_event_date"] = subset["event_date"].notna().astype(int)
    subset["has_title"] = subset["event_title"].str.len().gt(0).astype(int)
    subset = subset.sort_values(
        ["source_rank", "quality_rank", "has_event_date", "has_title", "dated_record_id"],
        ascending=[True, True, False, False, True],
    )
    return int(subset.index[0])


def date_gap_ok(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return True
    return abs((a - b).days) <= DEDUP_DATE_WINDOW_DAYS


def disjoint_set(n: int):
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    return parent, find, union


def spatial_date_deduplicate(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    dedup = gdf.copy()
    dedup["dedup_keep"] = True
    dedup["duplicate_cluster_id"] = ""
    dedup["duplicate_cluster_size"] = 1
    dedup["duplicate_reason"] = ""

    # Exact source event IDs are merged first where available.
    id_groups = {}
    for idx, value in dedup["source_event_id"].items():
        key = safe_str(value)
        if key:
            id_groups.setdefault(key, []).append(idx)
    for event_id, group in id_groups.items():
        if len(group) > 1:
            keep = best_record_indices(dedup, group)
            for idx in group:
                dedup.at[idx, "dedup_keep"] = idx == keep
                dedup.at[idx, "duplicate_cluster_id"] = f"event_id_{event_id}"
                dedup.at[idx, "duplicate_cluster_size"] = len(group)
                if idx != keep:
                    dedup.at[idx, "duplicate_reason"] = "same_source_event_id"

    remaining = dedup.loc[dedup["dedup_keep"] & dedup["event_year"].notna()].copy()
    if remaining.empty:
        log = dedup[["dated_record_id", "dedup_keep", "duplicate_cluster_id", "duplicate_cluster_size", "duplicate_reason"]]
        return dedup, log

    projected = remaining.to_crs("EPSG:32642")
    remaining["x_m"] = projected.geometry.x
    remaining["y_m"] = projected.geometry.y

    cluster_counter = 1
    for year, year_df in remaining.groupby("event_year", dropna=True):
        year_indices = list(year_df.index)
        n = len(year_indices)
        parent, find, union = disjoint_set(n)
        coords = year_df[["x_m", "y_m"]].to_numpy(float)
        dates = year_df["event_date"].tolist()
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if dist <= DEDUP_DISTANCE_M and date_gap_ok(dates[i], dates[j]):
                    union(i, j)
        groups: dict[int, list[int]] = {}
        for local_i, idx in enumerate(year_indices):
            groups.setdefault(find(local_i), []).append(idx)
        for group in groups.values():
            if len(group) <= 1:
                continue
            keep = best_record_indices(dedup, group)
            cluster_id = f"spatial_date_{int(year)}_{cluster_counter:04d}"
            cluster_counter += 1
            for idx in group:
                dedup.at[idx, "dedup_keep"] = idx == keep
                dedup.at[idx, "duplicate_cluster_id"] = cluster_id
                dedup.at[idx, "duplicate_cluster_size"] = len(group)
                if idx != keep:
                    dedup.at[idx, "duplicate_reason"] = (
                        f"same_year_within_{int(DEDUP_DISTANCE_M)}m_and_{DEDUP_DATE_WINDOW_DAYS}d"
                    )

    log = dedup[
        [
            "dated_record_id",
            "source_dataset",
            "source_event_id",
            "event_date_iso",
            "event_year",
            "dedup_keep",
            "duplicate_cluster_id",
            "duplicate_cluster_size",
            "duplicate_reason",
        ]
    ].copy()
    return dedup, log


def mark_training_overlap(dedup: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    samples = pd.read_csv(SAMPLES_V3, low_memory=False)
    positives = samples.loc[pd.to_numeric(samples["label"], errors="coerce") == 1].copy()
    positives = clean_lon_lat(positives, "longitude", "latitude")
    if positives.empty:
        dedup["nearest_2018_training_positive_m"] = np.nan
        dedup["overlaps_2018_training_positive_500m"] = False
        dedup["overlaps_2018_training_positive_1000m"] = False
        return dedup

    pos_gdf = gpd.GeoDataFrame(
        positives,
        geometry=gpd.points_from_xy(positives["longitude"], positives["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32642")
    event_proj = dedup.to_crs("EPSG:32642")
    pos_xy = np.column_stack([pos_gdf.geometry.x.to_numpy(), pos_gdf.geometry.y.to_numpy()])
    event_xy = np.column_stack([event_proj.geometry.x.to_numpy(), event_proj.geometry.y.to_numpy()])

    # A vectorized distance matrix is safe here because the clipped dated
    # inventory is small; it avoids adding another dependency.
    nearest = []
    chunk = 256
    for start in range(0, len(event_xy), chunk):
        arr = event_xy[start : start + chunk]
        dist = np.sqrt(((arr[:, None, :] - pos_xy[None, :, :]) ** 2).sum(axis=2))
        nearest.extend(dist.min(axis=1).tolist())
    dedup = dedup.copy()
    dedup["nearest_2018_training_positive_m"] = nearest
    dedup["overlaps_2018_training_positive_500m"] = (
        dedup["nearest_2018_training_positive_m"] <= TRAINING_OVERLAP_REPORTED_M
    )
    dedup["overlaps_2018_training_positive_1000m"] = (
        dedup["nearest_2018_training_positive_m"] <= TRAINING_OVERLAP_STRICT_M
    )
    return dedup


def raster_path(year: int, model_label: str) -> Path:
    slug = MODEL_SETS[model_label]
    return (
        ANNUAL_DIR
        / str(year)
        / "02_probability_maps_250m"
        / f"cpec_{year}_{slug}_stacked_ensemble_probability_250m.tif"
    )


def valid_values_from_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        mask &= arr != RASTER_NODATA
        valid = arr[mask].astype("float32")
        meta = {
            "valid_pixel_count": int(valid.size),
            "nodata_pixel_count": int(arr.size - valid.size),
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
            "resolution_x": float(abs(src.transform.a)),
            "resolution_y": float(abs(src.transform.e)),
        }
    return valid, meta


def sample_raster(path: Path, points: gpd.GeoDataFrame) -> np.ndarray:
    if points.empty:
        return np.array([], dtype=float)
    with rasterio.open(path) as src:
        pts = points.to_crs(src.crs) if points.crs != src.crs else points
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        nodata = src.nodata
    vals[~np.isfinite(vals)] = np.nan
    if nodata is not None:
        vals[np.isclose(vals, nodata)] = np.nan
    vals[np.isclose(vals, RASTER_NODATA)] = np.nan
    return vals


def auc_against_background(pos_scores: np.ndarray, background_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(background_scores, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def probability_diagnostics(validation: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    event_rows = []
    summary_rows = []
    stats_rows = []

    for year in YEARS:
        year_events = validation.loc[validation["event_year"].astype("Int64") == year].copy()
        for model_label in MODEL_SETS:
            path = raster_path(year, model_label)
            if not path.exists():
                continue
            valid, meta = valid_values_from_raster(path)
            p50, p80, p90, p95 = np.percentile(valid, [50, 80, 90, 95])
            stats_rows.append(
                {
                    "year": year,
                    "model_set": model_label,
                    "valid_pixel_count": meta["valid_pixel_count"],
                    "nodata_pixel_count": meta["nodata_pixel_count"],
                    "resolution_x": meta["resolution_x"],
                    "resolution_y": meta["resolution_y"],
                    "raster_mean_probability": float(np.mean(valid)),
                    "raster_median_probability": float(p50),
                    "raster_p80_probability": float(p80),
                    "raster_p90_probability": float(p90),
                    "raster_p95_probability": float(p95),
                }
            )

            if year_events.empty:
                summary_rows.append(
                    {
                        "validation_tier": "strict_independent",
                        "year": year,
                        "model_set": model_label,
                        "n_events": 0,
                        "mean_event_probability": np.nan,
                        "median_event_probability": np.nan,
                        "mean_event_percentile_rank": np.nan,
                        "median_event_percentile_rank": np.nan,
                        "hit_rate_above_p80": np.nan,
                        "hit_rate_above_p90": np.nan,
                        "background_separation_auc": np.nan,
                    }
                )
                continue

            sample_n = min(BACKGROUND_SAMPLE_SIZE, len(valid))
            bg_sample = rng.choice(valid, size=sample_n, replace=False)
            auc_n = min(AUC_BACKGROUND_SAMPLE_SIZE, len(bg_sample))
            bg_auc = rng.choice(bg_sample, size=auc_n, replace=False)
            probs = sample_raster(path, year_events)
            ranks = np.array([(bg_sample <= p).mean() if np.isfinite(p) else np.nan for p in probs])

            for event, prob, rank in zip(year_events.itertuples(), probs, ranks):
                event_rows.append(
                    {
                        "dated_record_id": event.dated_record_id,
                        "source_dataset": event.source_dataset,
                        "source_event_id": event.source_event_id,
                        "event_date": event.event_date_iso,
                        "event_year": year,
                        "event_title": event.event_title,
                        "location_quality": event.location_quality,
                        "landslide_category": event.landslide_category,
                        "nearest_2018_training_positive_m": event.nearest_2018_training_positive_m,
                        "validation_tier": event.validation_tier,
                        "model_set": model_label,
                        "probability": float(prob) if np.isfinite(prob) else np.nan,
                        "percentile_rank_in_year_model": float(rank) if np.isfinite(rank) else np.nan,
                        "above_year_model_p80": bool(np.isfinite(prob) and prob >= p80),
                        "above_year_model_p90": bool(np.isfinite(prob) and prob >= p90),
                        "longitude": event.longitude,
                        "latitude": event.latitude,
                    }
                )

            for tier, tier_df in year_events.groupby("validation_tier"):
                tier_ids = tier_df["dated_record_id"].tolist()
                tier_probs = np.array(
                    [
                        row["probability"]
                        for row in event_rows
                        if row["event_year"] == year
                        and row["model_set"] == model_label
                        and row["dated_record_id"] in tier_ids
                    ],
                    dtype=float,
                )
                tier_ranks = np.array(
                    [
                        row["percentile_rank_in_year_model"]
                        for row in event_rows
                        if row["event_year"] == year
                        and row["model_set"] == model_label
                        and row["dated_record_id"] in tier_ids
                    ],
                    dtype=float,
                )
                finite = np.isfinite(tier_probs)
                summary_rows.append(
                    {
                        "validation_tier": tier,
                        "year": year,
                        "model_set": model_label,
                        "n_events": int(finite.sum()),
                        "mean_event_probability": float(np.nanmean(tier_probs)) if finite.any() else np.nan,
                        "median_event_probability": float(np.nanmedian(tier_probs)) if finite.any() else np.nan,
                        "mean_event_percentile_rank": float(np.nanmean(tier_ranks)) if finite.any() else np.nan,
                        "median_event_percentile_rank": float(np.nanmedian(tier_ranks)) if finite.any() else np.nan,
                        "hit_rate_above_p80": float(np.nanmean(tier_probs >= p80)) if finite.any() else np.nan,
                        "hit_rate_above_p90": float(np.nanmean(tier_probs >= p90)) if finite.any() else np.nan,
                        "background_separation_auc": auc_against_background(tier_probs, bg_auc),
                    }
                )

    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows), pd.DataFrame(stats_rows)


def make_validation_candidates(dedup: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    candidates = dedup.loc[
        dedup["dedup_keep"]
        & dedup["event_year"].between(2017, 2024)
        & dedup["event_date"].notna()
    ].copy()
    candidates["validation_tier"] = "dated_deduplicated"
    strict = (
        ~candidates["overlaps_2018_training_positive_1000m"]
        & candidates["location_quality"].isin(["high", "medium", "unknown"])
    )
    candidates.loc[strict, "validation_tier"] = "strict_independent"
    candidates["use_for_primary_temporal_validation"] = candidates["validation_tier"].eq(
        "strict_independent"
    )
    return candidates


def write_vector(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    if path.exists():
        path.unlink()
    out = gdf.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out.to_file(path, layer=layer, driver="GPKG")


def write_tables(
    all_clipped: gpd.GeoDataFrame,
    dedup: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    event_probs: pd.DataFrame,
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    dedup_log: pd.DataFrame,
    qa: pd.DataFrame,
    out: OutputPaths,
) -> None:
    all_clipped.drop(columns="geometry").to_csv(out.standardized_csv, index=False)
    dedup.drop(columns="geometry").to_csv(out.deduplicated_csv, index=False)
    candidates.drop(columns="geometry").to_csv(out.validation_csv, index=False)
    event_probs.to_csv(out.event_probability_csv, index=False)
    summary.to_csv(out.summary_csv, index=False)
    stats.to_csv(out.raster_stats_csv, index=False)
    dedup_log.to_csv(out.dedup_log_csv, index=False)
    qa.to_csv(out.source_qa_csv, index=False)
    write_vector(all_clipped, out.standardized_gpkg, "all_sources_clipped")
    write_vector(dedup.loc[dedup["dedup_keep"]].copy(), out.deduplicated_gpkg, "deduplicated")
    write_vector(candidates, out.validation_gpkg, "validation_candidates_2017_2024")


def make_figures(candidates: gpd.GeoDataFrame, event_probs: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    if not candidates.empty:
        counts = (
            candidates.groupby(["event_year", "source_dataset"]).size().reset_index(name="count")
        )
        pivot = counts.pivot(index="event_year", columns="source_dataset", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=220)
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_xlabel("Event year")
        ax.set_ylabel("Dated validation candidates")
        ax.set_title("Dated CPEC landslide inventory candidates by source")
        ax.tick_params(axis="x", rotation=0)
        for i, total in enumerate(pivot.sum(axis=1)):
            ax.text(i, total + 0.5, str(int(total)), ha="center", va="bottom", fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "dated_inventory_candidate_counts_by_year_source.png", dpi=300)
        fig.savefig(FIG_DIR / "dated_inventory_candidate_counts_by_year_source.pdf")
        plt.close(fig)

    primary = summary.loc[
        (summary["model_set"] == PRIMARY_MODEL)
        & (summary["validation_tier"] == "strict_independent")
        & (summary["n_events"] > 0)
    ].copy()
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=220)
        ax.bar(primary["year"].astype(str), primary["median_event_percentile_rank"], color="#2f6f9f")
        ax.axhline(0.8, color="#c44e52", lw=1.4, ls="--", label="80th percentile")
        ax.axhline(0.9, color="#8b1a1a", lw=1.4, ls=":", label="90th percentile")
        for i, row in enumerate(primary.itertuples()):
            ax.text(
                i,
                max(row.median_event_percentile_rank - 0.06, 0.06),
                f"n={int(row.n_events)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Event year")
        ax.set_ylabel("Median percentile rank")
        ax.set_title("Independent dated events on fused annual susceptibility maps")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "strict_dated_events_fused_percentile_rank_by_year.png", dpi=300)
        fig.savefig(FIG_DIR / "strict_dated_events_fused_percentile_rank_by_year.pdf")
        plt.close(fig)

    strict_probs = event_probs.loc[event_probs["validation_tier"] == "strict_independent"].copy()
    strict_probs = strict_probs.loc[strict_probs["model_set"].isin(MODEL_SETS.keys())]
    if not strict_probs.empty:
        fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=220)
        order = list(MODEL_SETS.keys())
        tick_labels = ["Conventional", "AlphaEarth\nEmbeddings", "Conventional +\nAlphaEarth Embeddings"]
        data = [
            strict_probs.loc[strict_probs["model_set"] == label, "percentile_rank_in_year_model"].dropna()
            for label in order
        ]
        ax.boxplot(data, tick_labels=tick_labels, showmeans=True, patch_artist=True)
        ax.axhline(0.8, color="#c44e52", lw=1.4, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Percentile rank of dated events")
        ax.set_title("Dated-event validation by stacked feature set")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "strict_dated_events_percentile_rank_by_feature_set.png", dpi=300)
        fig.savefig(FIG_DIR / "strict_dated_events_percentile_rank_by_feature_set.pdf")
        plt.close(fig)


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        cols = list(df.columns) if len(df.columns) else ["No records"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        return header + "\n" + sep
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, sep, *rows])


def report_text(
    qa: pd.DataFrame,
    dedup: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    out: OutputPaths,
) -> str:
    total_clipped = int(qa["clipped_to_cpec"].sum())
    total_2017_2024 = int(qa["dated_2017_2024_clipped"].sum())
    dedup_kept = int(dedup["dedup_keep"].sum())
    candidate_n = int(len(candidates))
    strict_n = int(candidates["use_for_primary_temporal_validation"].sum()) if not candidates.empty else 0
    overlap_1000 = int(candidates["overlaps_2018_training_positive_1000m"].sum()) if not candidates.empty else 0

    primary = summary.loc[
        (summary["model_set"] == PRIMARY_MODEL)
        & (summary["validation_tier"] == "strict_independent")
        & (summary["n_events"] > 0)
    ].copy()
    if not primary.empty:
        primary_bits = []
        for row in primary.sort_values("year").itertuples():
            primary_bits.append(
                f"{int(row.year)}: n={int(row.n_events)}, median percentile={fmt(row.median_event_percentile_rank)}, "
                f"P80 hit rate={pct(row.hit_rate_above_p80)}, background AUC={fmt(row.background_separation_auc)}"
            )
        primary_summary = "\n".join(f"- {item}" for item in primary_bits)
    else:
        primary_summary = "- No strict independent dated events were available for annual probability extraction."

    source_table = markdown_table(qa)
    candidate_year_counts = (
        candidates.groupby(["event_year", "validation_tier"]).size().reset_index(name="n")
        if not candidates.empty
        else pd.DataFrame(columns=["event_year", "validation_tier", "n"])
    )
    candidate_table = markdown_table(candidate_year_counts)

    primary_stats = stats.loc[stats["model_set"] == PRIMARY_MODEL].copy()
    thresholds = markdown_table(
        primary_stats[
            ["year", "raster_mean_probability", "raster_p80_probability", "raster_p90_probability"]
        ]
    )

    return f"""# Dated Inventory Temporal Validation QA (2026-06-05)

## Purpose

This branch evaluates whether independently dated landslide records fall in high-probability zones of the annual 2017-2024 susceptibility maps. It does not retrain the model. The main diagnostic uses the Spatial-CV Stacked Ensemble map for Conventional + AlphaEarth Embeddings, with Conventional and AlphaEarth-only maps retained as comparative checks.

## Source QA

{source_table}

After clipping all dated sources to the official CPEC boundary, {total_clipped} records were available across all years and {total_2017_2024} records fell in the 2017-2024 dynamic-mapping window. Deduplication retained {dedup_kept} CPEC dated records across all years. For temporal validation, {candidate_n} dated 2017-2024 records were retained after source deduplication; {strict_n} met the strict independent tier after excluding events within {int(TRAINING_OVERLAP_STRICT_M)} m of the 2018 training positives and excluding low-location-quality records. {overlap_1000} dated records overlapped the 2018 training positives within {int(TRAINING_OVERLAP_STRICT_M)} m and are therefore reported separately rather than used as the primary independent validation tier.

## Candidate Counts

{candidate_table}

## Annual Fused Raster Thresholds

{thresholds}

## Primary Temporal Validation Result

{primary_summary}

## Interpretation

The dated validation branch is useful but deliberately conservative. The available independent dated records are concentrated in 2017-2018, because the HMA/GLC/COOLR sources currently available in the project contain few or no clipped CPEC events for 2019-2024. Therefore, these outputs support a dated-event validation check for the early dynamic period, while the 2019-2024 maps should still be described as dynamic-covariate transfer maps unless additional clean dated events are added.

For the paper, the defensible wording is: "Independent dated-event checks show that available 2017-2018 CPEC landslides preferentially fall in high-percentile zones of the annual susceptibility maps; however, the dated inventory is too sparse after 2018 to claim full event-by-year validation for 2019-2024." This avoids overstating the temporal experiment and directly addresses reviewer concerns about dated inventory quality.

## Managed Outputs

- Standardized clipped source inventory: `{out.standardized_csv}`
- Deduplicated dated CPEC inventory: `{out.deduplicated_csv}`
- 2017-2024 validation candidates: `{out.validation_csv}`
- Event probability extraction table: `{out.event_probability_csv}`
- Annual summary diagnostics: `{out.summary_csv}`
- Raster threshold table: `{out.raster_stats_csv}`
- Deduplication log: `{out.dedup_log_csv}`
- Figures: `{FIG_DIR}`
"""


def main() -> None:
    ensure_dirs()
    out = outputs()
    boundary = read_boundary()

    loaders = [
        ("HMA point", load_hma_point),
        ("HMA polygon representative point", load_hma_polygon_representative),
        ("NASA GLC legacy", load_glc),
        ("NASA COOLR reports", load_coolr_reports),
        ("NASA COOLR events", load_coolr_events),
    ]
    unclipped = {}
    clipped = {}
    for name, loader in loaders:
        print(f"Loading {name}...")
        gdf = loader()
        unclipped[name] = gdf
        clipped[name] = clip_to_boundary(gdf, boundary)
        print(f"  raw={len(gdf):,}, clipped={len(clipped[name]):,}")

    qa = source_qa(unclipped, clipped)
    all_clipped = concat_sources(clipped)
    print(f"All clipped dated records: {len(all_clipped):,}")

    dedup, dedup_log = spatial_date_deduplicate(all_clipped)
    dedup = mark_training_overlap(dedup)
    candidates = make_validation_candidates(dedup)
    print(
        "Validation candidates 2017-2024: "
        f"{len(candidates):,} total, {int(candidates['use_for_primary_temporal_validation'].sum()):,} strict"
    )

    event_probs, summary, stats = probability_diagnostics(candidates)
    write_tables(all_clipped, dedup, candidates, event_probs, summary, stats, dedup_log, qa, out)
    make_figures(candidates, event_probs, summary)
    out.report_md.write_text(report_text(qa, dedup, candidates, summary, stats, out), encoding="utf-8")

    print("Saved dated temporal validation outputs:")
    print(f"  {OUT_ROOT}")
    print(f"  {out.report_md}")
    print(f"  {FIG_DIR}")


if __name__ == "__main__":
    main()
