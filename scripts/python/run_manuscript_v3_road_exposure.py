"""Road exposure and road-distance-ablation stability for manuscript v3."""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import shapefile
from pyproj import Geod
from scipy.stats import spearmanr
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_manuscript_v3_stacks_to_baseline_rasters as mapping_runner  # noqa: E402
from run_manuscript_v3_harmonised_aoa import (  # noqa: E402
    LATENT_DIMENSIONS,
    THRESHOLD_QUANTILE,
    different_block_distance,
)
from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    PROJECT_ROOT,
    SEED,
    load_data,
    make_preprocessor,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

FUSED_RASTER = (
    PROJECT_ROOT
    / "04_maps"
    / "manuscript_v3_baseline_susceptibility_scores_250m"
    / "cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif"
)
NO_ROAD_BUNDLE = (
    PROJECT_ROOT
    / "03_models"
    / "manuscript_v3_road_distance_ablation_final"
    / "nested_spatial_stack_no_road_distance.joblib"
)
ROAD_SHP = (
    PROJECT_ROOT
    / "FINAL PAPER"
    / "00_Study_Area"
    / "Figure_1_ArcMap_layers"
    / "03_CPEC_road_network.shp"
)
KKH_SHP = ROAD_SHP.with_name("04_KKH_route.shp")
DOMAIN_SHP = ROAD_SHP.with_name("02_CPEC_transfer_subdomains.shp")
OUT_DIR = PROJECT_ROOT / "04_maps" / "manuscript_v3_road_exposure"
MAX_SEGMENT_M = 250.0
NODATA = -9999.0
GEOD = Geod(ellps="WGS84")


def iter_parts(record_shape):
    points = record_shape.points
    starts = list(record_shape.parts) + [len(points)]
    for start, end in zip(starts[:-1], starts[1:]):
        if end - start >= 2:
            yield points[start:end]


def midpoint_segment(lon1: float, lat1: float, lon2: float, lat2: float):
    az12, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    if not np.isfinite(distance) or distance <= 0:
        return None
    lonm, latm, _ = GEOD.fwd(lon1, lat1, az12, distance / 2.0)
    return lonm, latm, distance


def densify_roads() -> pd.DataFrame:
    kkh_reader = shapefile.Reader(str(KKH_SHP), encoding="latin1")
    kkh_ids = {int(rec.objectid) for rec in kkh_reader.records()}
    reader = shapefile.Reader(str(ROAD_SHP), encoding="latin1")
    rows: list[dict[str, object]] = []
    segment_id = 0
    for feature_id, sr in enumerate(reader.iterShapeRecords(), start=1):
        attributes = sr.record.as_dict()
        objectid = int(attributes.get("objectid") or feature_id)
        shield = str(attributes.get("shield") or "").strip()
        name = str(attributes.get("name1") or attributes.get("name2") or "").strip()
        route_label = shield or name or "Unnamed road"
        for part in iter_parts(sr.shape):
            for (lon1, lat1), (lon2, lat2) in zip(part[:-1], part[1:]):
                az12, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
                if not np.isfinite(distance) or distance <= 0:
                    continue
                pieces = max(1, int(math.ceil(distance / MAX_SEGMENT_M)))
                points = [(lon1, lat1)]
                if pieces > 1:
                    points.extend(GEOD.npts(lon1, lat1, lon2, lat2, pieces - 1))
                points.append((lon2, lat2))
                for start, end in zip(points[:-1], points[1:]):
                    middle = midpoint_segment(*start, *end)
                    if middle is None:
                        continue
                    lonm, latm, length_m = middle
                    segment_id += 1
                    rows.append(
                        {
                            "segment_id": segment_id,
                            "feature_id": feature_id,
                            "objectid": objectid,
                            "route_label": route_label,
                            "road_type": str(attributes.get("type") or "").strip(),
                            "rank": attributes.get("rank"),
                            "is_kkh_proxy": objectid in kkh_ids,
                            "start_lon": start[0],
                            "start_lat": start[1],
                            "end_lon": end[0],
                            "end_lat": end[1],
                            "longitude": lonm,
                            "latitude": latm,
                            "length_km": length_m / 1000.0,
                        }
                    )
    return pd.DataFrame(rows)


def sample_raster(dataset, coordinates, indexes=None, chunk_size: int = 10000) -> np.ndarray:
    arrays = []
    for start in range(0, len(coordinates), chunk_size):
        chunk = coordinates[start : start + chunk_size]
        arrays.append(np.asarray(list(dataset.sample(chunk, indexes=indexes)), dtype="float32"))
    return np.vstack(arrays)


def sample_alpha(base, coordinates) -> np.ndarray:
    xs = np.asarray([c[0] for c in coordinates])
    ys = np.asarray([c[1] for c in coordinates])
    rows, cols = rasterio.transform.rowcol(base.transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.full((len(coordinates), 64), np.nan, dtype="float32")
    assigned = np.zeros(len(coordinates), dtype=bool)
    for tile_path in sorted(mapping_runner.AE_DIR.glob("*.tif")):
        row_offset, col_offset = mapping_runner.tile_offsets(tile_path)
        with rasterio.open(tile_path) as tile:
            selected = (
                (~assigned)
                & (rows >= row_offset)
                & (rows < row_offset + tile.height)
                & (cols >= col_offset)
                & (cols < col_offset + tile.width)
            )
            indices = np.flatnonzero(selected)
            if not len(indices):
                continue
            coords = [coordinates[i] for i in indices]
            values[indices] = sample_raster(tile, coords, indexes=list(range(1, 65)))
            assigned[indices] = True
    if not assigned.all():
        raise RuntimeError(f"AlphaEarth tiles did not cover {int((~assigned).sum())} road segments.")
    values[values == NODATA] = np.nan
    return values


def predictor_frame(coordinates) -> pd.DataFrame:
    with (
        rasterio.open(mapping_runner.BASE) as base,
        rasterio.open(mapping_runner.LOCAL) as local,
        rasterio.open(mapping_runner.TERRAIN) as terrain,
        rasterio.open(mapping_runner.LITHOLOGY) as lithology,
    ):
        mapping_runner.verify_alignment(base, local, terrain, lithology)
        b = sample_raster(base, coordinates, indexes=[1, 4, 5, 6, 7, 8])
        l = sample_raster(local, coordinates, indexes=[1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
        t = sample_raster(terrain, coordinates, indexes=[1, 2])
        rock = sample_raster(lithology, coordinates, indexes=[1]).ravel()
        alpha = sample_alpha(base, coordinates)
    aspect = t[:, 1]
    conventional = pd.DataFrame(
        {
            "elevation_m": b[:, 0],
            "slope_deg": t[:, 0],
            "rain_monsoon_total": b[:, 1],
            "rain_max_1day": b[:, 2],
            "ndvi_median": b[:, 3],
            "ndvi_amplitude": b[:, 4],
            "log1p_dist_road_m": l[:, 0],
            "log1p_dist_river_m": l[:, 1],
            "log1p_dist_fault_m": l[:, 2],
            "profile_curvature": l[:, 3],
            "plan_curvature": l[:, 4],
            "tri": l[:, 5],
            "twi": l[:, 6],
            "valley_depth": l[:, 7],
            "aspect_sin": np.sin(np.deg2rad(aspect)),
            "aspect_cos": np.cos(np.deg2rad(aspect)),
            "modis_lc_type1": b[:, 5],
            "lithology_code": rock,
            "soil_type": l[:, 8],
            "eq_density_ms5": l[:, 9],
        }
    )
    return pd.concat(
        [conventional, pd.DataFrame(alpha, columns=mapping_runner.ALPHA_COLUMNS)], axis=1
    )


def raster_thresholds() -> dict[str, float]:
    score_values, disagreement_values = [], []
    with rasterio.open(FUSED_RASTER) as src:
        for _, window in src.block_windows(1):
            score = src.read(1, window=window)
            disagreement = src.read(2, window=window)
            valid = np.isfinite(score) & (score != src.nodata)
            if valid.any():
                score_values.append(score[valid])
                disagreement_values.append(disagreement[valid])
    score = np.concatenate(score_values)
    disagreement = np.concatenate(disagreement_values)
    return {
        "score_p80": float(np.quantile(score, 0.80)),
        "score_p90": float(np.quantile(score, 0.90)),
        "disagreement_p50": float(np.quantile(disagreement, 0.50)),
        "disagreement_p90": float(np.quantile(disagreement, 0.90)),
    }


def full_training_aoa(road_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    training, specs, _ = load_data()
    spec = specs["conventional_alphaearth_embeddings"]
    preprocess = make_preprocessor(spec)
    x_train = preprocess.fit_transform(training[spec.columns])
    dimensions = min(LATENT_DIMENSIONS, x_train.shape[1], len(training) - 1)
    pca = PCA(n_components=dimensions, whiten=True, random_state=SEED)
    z_train = pca.fit_transform(x_train)
    train_blocks = training.spatial_block_1deg.astype(str).to_numpy()
    reference = different_block_distance(z_train, train_blocks)
    threshold = float(np.quantile(reference, THRESHOLD_QUANTILE))
    neighbour = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_train)
    distances = []
    for start in range(0, len(road_frame), 10000):
        transformed = preprocess.transform(road_frame.iloc[start : start + 10000][spec.columns])
        latent = pca.transform(transformed)
        distances.append(neighbour.kneighbors(latent, return_distance=True)[0].ravel())
    distance = np.concatenate(distances)
    di = distance / threshold
    metadata = {
        "latent_dimensions": dimensions,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "reference_distance_threshold_p95": threshold,
    }
    return di, di <= 1.0, metadata


def domain_labels(segments: pd.DataFrame) -> np.ndarray:
    reader = shapefile.Reader(str(DOMAIN_SHP), encoding="latin1")
    labels = np.full(len(segments), "Unassigned", dtype=object)
    xs = segments.longitude.to_numpy(float)
    ys = segments.latitude.to_numpy(float)
    try:
        from shapely import contains_xy
    except ImportError:
        contains_xy = None
    display = {
        "KP-AJK": "KPK-AJK",
        "Kashgar (Xinjiang, China)": "Kashgar",
        "Punjab-Sindh lowland corridor": "Punjab-Sindh",
    }
    for sr in reader.iterShapeRecords():
        name = display.get(str(sr.record.domain), str(sr.record.domain))
        geom = shape(sr.shape.__geo_interface__)
        if contains_xy is not None:
            inside = contains_xy(geom, xs, ys)
        else:
            inside = np.array([geom.contains(shape({"type": "Point", "coordinates": p})) for p in zip(xs, ys)])
        labels[inside] = name
    return labels


def weighted_summary(data: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float]:
    length = data.length_km.to_numpy(float)
    total = length.sum()
    score = data.fused_score.to_numpy(float)
    ablated = data.no_road_score.to_numpy(float)
    high80 = score >= thresholds["score_p80"]
    high90 = score >= thresholds["score_p90"]
    supported90 = high90 & data.inside_aoa.to_numpy(bool) & (
        data.base_model_disagreement.to_numpy(float) <= thresholds["disagreement_p50"]
    )
    verification90 = high90 & ~supported90
    top_full = score >= np.quantile(score, 0.90)
    top_ablation = ablated >= np.quantile(ablated, 0.90)
    return {
        "total_length_km": total,
        "length_weighted_mean_score": float(np.average(score, weights=length)),
        "actual_length_ge_p80_km": float(length[high80].sum()),
        "actual_length_ge_p90_km": float(length[high90].sum()),
        "score_weighted_length_km": float(np.sum(length * score)),
        "supported_high_p90_length_km": float(length[supported90].sum()),
        "verification_priority_p90_length_km": float(length[verification90].sum()),
        "inside_aoa_length_pct": float(100.0 * length[data.inside_aoa].sum() / total),
        "length_weighted_mean_score_no_road_distance": float(np.average(ablated, weights=length)),
        "segment_score_spearman_rho": float(spearmanr(score, ablated).statistic),
        "top_decile_segment_overlap": float((top_full & top_ablation).sum() / max(1, top_full.sum())),
    }


def save_hotspot_geojson(data: pd.DataFrame, threshold: float, path: Path) -> None:
    features = []
    for row in data.loc[data.fused_score >= threshold].itertuples(index=False):
        properties = {
            "segment_id": int(row.segment_id),
            "route": row.route_label,
            "domain": row.subdomain,
            "length_km": round(float(row.length_km), 6),
            "score": round(float(row.fused_score), 6),
            "no_road": round(float(row.no_road_score), 6),
            "aoa": bool(row.inside_aoa),
            "disagree": round(float(row.base_model_disagreement), 6),
            "priority": row.priority_class,
        }
        line = LineString([(row.start_lon, row.start_lat), (row.end_lon, row.end_lat)])
        features.append({"type": "Feature", "properties": properties, "geometry": mapping(line)})
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segments = densify_roads()
    coordinates = list(zip(segments.longitude, segments.latitude))
    with rasterio.open(FUSED_RASTER) as fused:
        sampled = sample_raster(fused, coordinates, indexes=[1, 2])
    valid = np.isfinite(sampled[:, 0]) & (sampled[:, 0] != NODATA)
    segments = segments.loc[valid].reset_index(drop=True)
    sampled = sampled[valid]
    coordinates = [coordinates[i] for i in np.flatnonzero(valid)]
    segments["fused_score"] = sampled[:, 0]
    segments["base_model_disagreement"] = sampled[:, 1]
    segments["subdomain"] = domain_labels(segments)

    frame = predictor_frame(coordinates)
    no_road_bundle = joblib.load(NO_ROAD_BUNDLE)
    no_road_columns = no_road_bundle["feature_spec"]["columns"]
    no_road_score, _ = mapping_runner.predict_bundle(no_road_bundle, frame[no_road_columns])
    segments["no_road_score"] = no_road_score

    di, inside_aoa, aoa_metadata = full_training_aoa(frame)
    segments["dissimilarity_index"] = di
    segments["inside_aoa"] = inside_aoa
    thresholds = raster_thresholds()
    supported = (
        (segments.fused_score >= thresholds["score_p90"])
        & segments.inside_aoa
        & (segments.base_model_disagreement <= thresholds["disagreement_p50"])
    )
    verification = (segments.fused_score >= thresholds["score_p90"]) & ~supported
    segments["priority_class"] = "below_p90"
    segments.loc[supported, "priority_class"] = "supported_high_score"
    segments.loc[verification, "priority_class"] = "verification_priority_high_score"

    summary_rows = []
    for name, data in [
        ("Full CPEC road network", segments),
        ("KKH proxy route (N35/314)", segments[segments.is_kkh_proxy]),
    ]:
        row = {"road_group": name}
        row.update(weighted_summary(data, thresholds))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    named = segments[segments.route_label != "Unnamed road"].copy()
    route_rows = []
    for route, data in named.groupby("route_label"):
        if data.length_km.sum() < 20.0:
            continue
        row = {"route_label": route}
        row.update(weighted_summary(data, thresholds))
        route_rows.append(row)
    routes = pd.DataFrame(route_rows)
    if len(routes):
        routes["rank_by_p90_length"] = routes.actual_length_ge_p90_km.rank(
            method="min", ascending=False
        ).astype(int)
        routes["rank_by_mean_score"] = routes.length_weighted_mean_score.rank(
            method="min", ascending=False
        ).astype(int)
        routes = routes.sort_values(
            ["actual_length_ge_p90_km", "length_weighted_mean_score"], ascending=False
        )

    segments.to_csv(OUT_DIR / "road_segment_scores_250m.csv", index=False)
    summary.to_csv(OUT_DIR / "road_network_exposure_summary.csv", index=False)
    routes.to_csv(OUT_DIR / "named_route_exposure_ranking.csv", index=False)
    save_hotspot_geojson(
        segments, thresholds["score_p90"], OUT_DIR / "p90_road_hotspot_segments.geojson"
    )
    manifest = {
        "analysis": "manuscript_v3_road_exposure_and_ablation_stability",
        "score_interpretation": "case-control susceptibility score; not population occurrence probability",
        "road_densification_max_segment_m": MAX_SEGMENT_M,
        "score_thresholds": thresholds,
        "aoa": {
            **aoa_metadata,
            "method": "15-dimensional whitened PCA and p95 different-block nearest-neighbour threshold",
        },
        "priority_definition": {
            "supported_high_score": "score >= P90, inside AoA, disagreement <= median",
            "verification_priority_high_score": "score >= P90 and outside AoA or disagreement > median",
        },
        "road_ablation_model": str(NO_ROAD_BUNDLE),
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    if len(routes):
        print(routes.head(10).to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
