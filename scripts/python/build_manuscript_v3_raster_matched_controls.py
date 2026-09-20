"""Build repeated raster-derived, accessibility/terrain-matched controls.

Candidate pixels are sampled from the complete 250 m CPEC predictor grid,
excluded within 500 m of inventory positives, and matched without replacement
on elevation, corrected slope, road/river/fault proximity and monsoon rainfall.
All model predictors, including AlphaEarth bands, are sampled from the same
rasters used for mapping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"D:\DING PROJECT")
INPUT_TABLE = (
    PROJECT_ROOT
    / "01_clean_data"
    / "manuscript_v3_corrected_terrain"
    / "cpec_baseline_samples_corrected_terrain_lithology.csv"
)
BASE = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_gee_inputs_250m"
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
LOCAL = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
TERRAIN = (
    PROJECT_ROOT
    / "01_clean_data"
    / "manuscript_v3_corrected_terrain"
    / "cpec_copernicus_glo30_corrected_slope_aspect_250m.tif"
)
LITHOLOGY = (
    PROJECT_ROOT
    / "01_clean_data"
    / "manuscript_v3_corrected_terrain"
    / "cpec_corrected_lithology_classes_250m.tif"
)
AE_DIR = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_gee_inputs_250m"
    / "alphaearth_embeddings"
)
OUT_DIR = PROJECT_ROOT / "01_clean_data" / "manuscript_v3_raster_matched_controls"

SEED = 141300
REPEATS = 3
WINDOW_SIZE = 1024
POSITIVE_EXCLUSION_KM = 0.5
EARTH_RADIUS_KM = 6371.0088
MATCH_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "log1p_dist_road_m",
    "log1p_dist_river_m",
    "log1p_dist_fault_m",
    "rain_monsoon_total",
]
MATCH_WEIGHTS = np.asarray([1.0, 1.5, 1.0, 2.0, 1.2, 1.0], dtype=float)
ALPHA_COLUMNS = [f"A{i:02d}" for i in range(64)]


def verify_alignment(*datasets: rasterio.io.DatasetReader) -> None:
    ref = datasets[0]
    for dataset in datasets[1:]:
        if (
            dataset.width != ref.width
            or dataset.height != ref.height
            or dataset.crs != ref.crs
            or dataset.transform != ref.transform
        ):
            raise ValueError(f"Raster alignment mismatch: {dataset.name}")


def candidate_probability(log_road: np.ndarray) -> np.ndarray:
    probability = np.full(log_road.shape, 0.0006, dtype="float32")
    probability[log_road <= 10.0] = 0.003
    probability[log_road <= 8.0] = 0.020
    probability[log_road <= 6.0] = 0.120
    return probability


def scan_candidate_pixels(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    with (
        rasterio.open(BASE) as base,
        rasterio.open(LOCAL) as local,
        rasterio.open(TERRAIN) as terrain,
        rasterio.open(LITHOLOGY) as lithology,
    ):
        verify_alignment(base, local, terrain, lithology)
        for row_off in range(0, base.height, WINDOW_SIZE):
            for col_off in range(0, base.width, WINDOW_SIZE):
                height = min(WINDOW_SIZE, base.height - row_off)
                width = min(WINDOW_SIZE, base.width - col_off)
                window = Window(col_off, row_off, width, height)
                elevation = base.read(1, window=window, out_dtype="float32")
                road = local.read(1, window=window, out_dtype="float32")
                valid = np.isfinite(elevation) & np.isfinite(road)
                choose = valid & (rng.random(valid.shape) < candidate_probability(road))
                if not choose.any():
                    continue

                rr, cc = np.nonzero(choose)
                base_data = base.read([1, 4, 5, 6, 7, 8], window=window, out_dtype="float32")
                local_data = local.read([1, 2, 3, 5, 6, 7, 8, 9, 10, 11], window=window, out_dtype="float32")
                terrain_data = terrain.read([1, 2], window=window, out_dtype="float32")
                lith_data = lithology.read(1, window=window)
                global_rows = rr + row_off
                global_cols = cc + col_off
                x, y = rasterio.transform.xy(base.transform, global_rows, global_cols, offset="center")
                frame = pd.DataFrame(
                    {
                        "raster_row": global_rows,
                        "raster_col": global_cols,
                        "longitude": np.asarray(x),
                        "latitude": np.asarray(y),
                        "elevation_m": base_data[0, rr, cc],
                        "rain_monsoon_total": base_data[1, rr, cc],
                        "rain_max_1day": base_data[2, rr, cc],
                        "ndvi_median": base_data[3, rr, cc],
                        "ndvi_amplitude": base_data[4, rr, cc],
                        "modis_lc_type1": base_data[5, rr, cc],
                        "log1p_dist_road_m": local_data[0, rr, cc],
                        "log1p_dist_river_m": local_data[1, rr, cc],
                        "log1p_dist_fault_m": local_data[2, rr, cc],
                        "profile_curvature": local_data[3, rr, cc],
                        "plan_curvature": local_data[4, rr, cc],
                        "tri": local_data[5, rr, cc],
                        "twi": local_data[6, rr, cc],
                        "valley_depth": local_data[7, rr, cc],
                        "soil_type": local_data[8, rr, cc],
                        "eq_density_ms5": local_data[9, rr, cc],
                        "slope_deg": terrain_data[0, rr, cc],
                        "aspect_deg": terrain_data[1, rr, cc],
                        "lithology_code": lith_data[rr, cc],
                    }
                )
                rows.append(frame)
    candidates = pd.concat(rows, ignore_index=True)
    required = MATCH_COLUMNS + [
        "rain_max_1day",
        "ndvi_median",
        "ndvi_amplitude",
        "profile_curvature",
        "plan_curvature",
        "tri",
        "twi",
        "valley_depth",
    ]
    candidates = candidates[np.isfinite(candidates[required]).all(axis=1)].reset_index(drop=True)
    return candidates


def exclude_positive_buffer(candidates: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    positive_coords = np.deg2rad(positives[["latitude", "longitude"]].to_numpy(float))
    candidate_coords = np.deg2rad(candidates[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(positive_coords, metric="haversine")
    distance = tree.query(candidate_coords, k=1, return_distance=True)[0].ravel() * EARTH_RADIUS_KM
    out = candidates[distance >= POSITIVE_EXCLUSION_KM].copy()
    out["nearest_positive_distance_km"] = distance[distance >= POSITIVE_EXCLUSION_KM]
    return out.reset_index(drop=True)


def nearest_candidate_pool(
    candidates: pd.DataFrame, positives: pd.DataFrame, neighbours: int = 50
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, SimpleImputer]:
    combined = pd.concat([positives[MATCH_COLUMNS], candidates[MATCH_COLUMNS]], ignore_index=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(combined))
    pos_x = transformed[: len(positives)]
    candidate_x = transformed[len(positives) :]
    model = NearestNeighbors(n_neighbors=min(neighbours, len(candidates)), algorithm="auto", n_jobs=-1)
    model.fit(candidate_x)
    _, indices = model.kneighbors(pos_x)
    pool_indices = np.unique(indices.ravel())
    return candidates.iloc[pool_indices].reset_index(drop=True), pos_x, scaler, imputer


def parse_tile_offsets(path: Path) -> tuple[int, int]:
    groups = re.findall(r"-(\d{10})", path.stem)
    if len(groups) < 2:
        raise ValueError(f"Cannot parse tile offsets from {path.name}")
    return int(groups[-2]), int(groups[-1])


def sample_alphaearth(pool: pd.DataFrame) -> pd.DataFrame:
    result = np.full((len(pool), 64), np.nan, dtype="float32")
    assigned = np.zeros(len(pool), dtype=bool)
    for path in sorted(AE_DIR.glob("*.tif")):
        row_offset, col_offset = parse_tile_offsets(path)
        with rasterio.open(path) as src:
            use = (
                (pool.raster_row >= row_offset)
                & (pool.raster_row < row_offset + src.height)
                & (pool.raster_col >= col_offset)
                & (pool.raster_col < col_offset + src.width)
            )
            indices = np.flatnonzero(use.to_numpy())
            if not len(indices):
                continue
            local_rows = pool.loc[indices, "raster_row"].to_numpy(int) - row_offset
            local_cols = pool.loc[indices, "raster_col"].to_numpy(int) - col_offset
            block_groups: dict[tuple[int, int], list[int]] = {}
            for position, (row, col) in enumerate(zip(local_rows, local_cols)):
                block_groups.setdefault((row // 256, col // 256), []).append(position)
            for (block_row, block_col), positions in block_groups.items():
                row0, col0 = block_row * 256, block_col * 256
                height = min(256, src.height - row0)
                width = min(256, src.width - col0)
                data = src.read(window=Window(col0, row0, width, height), out_dtype="float32")
                for position in positions:
                    global_index = indices[position]
                    rr = local_rows[position] - row0
                    cc = local_cols[position] - col0
                    result[global_index, :] = data[:, rr, cc]
                    assigned[global_index] = True
    for column, values in zip(ALPHA_COLUMNS, result.T):
        pool[column] = values
    valid = assigned & np.isfinite(result).all(axis=1)
    return pool[valid].reset_index(drop=True)


def matching_space(
    positives: pd.DataFrame, pool: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([positives[MATCH_COLUMNS], pool[MATCH_COLUMNS]], ignore_index=True)
    values = SimpleImputer(strategy="median").fit_transform(combined)
    values = StandardScaler().fit_transform(values) * MATCH_WEIGHTS
    return values[: len(positives)], values[len(positives) :]


def greedy_match(
    positives: pd.DataFrame, pool: pd.DataFrame, seed: int, neighbours: int = 100
) -> pd.DataFrame:
    pos_x, pool_x = matching_space(positives, pool)
    model = NearestNeighbors(n_neighbors=min(neighbours, len(pool)), n_jobs=-1).fit(pool_x)
    distances, indices = model.kneighbors(pos_x)
    rng = np.random.default_rng(seed)
    # Match the hardest positive samples first to preserve common support.
    order = np.argsort(distances[:, 0])[::-1]
    used: set[int] = set()
    selected = np.full(len(positives), -1, dtype=int)
    selected_distance = np.full(len(positives), np.nan, dtype=float)
    for positive_index in order:
        available = [rank for rank, candidate in enumerate(indices[positive_index]) if int(candidate) not in used]
        if not available:
            raise RuntimeError("Insufficient unique candidate controls within the matching neighbourhood.")
        shortlist = available[: min(3, len(available))]
        choice_rank = int(rng.choice(shortlist))
        candidate = int(indices[positive_index, choice_rank])
        used.add(candidate)
        selected[positive_index] = candidate
        selected_distance[positive_index] = distances[positive_index, choice_rank]
    matched = pool.iloc[selected].copy().reset_index(drop=True)
    matched["match_distance_standardised"] = selected_distance
    return matched


def standardised_differences(positives: pd.DataFrame, controls: pd.DataFrame, repeat: int) -> pd.DataFrame:
    rows = []
    for factor in MATCH_COLUMNS:
        pos = pd.to_numeric(positives[factor], errors="coerce")
        neg = pd.to_numeric(controls[factor], errors="coerce")
        pooled_sd = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
        smd = (pos.mean() - neg.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append(
            {
                "repeat": repeat,
                "factor": factor,
                "positive_mean": pos.mean(),
                "control_mean": neg.mean(),
                "standardised_mean_difference": smd,
                "absolute_standardised_mean_difference": abs(smd),
            }
        )
    return pd.DataFrame(rows)


def assemble_table(positives: pd.DataFrame, controls: pd.DataFrame, repeat: int) -> pd.DataFrame:
    controls = controls.copy()
    controls["label"] = 0
    controls["inventory_id"] = [f"RMC_{repeat:02d}_{i:05d}" for i in range(len(controls))]
    controls["hazard_type"] = "non_landslide"
    controls["source"] = "raster_terrain_accessibility_matched_control"
    controls["use_role"] = "matched_control_sensitivity"
    controls["confidence"] = "not_applicable"
    controls["event_year"] = np.nan
    controls["positive_buffer_m"] = POSITIVE_EXCLUSION_KM * 1000.0
    controls["negative_ratio"] = 1.0
    lon_block = np.floor(controls.longitude).astype(int)
    lat_block = np.floor(controls.latitude).astype(int)
    controls["spatial_block_1deg"] = lon_block.astype(str) + "_" + lat_block.astype(str)
    controls["spatial_fold_5"] = (np.abs(lon_block * 31 + lat_block * 17) % 5) + 1
    controls["cpec_admin_transfer_domain"] = "raster-matched control"
    controls["admin_unit"] = "raster-matched control"
    controls["dist_road_m"] = np.expm1(controls["log1p_dist_road_m"])
    controls["dist_river_m"] = np.expm1(controls["log1p_dist_river_m"])
    controls["dist_fault_m"] = np.expm1(controls["log1p_dist_fault_m"])
    all_columns = sorted(set(positives.columns) | set(controls.columns))
    combined = pd.concat(
        [positives.reindex(columns=all_columns), controls.reindex(columns=all_columns)],
        ignore_index=True,
    )
    return combined.sample(frac=1.0, random_state=SEED + repeat).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT_TABLE, encoding="utf-8-sig")
    positives = source[source.label == 1].copy().reset_index(drop=True)
    print("Scanning candidate pixels...", flush=True)
    candidates = scan_candidate_pixels()
    print(f"Candidate pixels before buffer: {len(candidates):,}", flush=True)
    candidates = exclude_positive_buffer(candidates, positives)
    print(f"Candidate pixels after 500 m buffer: {len(candidates):,}", flush=True)
    pool, _, _, _ = nearest_candidate_pool(candidates, positives, neighbours=50)
    print(f"Nearest-neighbour candidate pool: {len(pool):,}", flush=True)
    pool = sample_alphaearth(pool)
    print(f"Candidate pool with complete AlphaEarth data: {len(pool):,}", flush=True)
    if len(pool) < len(positives) * 1.2:
        raise RuntimeError("Too few complete candidate controls for stable matching.")
    pool_path = OUT_DIR / "complete_raster_candidate_pool.parquet"
    pool.to_parquet(pool_path, index=False)

    balance_tables = []
    output_tables = []
    for repeat in range(1, REPEATS + 1):
        controls = greedy_match(positives, pool, SEED + repeat - 1)
        balance = standardised_differences(positives, controls, repeat)
        table = assemble_table(positives, controls, repeat)
        path = OUT_DIR / f"cpec_baseline_raster_matched_controls_repeat_{repeat:02d}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        balance_tables.append(balance)
        output_tables.append(str(path))
        print(
            f"Repeat {repeat}: max |SMD|={balance.absolute_standardised_mean_difference.max():.3f}",
            flush=True,
        )

    balance = pd.concat(balance_tables, ignore_index=True)
    balance.to_csv(OUT_DIR / "raster_matched_control_balance.csv", index=False)
    manifest = {
        "input_positive_table": str(INPUT_TABLE),
        "candidate_base_raster": str(BASE),
        "candidate_local_raster": str(LOCAL),
        "corrected_terrain_raster": str(TERRAIN),
        "corrected_lithology_raster": str(LITHOLOGY),
        "alphaearth_tile_directory": str(AE_DIR),
        "matching_columns": MATCH_COLUMNS,
        "matching_weights": MATCH_WEIGHTS.tolist(),
        "positive_exclusion_buffer_m": int(POSITIVE_EXCLUSION_KM * 1000),
        "matching": "greedy nearest-neighbour without replacement; hardest positives first; random choice among three nearest available controls",
        "candidate_count_after_buffer": len(candidates),
        "complete_pool_count": len(pool),
        "complete_pool_file": str(pool_path),
        "repeats": REPEATS,
        "outputs": output_tables,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(balance.to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
