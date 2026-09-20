"""Create 2018 area-of-applicability and transfer-confidence rasters.

This implements the raster-level reliability step that follows the
administrative transferability tests. It uses a variable-importance weighted
nearest-neighbour dissimilarity index in the model predictor space:

- predictors are standardized using the 2018 V3 sample table;
- predictors are weighted by the fitted Extra Trees model importance;
- high-dimensional embedding/fused spaces are reduced with PCA for stable
  nearest-neighbour distances;
- the area-of-applicability threshold is the 95th percentile of leave-one-out
  nearest-neighbour distance among training samples.

Outputs are clipped/masked to the official CPEC study area.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("GDAL_DATA", r"D:\MINICONDA\Library\share\gdal")

from pyproj import datadir

datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
SAMPLE = PROJECT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
MODEL_DIR = PROJECT / "03_models" / "v3_model_family_2018_spatial_cv"
GEE_DIR = PROJECT / "04_maps" / "stacked_ensemble_gee_inputs_250m"
CONVENTIONAL_BASE = (
    GEE_DIR
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
LOCAL_V3 = (
    PROJECT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
ALPHA_DIR = GEE_DIR / "alphaearth_embeddings"
FUSED_DIR = GEE_DIR / "conventional_base_alphaearth_embeddings"
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)
PAK_ADMIN1 = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\gadm41_PAK_shp\gadm41_PAK_1.shp"
)
KASHGAR = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\New Folder\cpecpart.shp"
)

OUT_DIR = PROJECT / "04_maps" / "area_of_applicability_transfer_confidence_250m"
REPORT_DIR = PROJECT / "05_reports"
FIG_DIR = REPORT_DIR / "figures" / "area_of_applicability_transfer_confidence_250m"
PACKAGE_MAPS = (
    PACKAGE
    / "04_probability_maps_250m"
    / "03_transfer_confidence_area_of_applicability"
)
PACKAGE_FIGS = PACKAGE / "03_figures" / "06_area_of_applicability_transfer_confidence"
PACKAGE_METHODS = PACKAGE / "01_methods_and_decisions"
PACKAGE_TABLES = PACKAGE / "02_model_performance_tables"
PACKAGE_SCRIPTS = PACKAGE / "07_reproducible_scripts"

CONVENTIONAL_BASE_BANDS = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "modis_lc_type1",
]
LOCAL_V3_BANDS = [
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
CONVENTIONAL_BANDS = CONVENTIONAL_BASE_BANDS + LOCAL_V3_BANDS
ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]
FUSED_BANDS = CONVENTIONAL_BANDS + ALPHA_BANDS
FLOAT_NODATA = np.float32(-9999.0)
UINT8_NODATA = np.uint8(255)
DOMAIN_ORDER = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]


@dataclass
class FeatureSetConfig:
    key: str
    label: str
    internal: str
    bands: list[str]


FEATURE_SETS = {
    "conventional": FeatureSetConfig(
        key="conventional",
        label="Conventional",
        internal="conventional_v3",
        bands=CONVENTIONAL_BANDS,
    ),
    "alphaearth_embeddings": FeatureSetConfig(
        key="alphaearth_embeddings",
        label="AlphaEarth Embeddings",
        internal="alphaearth_embeddings",
        bands=ALPHA_BANDS,
    ),
    "conventional_alphaearth_embeddings": FeatureSetConfig(
        key="conventional_alphaearth_embeddings",
        label="Conventional + AlphaEarth Embeddings",
        internal="conventional_v3_alphaearth_embeddings",
        bands=FUSED_BANDS,
    ),
}


@dataclass
class FeatureSpace:
    config: FeatureSetConfig
    medians: np.ndarray
    scaler: StandardScaler
    weights: np.ndarray
    pca: PCA | None
    tree: cKDTree
    threshold: float
    transformed_dims: int
    pca_explained_variance: float | None

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype="float32")
        finite = np.isfinite(arr)
        if not finite.all():
            arr = arr.copy()
            bad_rows, bad_cols = np.where(~finite)
            arr[bad_rows, bad_cols] = self.medians[bad_cols]
        z = self.scaler.transform(arr).astype("float32")
        z *= self.weights.astype("float32")
        if self.pca is not None:
            z = self.pca.transform(z).astype("float32")
        return z

    def query_index(self, arr: np.ndarray) -> np.ndarray:
        z = self.transform(arr)
        distances, _ = self.tree.query(z, k=1, workers=-1)
        return (distances.astype("float32") / np.float32(self.threshold)).astype("float32")


def ensure_dirs() -> None:
    for path in [
        OUT_DIR,
        REPORT_DIR,
        FIG_DIR,
        PACKAGE_MAPS,
        PACKAGE_FIGS,
        PACKAGE_METHODS,
        PACKAGE_TABLES,
        PACKAGE_SCRIPTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def output_profile(ref: rasterio.DatasetReader, dtype: str, nodata: float | int) -> dict:
    profile = ref.profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": dtype,
            "compress": "deflate",
            "predictor": 2 if dtype == "float32" else 1,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
            "nodata": nodata,
        }
    )
    return profile


def tile_offset(ref: rasterio.DatasetReader, tile: rasterio.DatasetReader) -> tuple[int, int]:
    col = int(round((tile.transform.c - ref.transform.c) / ref.transform.a))
    row = int(round((ref.transform.f - tile.transform.f) / abs(ref.transform.e)))
    return row, col


def rasterize_study_area(ref: rasterio.DatasetReader) -> np.ndarray:
    boundary = gpd.read_file(BOUNDARY).to_crs(ref.crs)
    mask = rasterize(
        [(geom, 1) for geom in boundary.geometry if geom is not None and not geom.is_empty],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask.astype(bool)


def build_admin_domain_rasters(ref: rasterio.DatasetReader) -> dict[str, np.ndarray]:
    pak = gpd.read_file(PAK_ADMIN1).to_crs(ref.crs)
    kas = gpd.read_file(KASHGAR).to_crs(ref.crs)
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
    parts = []
    for _, row in pak.iterrows():
        domain = mapping.get(row["NAME_1"])
        if domain:
            parts.append({"domain": domain, "geometry": row.geometry})
    for geom in kas.geometry:
        if geom is not None and not geom.is_empty:
            parts.append({"domain": "Kashgar (Xinjiang, China)", "geometry": geom})
    domains = gpd.GeoDataFrame(parts, geometry="geometry", crs=pak.crs)
    out = {}
    for domain in DOMAIN_ORDER:
        geoms = domains.loc[domains["domain"] == domain, "geometry"]
        out[domain] = rasterize(
            [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty],
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
    return out


def extra_trees_weights(config: FeatureSetConfig) -> np.ndarray:
    model_path = MODEL_DIR / f"model_{config.internal}_extra_trees.joblib"
    model = joblib.load(model_path)
    importance = getattr(model, "feature_importances_", None)
    if importance is None and hasattr(model, "named_steps") and "model" in model.named_steps:
        importance = getattr(model.named_steps["model"], "feature_importances_", None)
    if importance is None or len(importance) != len(config.bands):
        return np.ones(len(config.bands), dtype="float32")
    importance = np.asarray(importance, dtype="float64")
    importance = np.maximum(importance, 1e-9)
    weights = np.sqrt(importance / np.mean(importance))
    weights = np.clip(weights, 0.25, 3.0)
    return weights.astype("float32")


def fit_feature_space(df: pd.DataFrame, config: FeatureSetConfig) -> tuple[FeatureSpace, dict[str, object]]:
    x = df[config.bands].replace([np.inf, -np.inf], np.nan).copy()
    medians = x.median(axis=0).astype("float32").to_numpy()
    x = x.fillna(dict(zip(config.bands, medians))).astype("float32").to_numpy()
    scaler = StandardScaler()
    z = scaler.fit_transform(x).astype("float32")
    weights = extra_trees_weights(config)
    z *= weights

    pca: PCA | None = None
    pca_explained: float | None = None
    transformed = z
    if len(config.bands) > 10:
        pca_full = PCA(n_components=0.95, svd_solver="full", random_state=141300)
        pca_full.fit(z)
        n_components = min(int(pca_full.n_components_), 20)
        pca = PCA(n_components=n_components, svd_solver="full", random_state=141300)
        transformed = pca.fit_transform(z).astype("float32")
        pca_explained = float(np.sum(pca.explained_variance_ratio_))

    threshold_source = "spatial_cv_validation_to_training_95pct"
    if "spatial_fold_5" in df.columns:
        folds = df["spatial_fold_5"].to_numpy()
        spatial_distances = []
        for fold in sorted(pd.Series(folds).dropna().unique()):
            train_idx = folds != fold
            test_idx = folds == fold
            if int(train_idx.sum()) == 0 or int(test_idx.sum()) == 0:
                continue
            fold_tree = cKDTree(transformed[train_idx], leafsize=64)
            fold_dist, _ = fold_tree.query(transformed[test_idx], k=1, workers=-1)
            spatial_distances.append(fold_dist)
        if spatial_distances:
            nearest_reference = np.concatenate(spatial_distances)
        else:
            threshold_source = "leave_one_out_nearest_neighbor_95pct"
            loo_tree = cKDTree(transformed, leafsize=64)
            loo_dist, _ = loo_tree.query(transformed, k=2, workers=-1)
            nearest_reference = loo_dist[:, 1]
    else:
        threshold_source = "leave_one_out_nearest_neighbor_95pct"
        loo_tree = cKDTree(transformed, leafsize=64)
        loo_dist, _ = loo_tree.query(transformed, k=2, workers=-1)
        nearest_reference = loo_dist[:, 1]

    threshold = float(np.quantile(nearest_reference, 0.95))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Invalid AOA threshold for {config.label}: {threshold}")

    tree = cKDTree(transformed, leafsize=64)
    feature_space = FeatureSpace(
        config=config,
        medians=medians,
        scaler=scaler,
        weights=weights,
        pca=pca,
        tree=tree,
        threshold=threshold,
        transformed_dims=int(transformed.shape[1]),
        pca_explained_variance=pca_explained,
    )
    info = {
        "feature_set": config.label,
        "n_features_original": len(config.bands),
        "n_features_for_distance": int(transformed.shape[1]),
        "pca_used": pca is not None,
        "pca_explained_variance": pca_explained,
        "aoa_threshold": threshold,
        "aoa_threshold_source": threshold_source,
        "weight_source": "Extra Trees feature importance from the fitted 2018 model family",
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
    }
    return feature_space, info


def flatten_for_model(arr: np.ndarray) -> np.ndarray:
    n_bands, rows, cols = arr.shape
    return arr.reshape(n_bands, rows * cols).T


def write_window_outputs(
    feature_space: FeatureSpace,
    arr: np.ndarray,
    study_mask: np.ndarray,
    win: Window,
    index_dst: rasterio.DatasetWriter,
    mask_dst: rasterio.DatasetWriter,
    confidence_dst: rasterio.DatasetWriter,
) -> tuple[int, int, float, float]:
    rows, cols = study_mask.shape
    flat_mask = study_mask.reshape(rows * cols)
    index_block = np.full(rows * cols, FLOAT_NODATA, dtype="float32")
    confidence_block = np.full(rows * cols, FLOAT_NODATA, dtype="float32")
    aoa_block = np.full(rows * cols, UINT8_NODATA, dtype="uint8")
    if not flat_mask.any():
        index_dst.write(index_block.reshape(rows, cols), 1, window=win)
        mask_dst.write(aoa_block.reshape(rows, cols), 1, window=win)
        confidence_dst.write(confidence_block.reshape(rows, cols), 1, window=win)
        return 0, 0, 0.0, 0.0

    x = flatten_for_model(arr)[flat_mask]
    valid_rows = np.isfinite(x).any(axis=1)
    di = np.full(x.shape[0], np.nan, dtype="float32")
    if valid_rows.any():
        di[valid_rows] = feature_space.query_index(x[valid_rows])
    in_aoa = np.isfinite(di) & (di <= 1.0)
    confidence = np.where(np.isfinite(di), np.clip(1.0 - di, 0.0, 1.0), np.nan).astype("float32")

    idx_positions = np.flatnonzero(flat_mask)
    index_block[idx_positions] = np.where(np.isfinite(di), di, FLOAT_NODATA).astype("float32")
    confidence_block[idx_positions] = np.where(np.isfinite(confidence), confidence, FLOAT_NODATA).astype("float32")
    aoa_block[idx_positions] = np.where(in_aoa, 1, 0).astype("uint8")

    index_dst.write(index_block.reshape(rows, cols), 1, window=win)
    mask_dst.write(aoa_block.reshape(rows, cols), 1, window=win)
    confidence_dst.write(confidence_block.reshape(rows, cols), 1, window=win)
    return int(np.isfinite(di).sum()), int(in_aoa.sum()), float(np.nansum(di)), float(np.nansum(confidence))


def output_paths(config: FeatureSetConfig) -> dict[str, Path]:
    prefix = f"cpec_2018_{config.key}"
    return {
        "index": OUT_DIR / f"{prefix}_aoa_dissimilarity_index_250m.tif",
        "mask": OUT_DIR / f"{prefix}_in_area_of_applicability_250m.tif",
        "confidence": OUT_DIR / f"{prefix}_transfer_confidence_250m.tif",
    }


def process_conventional(
    feature_space: FeatureSpace,
    study_mask: np.ndarray,
    block_size: int,
) -> dict[str, object]:
    paths = output_paths(feature_space.config)
    with rasterio.open(CONVENTIONAL_BASE) as base_src, rasterio.open(LOCAL_V3) as local_src:
        profile_float = output_profile(base_src, "float32", float(FLOAT_NODATA))
        profile_u8 = output_profile(base_src, "uint8", int(UINT8_NODATA))
        with rasterio.open(paths["index"], "w", **profile_float) as index_dst, rasterio.open(
            paths["mask"], "w", **profile_u8
        ) as mask_dst, rasterio.open(paths["confidence"], "w", **profile_float) as confidence_dst:
            totals = [0, 0, 0.0, 0.0]
            processed_windows = 0
            for win in iter_windows(base_src.width, base_src.height, block_size):
                base = base_src.read(window=win, out_dtype="float32")
                local = local_src.read(window=win, out_dtype="float32")
                arr = np.concatenate([base, local], axis=0)
                submask = study_mask[
                    int(win.row_off) : int(win.row_off + win.height),
                    int(win.col_off) : int(win.col_off + win.width),
                ]
                vals = write_window_outputs(
                    feature_space,
                    arr,
                    submask,
                    win,
                    index_dst,
                    mask_dst,
                    confidence_dst,
                )
                totals = [a + b for a, b in zip(totals, vals)]
                processed_windows += 1
                if processed_windows % 25 == 0:
                    print(
                        f"  {feature_space.config.label}: {processed_windows} windows processed",
                        flush=True,
                    )
            for dst, desc in [
                (index_dst, "aoa_dissimilarity_index_conventional"),
                (mask_dst, "in_area_of_applicability_conventional"),
                (confidence_dst, "transfer_confidence_conventional"),
            ]:
                dst.set_band_description(1, desc)
                dst.update_tags(
                    year="2018",
                    feature_set=feature_space.config.label,
                    method="weighted nearest-neighbour area of applicability",
                    aoa_threshold=str(feature_space.threshold),
                )
    return summarize_totals(feature_space.config, paths, totals)


def process_alpha_or_fused(
    feature_space: FeatureSpace,
    study_mask: np.ndarray,
    block_size: int,
) -> dict[str, object]:
    is_fused = feature_space.config.key == "conventional_alphaearth_embeddings"
    tile_dir = FUSED_DIR if is_fused else ALPHA_DIR
    tiles = sorted(tile_dir.glob("*.tif"))
    if len(tiles) != 6:
        raise RuntimeError(f"Expected 6 tiles in {tile_dir}, found {len(tiles)}")

    paths = output_paths(feature_space.config)
    with rasterio.open(CONVENTIONAL_BASE) as ref:
        profile_float = output_profile(ref, "float32", float(FLOAT_NODATA))
        profile_u8 = output_profile(ref, "uint8", int(UINT8_NODATA))
        local_ctx = rasterio.open(LOCAL_V3) if is_fused else None
        try:
            with rasterio.open(paths["index"], "w", **profile_float) as index_dst, rasterio.open(
                paths["mask"], "w", **profile_u8
            ) as mask_dst, rasterio.open(paths["confidence"], "w", **profile_float) as confidence_dst:
                # Tiled GEE exports can miss tiny edge slivers. Initialize the
                # whole study area as outside-AOA/confidence 0, then overwrite
                # every pixel covered by valid predictor tiles below. This
                # prevents nodata holes inside the official boundary.
                for init_win in iter_windows(ref.width, ref.height, block_size):
                    rows, cols = int(init_win.height), int(init_win.width)
                    submask = study_mask[
                        int(init_win.row_off) : int(init_win.row_off + init_win.height),
                        int(init_win.col_off) : int(init_win.col_off + init_win.width),
                    ]
                    index_init = np.full((rows, cols), FLOAT_NODATA, dtype="float32")
                    confidence_init = np.full((rows, cols), FLOAT_NODATA, dtype="float32")
                    aoa_init = np.full((rows, cols), UINT8_NODATA, dtype="uint8")
                    index_init[submask] = np.float32(1.001)
                    confidence_init[submask] = np.float32(0.0)
                    aoa_init[submask] = np.uint8(0)
                    index_dst.write(index_init, 1, window=init_win)
                    confidence_dst.write(confidence_init, 1, window=init_win)
                    mask_dst.write(aoa_init, 1, window=init_win)
                totals = [0, 0, 0.0, 0.0]
                processed_windows = 0
                for tile_path in tiles:
                    print(f"  tile {tile_path.name}", flush=True)
                    with rasterio.open(tile_path) as tile:
                        row0, col0 = tile_offset(ref, tile)
                        for win in iter_windows(tile.width, tile.height, block_size):
                            tile_arr = tile.read(window=win, out_dtype="float32")
                            dst_win = Window(
                                col0 + win.col_off,
                                row0 + win.row_off,
                                win.width,
                                win.height,
                            )
                            submask = study_mask[
                                int(dst_win.row_off) : int(dst_win.row_off + dst_win.height),
                                int(dst_win.col_off) : int(dst_win.col_off + dst_win.width),
                            ]
                            if is_fused:
                                if local_ctx is None:
                                    raise RuntimeError("Local V3 source not open.")
                                local_arr = local_ctx.read(window=dst_win, out_dtype="float32")
                                base_arr = tile_arr[: len(CONVENTIONAL_BASE_BANDS), :, :]
                                alpha_arr = tile_arr[len(CONVENTIONAL_BASE_BANDS) :, :, :]
                                arr = np.concatenate([base_arr, local_arr, alpha_arr], axis=0)
                            else:
                                arr = tile_arr
                            vals = write_window_outputs(
                                feature_space,
                                arr,
                                submask,
                                dst_win,
                                index_dst,
                                mask_dst,
                                confidence_dst,
                            )
                            totals = [a + b for a, b in zip(totals, vals)]
                            processed_windows += 1
                            if processed_windows % 25 == 0:
                                print(
                                    f"  {feature_space.config.label}: {processed_windows} tile windows processed",
                                    flush=True,
                                )
                for dst, desc in [
                    (index_dst, f"aoa_dissimilarity_index_{feature_space.config.key}"),
                    (mask_dst, f"in_area_of_applicability_{feature_space.config.key}"),
                    (confidence_dst, f"transfer_confidence_{feature_space.config.key}"),
                ]:
                    dst.set_band_description(1, desc)
                    dst.update_tags(
                        year="2018",
                        feature_set=feature_space.config.label,
                        method="weighted nearest-neighbour area of applicability",
                        aoa_threshold=str(feature_space.threshold),
                    )
        finally:
            if local_ctx is not None:
                local_ctx.close()
    return summarize_totals(feature_space.config, paths, totals)


def summarize_totals(
    config: FeatureSetConfig,
    paths: dict[str, Path],
    totals: list[float],
) -> dict[str, object]:
    valid_pixels = int(totals[0])
    in_aoa_pixels = int(totals[1])
    mean_di = float(totals[2] / valid_pixels) if valid_pixels else np.nan
    mean_conf = float(totals[3] / valid_pixels) if valid_pixels else np.nan
    return {
        "feature_set": config.label,
        "valid_study_area_pixels": valid_pixels,
        "in_area_of_applicability_pixels": in_aoa_pixels,
        "outside_area_of_applicability_pixels": valid_pixels - in_aoa_pixels,
        "in_area_of_applicability_percent": 100.0 * in_aoa_pixels / valid_pixels if valid_pixels else np.nan,
        "mean_dissimilarity_index": mean_di,
        "mean_transfer_confidence": mean_conf,
        "index_raster": str(paths["index"]),
        "mask_raster": str(paths["mask"]),
        "confidence_raster": str(paths["confidence"]),
    }


def zonal_summary_by_domain(summary_df: pd.DataFrame, domain_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for _, record in summary_df.iterrows():
        with rasterio.open(record["mask_raster"]) as mask_src, rasterio.open(record["confidence_raster"]) as conf_src:
            aoa = mask_src.read(1)
            conf = conf_src.read(1)
            valid = aoa != int(UINT8_NODATA)
            for domain, dmask in domain_masks.items():
                zone = valid & dmask
                n = int(zone.sum())
                if n == 0:
                    continue
                in_count = int((aoa[zone] == 1).sum())
                conf_vals = conf[zone]
                conf_vals = conf_vals[np.isfinite(conf_vals) & (conf_vals != float(FLOAT_NODATA))]
                rows.append(
                    {
                        "feature_set": record["feature_set"],
                        "domain": domain,
                        "study_area_pixels": n,
                        "in_area_of_applicability_pixels": in_count,
                        "in_area_of_applicability_percent": 100.0 * in_count / n,
                        "mean_transfer_confidence": float(np.nanmean(conf_vals)) if len(conf_vals) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def read_downsample(path: Path, max_size: int = 1400) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        scale = max(src.width / max_size, src.height / max_size, 1)
        out_h = int(src.height / scale)
        out_w = int(src.width / scale)
        arr = src.read(1, out_shape=(out_h, out_w), masked=True)
        bounds = src.bounds
    return arr, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def save_all(fig: plt.Figure, name: str) -> None:
    for folder in [FIG_DIR, PACKAGE_FIGS]:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")


def make_figures(summary_df: pd.DataFrame, zonal_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )
    title_map = {
        "Conventional": "Conventional",
        "AlphaEarth Embeddings": "AlphaEarth\nEmbeddings",
        "Conventional + AlphaEarth Embeddings": "Conventional +\nAlphaEarth Embeddings",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), constrained_layout=False)
    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        arr, extent = read_downsample(Path(row["confidence_raster"]))
        arr = np.ma.masked_where(arr == float(FLOAT_NODATA), arr)
        im = ax.imshow(arr, extent=extent, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title_map.get(row["feature_set"], row["feature_set"]), fontweight="bold")
        ax.set_xlabel("Longitude", labelpad=2)
        ax.set_ylabel("Latitude", labelpad=2)
        ax.tick_params(length=3, width=0.8)
        ax.grid(color="#ffffff", alpha=0.25, linewidth=0.3)
    fig.suptitle("CPEC 2018 Transfer-Confidence Maps", fontweight="bold", fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.89, bottom=0.12, top=0.82, wspace=0.24)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.014, 0.58])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Transfer confidence (0-1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_all(fig, "figure_cpec_2018_transfer_confidence_maps")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), constrained_layout=False)
    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        arr, extent = read_downsample(Path(row["mask_raster"]))
        arr = np.ma.masked_where(arr == int(UINT8_NODATA), arr)
        im = ax.imshow(arr, extent=extent, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title_map.get(row["feature_set"], row["feature_set"]), fontweight="bold")
        ax.set_xlabel("Longitude", labelpad=2)
        ax.set_ylabel("Latitude", labelpad=2)
        ax.tick_params(length=3, width=0.8)
        ax.grid(color="#ffffff", alpha=0.25, linewidth=0.3)
    fig.suptitle("CPEC 2018 Area of Applicability", fontweight="bold", fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.89, bottom=0.12, top=0.82, wspace=0.24)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.014, 0.58])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Inside area of applicability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_all(fig, "figure_cpec_2018_area_of_applicability_masks")
    plt.close(fig)

    feature_order = ["Conventional", "AlphaEarth Embeddings", "Conventional + AlphaEarth Embeddings"]
    pivot = zonal_df.pivot(index="domain", columns="feature_set", values="in_area_of_applicability_percent")
    pivot = pivot.reindex(DOMAIN_ORDER)[feature_order]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    im = ax.imshow(pivot.values, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(
        np.arange(pivot.shape[1]),
        ["Conventional", "AlphaEarth\nEmbeddings", "Conventional +\nAlphaEarth\nEmbeddings"],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(
        np.arange(pivot.shape[0]),
        [
            "Kashgar\n(Xinjiang, China)",
            "Gilgit-Baltistan",
            "KP-AJK",
            "Balochistan",
            "Punjab-Sindh\nlowland corridor",
        ],
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color="#0f172a")
    ax.set_title("Area-of-Applicability Coverage by CPEC Subdomain", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, label="Study-area pixels inside AOA (%)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_all(fig, "figure_cpec_2018_aoa_coverage_by_subdomain")
    plt.close(fig)


def write_report(
    method_info: list[dict[str, object]],
    summary_df: pd.DataFrame,
    zonal_df: pd.DataFrame,
) -> None:
    method_table = summary_df[
        [
            "feature_set",
            "valid_study_area_pixels",
            "in_area_of_applicability_percent",
            "mean_transfer_confidence",
            "mean_dissimilarity_index",
        ]
    ].copy()
    method_table["in_area_of_applicability_percent"] = method_table[
        "in_area_of_applicability_percent"
    ].map(lambda x: f"{x:.2f}")
    method_table["mean_transfer_confidence"] = method_table["mean_transfer_confidence"].map(lambda x: f"{x:.3f}")
    method_table["mean_dissimilarity_index"] = method_table["mean_dissimilarity_index"].map(lambda x: f"{x:.3f}")

    zonal_table = zonal_df.copy()
    zonal_table["in_area_of_applicability_percent"] = zonal_table[
        "in_area_of_applicability_percent"
    ].map(lambda x: f"{x:.2f}")
    zonal_table["mean_transfer_confidence"] = zonal_table["mean_transfer_confidence"].map(lambda x: f"{x:.3f}")

    report = f"""# CPEC 2018 Area-of-Applicability and Transfer Confidence

Date: 2026-05-10

## Purpose

This output converts the sample-level transferability analysis into raster-level reliability information. It identifies where 2018 susceptibility predictions are made inside the environmental/embedding feature space represented by the training samples, and where predictions should be interpreted more cautiously because they require extrapolation.

## Method Summary

The method follows the area-of-applicability logic used in spatial machine learning and environmental prediction: predictions are most reliable where the predictor-space dissimilarity from training samples is low. For each feature set, predictors were standardized using the 2018 sample table, weighted by the fitted Extra Trees feature importance, compressed with PCA when more than 10 predictors were present, and compared to the nearest training sample in feature space. PCA was used to avoid unstable high-dimensional distances while retaining the dominant predictor-space structure. The area-of-applicability threshold is the 95th percentile of spatial-CV validation-to-training nearest-neighbour distances, so the threshold is calibrated to the same spatial generalization problem used in model evaluation.

Three products were created for each feature set:

- Dissimilarity index: nearest-neighbour feature-space distance divided by the 95th percentile training threshold. Lower is more familiar.
- Area-of-applicability mask: 1 means the pixel is inside the 95th percentile training envelope; 0 means extrapolative.
- Transfer confidence: 0-1 map derived from the dissimilarity index. Higher means the pixel is closer to known training conditions.

## Overall Coverage

{df_to_markdown(method_table)}

## Subdomain Coverage

{df_to_markdown(zonal_table)}

## Method Parameters

```json
{json.dumps(method_info, indent=2)}
```

## Literature Basis

- Area-of-applicability and dissimilarity-index concepts are based on spatial/environmental ML transferability literature, especially Meyer and Pebesma's work on predicting into unknown space: https://doi.org/10.1111/2041-210X.13650
- The implementation follows the same core logic used in the CAST area-of-applicability family of spatial prediction tools: https://arxiv.org/abs/2404.06978
- Recent landslide susceptibility and transfer-learning papers emphasize that high model accuracy alone is not enough; spatial transferability, domain shift, and extrapolation risk must also be reported.
- This step therefore supports the updated teacher-approved direction: transferability-aware dynamic landslide susceptibility mapping for CPEC/KKH.

## Subdomain Naming Note

The subdomain labels are analysis-domain labels created for transferability and reporting. They are based on political boundaries but grouped where sample counts would otherwise be too small for stable spatial transfer testing.

- Former FATA is assigned to KP because it was merged with Khyber Pakhtunkhwa under Pakistan's 25th Constitutional Amendment in 2018.
- AJK is grouped with KP only as a northern-western mountainous transfer domain; it is not treated as the same administrative unit.
- Islamabad is grouped into the Punjab-Sindh lowland corridor only for modelling stability because it had one sample in the 2018 V3 table.

## Output Folders

- Managed rasters: `{OUT_DIR}`
- Clean package rasters: `{PACKAGE_MAPS}`
- Figures: `{PACKAGE_FIGS}`
"""
    for path in [
        REPORT_DIR / "cpec_2018_area_of_applicability_transfer_confidence_report.md",
        PACKAGE_METHODS / "cpec_2018_area_of_applicability_transfer_confidence_report.md",
    ]:
        path.write_text(report, encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def copy_outputs_to_package(summary_df: pd.DataFrame) -> None:
    for _, row in summary_df.iterrows():
        for key in ["index_raster", "mask_raster", "confidence_raster"]:
            src = Path(row[key])
            shutil.copy2(src, PACKAGE_MAPS / src.name)
    shutil.copy2(
        Path(__file__),
        PACKAGE_SCRIPTS / "create_2018_area_of_applicability_transfer_confidence.py",
    )


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(SAMPLE)
    method_info = []
    summaries = []

    with rasterio.open(CONVENTIONAL_BASE) as ref:
        study_mask = rasterize_study_area(ref)
        domain_masks = build_admin_domain_rasters(ref)
        print(f"Study-area pixels: {int(study_mask.sum()):,}", flush=True)

    for key in ["conventional", "alphaearth_embeddings", "conventional_alphaearth_embeddings"]:
        config = FEATURE_SETS[key]
        print(f"Fitting feature space: {config.label}", flush=True)
        feature_space, info = fit_feature_space(df, config)
        method_info.append(info)
        print(
            f"Processing raster AOA: {config.label} "
            f"({info['n_features_original']} original features -> {info['n_features_for_distance']} distance dimensions)",
            flush=True,
        )
        if key == "conventional":
            summary = process_conventional(feature_space, study_mask, block_size=512)
        else:
            summary = process_alpha_or_fused(feature_space, study_mask, block_size=512)
        summaries.append(summary)
        print(
            f"  coverage {summary['in_area_of_applicability_percent']:.2f}% | "
            f"mean confidence {summary['mean_transfer_confidence']:.3f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "cpec_2018_area_of_applicability_summary.csv", index=False)
    summary_df.to_csv(PACKAGE_TABLES / "cpec_2018_area_of_applicability_summary.csv", index=False)

    zonal_df = zonal_summary_by_domain(summary_df, domain_masks)
    zonal_df.to_csv(OUT_DIR / "cpec_2018_area_of_applicability_by_subdomain.csv", index=False)
    zonal_df.to_csv(PACKAGE_TABLES / "cpec_2018_area_of_applicability_by_subdomain.csv", index=False)

    make_figures(summary_df, zonal_df)
    write_report(method_info, summary_df, zonal_df)
    copy_outputs_to_package(summary_df)
    print(f"Saved AOA outputs to: {OUT_DIR}", flush=True)
    print(f"Saved clean package copies to: {PACKAGE_MAPS}", flush=True)


if __name__ == "__main__":
    main()
