"""Apply fused stacked ensemble as resumable probability tiles, then mosaic.

This is the safer version for the heaviest raster:
Conventional + AlphaEarth Embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(r"D:\DING PROJECT")
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
GEE_DIR = PROJECT_ROOT / "04_maps" / "stacked_ensemble_gee_inputs_250m"
FUSED_DIR = GEE_DIR / "conventional_base_alphaearth_embeddings"
LOCAL_V3 = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
REF = (
    GEE_DIR
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
TILE_OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble_tiles" / "fused"
OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
OUT = OUT_DIR / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif"
QA_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble_qa"

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
ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]
FUSED_BANDS = CONVENTIONAL_BASE_BANDS + LOCAL_V3_BANDS + ALPHA_BANDS
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]


def load_models() -> tuple[dict[str, object], object]:
    internal = "conventional_v3_alphaearth_embeddings"
    base = {}
    for key in BASE_MODEL_KEYS:
        base[key] = joblib.load(MODEL_DIR / f"model_{internal}_{key}.joblib")
    meta = joblib.load(MODEL_DIR / f"model_{internal}_stacked_l2_logistic.joblib")
    return base, meta


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype="float32")


def clean_matrix(flat: np.ndarray) -> pd.DataFrame:
    invalid = ~np.isfinite(flat).all(axis=1)
    if invalid.any():
        means = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
        flat[invalid] = np.where(np.isfinite(flat[invalid]), flat[invalid], means)
    return pd.DataFrame(flat, columns=FUSED_BANDS)


def stacked_probability(base_models: dict[str, object], meta_model: object, x: pd.DataFrame) -> np.ndarray:
    base_probs = pd.DataFrame(
        {key: predict_positive(model, x) for key, model in base_models.items()},
        columns=BASE_MODEL_KEYS,
    )
    return np.clip(predict_positive(meta_model, base_probs), 0.0, 1.0)


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def tile_offset(ref: rasterio.DatasetReader, tile: rasterio.DatasetReader) -> tuple[int, int]:
    col = int(round((tile.transform.c - ref.transform.c) / ref.transform.a))
    row = int(round((ref.transform.f - tile.transform.f) / abs(ref.transform.e)))
    return row, col


def valid_tile(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == 1 and src.width > 0 and src.height > 0
    except Exception:
        return False


def output_profile(src: rasterio.DatasetReader) -> dict:
    profile = src.profile.copy()
    profile.update(
        {
            "count": 1,
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
    return profile


def predict_tiles(block_size: int = 256) -> list[Path]:
    TILE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(FUSED_DIR.glob("*.tif"))
    if len(tiles) != 6:
        raise RuntimeError(f"Expected 6 fused input tiles, found {len(tiles)}")
    base_models, meta_model = load_models()
    outputs = []
    with rasterio.open(REF) as ref, rasterio.open(LOCAL_V3) as local_src:
        for tile_path in tiles:
            out_tile = TILE_OUT_DIR / tile_path.name.replace(
                "conventional_base_alphaearth_embeddings_gee_available_predictor_stack",
                "conventional_alphaearth_embeddings_stacked_ensemble_probability",
            )
            outputs.append(out_tile)
            if valid_tile(out_tile):
                print(f"SKIP existing {out_tile}", flush=True)
                continue
            print(f"START {tile_path.name}", flush=True)
            with rasterio.open(tile_path) as tile:
                row0, col0 = tile_offset(ref, tile)
                with rasterio.open(out_tile, "w", **output_profile(tile)) as dst:
                    for win in iter_windows(tile.width, tile.height, block_size):
                        tile_arr = tile.read(window=win, out_dtype="float32")
                        local_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        local_arr = local_src.read(window=local_win, out_dtype="float32")
                        base_arr = tile_arr[: len(CONVENTIONAL_BASE_BANDS), :, :]
                        alpha_arr = tile_arr[len(CONVENTIONAL_BASE_BANDS) :, :, :]
                        arr = np.concatenate([base_arr, local_arr, alpha_arr], axis=0)
                        n_bands, rows, cols = arr.shape
                        x = clean_matrix(arr.reshape(n_bands, rows * cols).T)
                        prob = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                        dst.write(prob, 1, window=win)
                    dst.set_band_description(1, "prob_conventional_alphaearth_embeddings_stacked_ensemble")
                    dst.update_tags(year="2018", model="Spatial-CV Stacked Ensemble", feature_set="Conventional + AlphaEarth Embeddings")
            print(f"DONE {out_tile}", flush=True)
    return outputs


def mosaic_tiles(tile_paths: list[Path]) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with rasterio.open(REF) as ref:
        with rasterio.open(OUT, "w", **output_profile(ref)) as dst:
            for tile_path in tile_paths:
                if not valid_tile(tile_path):
                    raise RuntimeError(f"Missing/invalid predicted tile: {tile_path}")
                with rasterio.open(tile_path) as tile:
                    row0, col0 = tile_offset(ref, tile)
                    data = tile.read(1, out_dtype="float32")
                    dst.write(data, 1, window=Window(col0, row0, tile.width, tile.height))
            dst.set_band_description(1, "prob_conventional_alphaearth_embeddings_stacked_ensemble")
            dst.update_tags(year="2018", model="Spatial-CV Stacked Ensemble", feature_set="Conventional + AlphaEarth Embeddings")
    with rasterio.open(OUT) as src:
        data = src.read(1)
        finite = np.isfinite(data)
        stats = {
            "path": str(OUT),
            "size_gb": round(OUT.stat().st_size / 1024**3, 3),
            "min": float(np.nanmin(data[finite])),
            "max": float(np.nanmax(data[finite])),
            "mean": float(np.nanmean(data[finite])),
            "finite_pixels": int(finite.sum()),
            "total_pixels": int(data.size),
            "nonfinite_pixels": int((~finite).sum()),
        }
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "fused_stacked_ensemble_probability_raster_2018_qa.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    tile_paths = predict_tiles()
    stats = mosaic_tiles(tile_paths)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
