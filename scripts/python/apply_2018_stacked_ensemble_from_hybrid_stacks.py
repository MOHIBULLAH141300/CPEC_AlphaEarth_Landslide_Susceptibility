"""Create the three 2018 stacked-ensemble probability rasters from hybrid stacks.

Inputs:
- GEE base conventional stack, 8 bands
- GEE AlphaEarth tile stack, 64 bands
- GEE base conventional + AlphaEarth tile stack, 72 bands
- local v3 stack, 11 bands

Outputs:
- Conventional stacked ensemble probability
- AlphaEarth Embeddings stacked ensemble probability
- Conventional + AlphaEarth Embeddings stacked ensemble probability

No susceptibility classes are created.
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
LOCAL_V3 = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
CONVENTIONAL_BASE = (
    GEE_DIR
    / "conventional_base"
    / "cpec_2018_conventional_base_gee_available_predictor_stack_250m.tif"
)
ALPHA_DIR = GEE_DIR / "alphaearth_embeddings"
FUSED_DIR = GEE_DIR / "conventional_base_alphaearth_embeddings"
OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
QA_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble_qa"
REPORT = PROJECT_ROOT / "05_reports" / "stacked_ensemble_probability_rasters_2018_report.md"

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
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]

OUTPUTS = {
    "conventional": {
        "internal": "conventional_v3",
        "label": "Conventional",
        "bands": CONVENTIONAL_BANDS,
        "path": OUT_DIR / "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
    },
    "alphaearth_embeddings": {
        "internal": "alphaearth_embeddings",
        "label": "AlphaEarth Embeddings",
        "bands": ALPHA_BANDS,
        "path": OUT_DIR / "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    },
    "conventional_alphaearth_embeddings": {
        "internal": "conventional_v3_alphaearth_embeddings",
        "label": "Conventional + AlphaEarth Embeddings",
        "bands": FUSED_BANDS,
        "path": OUT_DIR / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    },
}


def load_models(internal: str) -> tuple[dict[str, object], object]:
    base = {}
    for key in BASE_MODEL_KEYS:
        path = MODEL_DIR / f"model_{internal}_{key}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        base[key] = joblib.load(path)
    meta_path = MODEL_DIR / f"model_{internal}_stacked_l2_logistic.joblib"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return base, joblib.load(meta_path)


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype="float32")


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def output_profile(ref: rasterio.DatasetReader) -> dict:
    profile = ref.profile.copy()
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


def clean_matrix(flat: np.ndarray, bands: list[str]) -> pd.DataFrame:
    invalid = ~np.isfinite(flat).all(axis=1)
    if invalid.any():
        means = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
        flat[invalid] = np.where(np.isfinite(flat[invalid]), flat[invalid], means)
    return pd.DataFrame(flat, columns=bands)


def stacked_probability(base_models: dict[str, object], meta_model: object, x: pd.DataFrame) -> np.ndarray:
    base_probs = pd.DataFrame(
        {key: predict_positive(model, x) for key, model in base_models.items()},
        columns=BASE_MODEL_KEYS,
    )
    return np.clip(predict_positive(meta_model, base_probs), 0.0, 1.0)


def tile_offset(ref: rasterio.DatasetReader, tile: rasterio.DatasetReader) -> tuple[int, int]:
    col = int(round((tile.transform.c - ref.transform.c) / ref.transform.a))
    row = int(round((ref.transform.f - tile.transform.f) / abs(ref.transform.e)))
    return row, col


def apply_conventional(block_size: int = 256) -> dict[str, object]:
    cfg = OUTPUTS["conventional"]
    base_models, meta_model = load_models(str(cfg["internal"]))
    out_path = Path(cfg["path"])
    with rasterio.open(CONVENTIONAL_BASE) as base_src, rasterio.open(LOCAL_V3) as local_src:
        with rasterio.open(out_path, "w", **output_profile(base_src)) as dst:
            for win in iter_windows(base_src.width, base_src.height, block_size):
                base = base_src.read(window=win, out_dtype="float32")
                local = local_src.read(window=win, out_dtype="float32")
                arr = np.concatenate([base, local], axis=0)
                n_bands, rows, cols = arr.shape
                x = clean_matrix(arr.reshape(n_bands, rows * cols).T, CONVENTIONAL_BANDS)
                prob = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                dst.write(prob, 1, window=win)
            dst.set_band_description(1, "prob_conventional_stacked_ensemble")
            dst.update_tags(year="2018", model="Spatial-CV Stacked Ensemble", feature_set=str(cfg["label"]))
    return raster_stats(out_path)


def apply_alpha(block_size: int = 256) -> dict[str, object]:
    cfg = OUTPUTS["alphaearth_embeddings"]
    base_models, meta_model = load_models(str(cfg["internal"]))
    out_path = Path(cfg["path"])
    tiles = sorted(ALPHA_DIR.glob("*.tif"))
    if len(tiles) != 6:
        raise RuntimeError(f"Expected 6 AlphaEarth tiles, found {len(tiles)}")
    with rasterio.open(CONVENTIONAL_BASE) as ref:
        with rasterio.open(out_path, "w", **output_profile(ref)) as dst:
            for tile_path in tiles:
                with rasterio.open(tile_path) as tile:
                    row0, col0 = tile_offset(ref, tile)
                    for win in iter_windows(tile.width, tile.height, block_size):
                        arr = tile.read(window=win, out_dtype="float32")
                        n_bands, rows, cols = arr.shape
                        x = clean_matrix(arr.reshape(n_bands, rows * cols).T, ALPHA_BANDS)
                        prob = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                        dst_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        dst.write(prob, 1, window=dst_win)
            dst.set_band_description(1, "prob_alphaearth_embeddings_stacked_ensemble")
            dst.update_tags(year="2018", model="Spatial-CV Stacked Ensemble", feature_set=str(cfg["label"]))
    return raster_stats(out_path)


def apply_fused(block_size: int = 256) -> dict[str, object]:
    cfg = OUTPUTS["conventional_alphaearth_embeddings"]
    base_models, meta_model = load_models(str(cfg["internal"]))
    out_path = Path(cfg["path"])
    tiles = sorted(FUSED_DIR.glob("*.tif"))
    if len(tiles) != 6:
        raise RuntimeError(f"Expected 6 fused tiles, found {len(tiles)}")
    with rasterio.open(CONVENTIONAL_BASE) as ref, rasterio.open(LOCAL_V3) as local_src:
        with rasterio.open(out_path, "w", **output_profile(ref)) as dst:
            for tile_path in tiles:
                with rasterio.open(tile_path) as tile:
                    row0, col0 = tile_offset(ref, tile)
                    for win in iter_windows(tile.width, tile.height, block_size):
                        tile_arr = tile.read(window=win, out_dtype="float32")
                        local_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        local_arr = local_src.read(window=local_win, out_dtype="float32")
                        base_arr = tile_arr[: len(CONVENTIONAL_BASE_BANDS), :, :]
                        alpha_arr = tile_arr[len(CONVENTIONAL_BASE_BANDS) :, :, :]
                        arr = np.concatenate([base_arr, local_arr, alpha_arr], axis=0)
                        n_bands, rows, cols = arr.shape
                        x = clean_matrix(arr.reshape(n_bands, rows * cols).T, FUSED_BANDS)
                        prob = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                        dst_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        dst.write(prob, 1, window=dst_win)
            dst.set_band_description(1, "prob_conventional_alphaearth_embeddings_stacked_ensemble")
            dst.update_tags(year="2018", model="Spatial-CV Stacked Ensemble", feature_set=str(cfg["label"]))
    return raster_stats(out_path)


def raster_stats(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=False)
        finite = np.isfinite(data)
        return {
            "path": str(path),
            "size_gb": round(path.stat().st_size / 1024**3, 3),
            "bands": src.count,
            "shape": [src.height, src.width],
            "crs": str(src.crs),
            "res": list(src.res),
            "min": float(np.nanmin(data[finite])),
            "max": float(np.nanmax(data[finite])),
            "mean": float(np.nanmean(data[finite])),
            "finite_pixels": int(finite.sum()),
            "total_pixels": int(data.size),
            "nonfinite_pixels": int((~finite).sum()),
        }


def write_report(stats: list[dict[str, object]]) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(QA_DIR / "stacked_ensemble_probability_rasters_2018_qa.csv", index=False)
    (QA_DIR / "stacked_ensemble_probability_rasters_2018_qa.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 2018 Spatial-CV Stacked Ensemble Probability Rasters",
        "",
        "These are continuous probability rasters from the saved local Spatial-CV Stacked Ensemble models.",
        "",
        "| Raster | Size GB | Min | Max | Mean | Non-finite pixels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| `{row['path']}` | {row['size_gb']:.3f} | {row['min']:.6f} | {row['max']:.6f} | "
            f"{row['mean']:.6f} | {row['nonfinite_pixels']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Conventional and Fused rasters use the 8 GEE base factors plus the 11 local v3 factors.",
            "- AlphaEarth Embeddings uses the 64 GEE AlphaEarth bands.",
            "- No susceptibility classes are created.",
            "- A 30 m cartographic display resample can be created from these rasters after QA.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = [
        apply_conventional(),
        apply_alpha(),
        apply_fused(),
    ]
    write_report(stats)
    print(REPORT)


if __name__ == "__main__":
    main()
