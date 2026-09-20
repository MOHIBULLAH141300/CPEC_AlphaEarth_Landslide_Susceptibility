"""Apply the fixed 2018 V3 stacked models to one annual predictor year.

Inputs per year:
- annual GEE base predictors, 8 bands
- annual AlphaEarth embeddings, 64 bands, possibly split into multiple tiles
- fixed local V3 static/semi-static factor stack, 11 bands

Outputs per year:
- Conventional stacked probability, 250 m
- AlphaEarth Embeddings stacked probability, 250 m
- Conventional + AlphaEarth Embeddings stacked probability, 250 m

This script is for the Option 1 temporal workflow: the trained 2018 V3 model is
held fixed, and only annual dynamic predictor values change.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio
from rasterio import features
from rasterio.windows import Window

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT_ROOT = Path(r"D:\DING PROJECT")
WORKSPACE = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
SAMPLE_TABLE = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
LOCAL_V3 = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)

BASE_BANDS = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "modis_lc_type1",
]
LOCAL_BANDS = [
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
CONVENTIONAL_BANDS = BASE_BANDS + LOCAL_BANDS
ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]
FUSED_BANDS = CONVENTIONAL_BANDS + ALPHA_BANDS
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]
NODATA = -9999.0

OUTPUTS = {
    "conventional": {
        "internal": "conventional_v3",
        "label": "Conventional",
        "bands": CONVENTIONAL_BANDS,
        "description": "prob_conventional_stacked_ensemble",
    },
    "alphaearth_embeddings": {
        "internal": "alphaearth_embeddings",
        "label": "AlphaEarth Embeddings",
        "bands": ALPHA_BANDS,
        "description": "prob_alphaearth_embeddings_stacked_ensemble",
    },
    "conventional_alphaearth_embeddings": {
        "internal": "conventional_v3_alphaearth_embeddings",
        "label": "Conventional + AlphaEarth Embeddings",
        "bands": FUSED_BANDS,
        "description": "prob_conventional_alphaearth_embeddings_stacked_ensemble",
    },
}


def year_dirs(year: int) -> dict[str, Path]:
    root = WORKSPACE / str(year)
    return {
        "root": root,
        "raw": root / "00_raw_exports_from_gee",
        "pred": root / "02_probability_maps_250m",
        "qa": root / "05_year_specific_qa",
        "logs": root / "06_processing_logs",
    }


def find_single(raw_dir: Path, pattern: str) -> Path:
    matches = sorted(raw_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file for {pattern} in {raw_dir}, found {len(matches)}")
    return matches[0]


def find_alpha_tiles(raw_dir: Path, year: int) -> list[Path]:
    tiles = sorted(raw_dir.glob(f"cpec_{year}_alphaearth_embeddings_250m*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No AlphaEarth export tiles found in {raw_dir}")
    return tiles


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


def fill_values() -> dict[str, float]:
    df = pd.read_csv(SAMPLE_TABLE, encoding="utf-8-sig")
    bands = CONVENTIONAL_BANDS + ALPHA_BANDS
    out = {}
    for band in bands:
        if band in df.columns:
            out[band] = float(pd.to_numeric(df[band], errors="coerce").mean())
        else:
            out[band] = 0.0
    return out


def clean_matrix(flat: np.ndarray, bands: list[str], fills: dict[str, float]) -> pd.DataFrame:
    for idx, band in enumerate(bands):
        col = flat[:, idx]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = fills.get(band, 0.0)
            flat[:, idx] = col
    return pd.DataFrame(flat, columns=bands)


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype="float32")


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


def output_profile(ref: rasterio.DatasetReader) -> dict:
    profile = ref.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
        nodata=NODATA,
    )
    return profile


def study_mask(ref: rasterio.DatasetReader) -> np.ndarray:
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:32642", allow_override=True)
    boundary = boundary.to_crs(ref.crs)
    return features.rasterize(
        [(geom, 1) for geom in boundary.geometry if geom is not None and not geom.is_empty],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def tile_offset(ref: rasterio.DatasetReader, tile: rasterio.DatasetReader) -> tuple[int, int]:
    col = int(round((tile.transform.c - ref.transform.c) / ref.transform.a))
    row = int(round((ref.transform.f - tile.transform.f) / abs(ref.transform.e)))
    return row, col


def check_alignment(ref: rasterio.DatasetReader, other: rasterio.DatasetReader, name: str) -> None:
    if ref.crs != other.crs:
        raise ValueError(f"CRS mismatch for {name}: {ref.crs} vs {other.crs}")
    if abs(ref.transform.a - other.transform.a) > 1e-12 or abs(ref.transform.e - other.transform.e) > 1e-12:
        raise ValueError(f"Resolution mismatch for {name}: {ref.transform} vs {other.transform}")


def stats(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        valid = arr[(arr != NODATA) & np.isfinite(arr)]
        return {
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "valid_pixels": int(valid.size),
            "nodata_pixels": int(np.sum(arr == NODATA)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
        }


def apply_conventional(year: int, base_path: Path, mask: np.ndarray, fills: dict[str, float], block_size: int) -> dict[str, object]:
    cfg = OUTPUTS["conventional"]
    base_models, meta_model = load_models(cfg["internal"])
    dirs = year_dirs(year)
    out = dirs["pred"] / f"cpec_{year}_conventional_stacked_ensemble_probability_250m.tif"
    with rasterio.open(base_path) as base_src, rasterio.open(LOCAL_V3) as local_src:
        check_alignment(base_src, local_src, "local V3 stack")
        with rasterio.open(out, "w", **output_profile(base_src)) as dst:
            for win in iter_windows(base_src.width, base_src.height, block_size):
                base = base_src.read(window=win, out_dtype="float32")
                local = local_src.read(window=win, out_dtype="float32")
                arr = np.concatenate([base, local], axis=0)
                bands, rows, cols = arr.shape
                x = clean_matrix(arr.reshape(bands, rows * cols).T, CONVENTIONAL_BANDS, fills)
                pred = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                m = mask[
                    int(win.row_off) : int(win.row_off + win.height),
                    int(win.col_off) : int(win.col_off + win.width),
                ]
                pred[~m] = NODATA
                dst.write(pred, 1, window=win)
            dst.set_band_description(1, cfg["description"])
            dst.update_tags(year=str(year), model="Spatial-CV Stacked Ensemble", feature_set=cfg["label"])
    return stats(out)


def apply_alpha(year: int, base_path: Path, alpha_tiles: list[Path], mask: np.ndarray, fills: dict[str, float], block_size: int) -> dict[str, object]:
    cfg = OUTPUTS["alphaearth_embeddings"]
    base_models, meta_model = load_models(cfg["internal"])
    dirs = year_dirs(year)
    out = dirs["pred"] / f"cpec_{year}_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"
    with rasterio.open(base_path) as ref:
        with rasterio.open(out, "w", **output_profile(ref)) as dst:
            for tile_path in alpha_tiles:
                with rasterio.open(tile_path) as tile:
                    check_alignment(ref, tile, tile_path.name)
                    row0, col0 = tile_offset(ref, tile)
                    for win in iter_windows(tile.width, tile.height, block_size):
                        arr = tile.read(window=win, out_dtype="float32")
                        bands, rows, cols = arr.shape
                        x = clean_matrix(arr.reshape(bands, rows * cols).T, ALPHA_BANDS, fills)
                        pred = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                        dst_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        m = mask[
                            int(dst_win.row_off) : int(dst_win.row_off + dst_win.height),
                            int(dst_win.col_off) : int(dst_win.col_off + dst_win.width),
                        ]
                        pred[~m] = NODATA
                        dst.write(pred, 1, window=dst_win)
            dst.set_band_description(1, cfg["description"])
            dst.update_tags(year=str(year), model="Spatial-CV Stacked Ensemble", feature_set=cfg["label"])
    return stats(out)


def apply_fused(year: int, base_path: Path, alpha_tiles: list[Path], mask: np.ndarray, fills: dict[str, float], block_size: int) -> dict[str, object]:
    cfg = OUTPUTS["conventional_alphaearth_embeddings"]
    base_models, meta_model = load_models(cfg["internal"])
    dirs = year_dirs(year)
    out = dirs["pred"] / f"cpec_{year}_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"
    with rasterio.open(base_path) as base_src, rasterio.open(LOCAL_V3) as local_src:
        check_alignment(base_src, local_src, "local V3 stack")
        with rasterio.open(out, "w", **output_profile(base_src)) as dst:
            for tile_path in alpha_tiles:
                with rasterio.open(tile_path) as tile:
                    check_alignment(base_src, tile, tile_path.name)
                    row0, col0 = tile_offset(base_src, tile)
                    for win in iter_windows(tile.width, tile.height, block_size):
                        alpha = tile.read(window=win, out_dtype="float32")
                        dst_win = Window(col0 + win.col_off, row0 + win.row_off, win.width, win.height)
                        base = base_src.read(window=dst_win, out_dtype="float32")
                        local = local_src.read(window=dst_win, out_dtype="float32")
                        arr = np.concatenate([base, local, alpha], axis=0)
                        bands, rows, cols = arr.shape
                        x = clean_matrix(arr.reshape(bands, rows * cols).T, FUSED_BANDS, fills)
                        pred = stacked_probability(base_models, meta_model, x).reshape(rows, cols).astype("float32")
                        m = mask[
                            int(dst_win.row_off) : int(dst_win.row_off + dst_win.height),
                            int(dst_win.col_off) : int(dst_win.col_off + dst_win.width),
                        ]
                        pred[~m] = NODATA
                        dst.write(pred, 1, window=dst_win)
            dst.set_band_description(1, cfg["description"])
            dst.update_tags(year=str(year), model="Spatial-CV Stacked Ensemble", feature_set=cfg["label"])
    return stats(out)


def write_report(year: int, qa: list[dict[str, object]]) -> None:
    dirs = year_dirs(year)
    dirs["qa"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qa).to_csv(dirs["qa"] / f"cpec_{year}_annual_probability_qa.csv", index=False)
    (dirs["qa"] / f"cpec_{year}_annual_probability_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Annual Probability Raster QA: {year}",
        "",
        "Model strategy: fixed 2018 V3 Spatial-CV Stacked Ensemble applied to annual predictors.",
        "",
        "| Feature set | Valid pixels | Nodata pixels | Min | Max | Mean | Std | File |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = ["Conventional", "AlphaEarth Embeddings", "Conventional + AlphaEarth Embeddings"]
    for label, row in zip(labels, qa):
        lines.append(
            f"| {label} | {row['valid_pixels']} | {row['nodata_pixels']} | {row['min']:.4f} | {row['max']:.4f} | {row['mean']:.4f} | {row['std']:.4f} | `{row['path']}` |"
        )
    (dirs["qa"] / f"cpec_{year}_annual_probability_qa.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    dirs = year_dirs(args.year)
    for key in ["pred", "qa", "logs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    base_path = find_single(
        dirs["raw"], f"cpec_{args.year}_annual_dynamic_base_predictors_250m*.tif"
    )
    alpha_tiles = find_alpha_tiles(dirs["raw"], args.year)
    fills = fill_values()
    with rasterio.open(base_path) as ref:
        mask = study_mask(ref)

    qa = [
        apply_conventional(args.year, base_path, mask, fills, args.block_size),
        apply_alpha(args.year, base_path, alpha_tiles, mask, fills, args.block_size),
        apply_fused(args.year, base_path, alpha_tiles, mask, fills, args.block_size),
    ]
    write_report(args.year, qa)
    print(json.dumps(qa, indent=2))
    print(f"Wrote annual probability maps to: {dirs['pred']}")


if __name__ == "__main__":
    main()
