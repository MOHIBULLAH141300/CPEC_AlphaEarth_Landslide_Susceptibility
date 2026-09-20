from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window


PROJECT_ROOT = Path(r"D:\DING PROJECT")
YEAR = 2018
WORKSPACE = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
YEAR_DIR = WORKSPACE / str(YEAR)
RAW_DIR = YEAR_DIR / "00_raw_exports_from_gee"
SEISMIC_DIR = YEAR_DIR / "01_predictor_rasters_250m" / "annual_seismicity"
PROB_DIR = YEAR_DIR / "02_probability_maps_250m"
OUT_DIR = YEAR_DIR / "03_scenario_maps_250m" / "seismic_rainfall"
QA_DIR = YEAR_DIR / "05_year_specific_qa"

MODEL_DIR = PROJECT_ROOT / "03_models" / "seismic_factor_ablation_2018"
SAMPLE_TABLE = MODEL_DIR / "cpec_2018_lsm_samples_v3_seismic.csv"
LOCAL_V3 = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_local_v3_factors_250m"
    / "cpec_2018_local_v3_factors_250m.tif"
)
GEM_PGA_SOURCE = (
    PROJECT_ROOT
    / r"01_clean_data\new_step_required_data\B3_seismic_pga"
    / r"GEM-GSHM_PGA-475y-rock_v2023\v2023_1_pga_475_rock_3min.tif"
)
GEM_PGA_ALIGNED = SEISMIC_DIR / "cpec_2018_gem_pga_475yr_rock_g_250m.tif"
TEMPLATE_PROB = PROB_DIR / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"

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
ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]
SEISMIC_BANDS = [
    "gem_pga_475yr_rock_g",
    "annual_mag_weighted_eq_density_m4plus_per_1000km2",
    "distance_to_annual_eq_m5plus_km",
    "annual_max_shakemap_pga_percent_g",
]
FUSED_SEISMIC_BANDS = BASE_BANDS + LOCAL_BANDS + ALPHA_BANDS + SEISMIC_BANDS
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]
NODATA = -9999.0

SCENARIOS = {
    "baseline": {
        "label": "2018 baseline with seismic factors",
        "heavy_rain": False,
        "strong_eq": False,
        "path": OUT_DIR / "cpec_2018_fused_seismic_stacked_baseline_probability_250m.tif",
    },
    "heavy_rainfall": {
        "label": "Observed-period heavy-rainfall stress",
        "heavy_rain": True,
        "strong_eq": False,
        "path": OUT_DIR / "cpec_2018_fused_seismic_stacked_heavy_rainfall_stress_probability_250m.tif",
    },
    "strong_earthquake": {
        "label": "Observed-period strong-earthquake stress",
        "heavy_rain": False,
        "strong_eq": True,
        "path": OUT_DIR / "cpec_2018_fused_seismic_stacked_strong_earthquake_stress_probability_250m.tif",
    },
    "compound": {
        "label": "Compound heavy-rainfall and strong-earthquake stress",
        "heavy_rain": True,
        "strong_eq": True,
        "path": OUT_DIR / "cpec_2018_fused_seismic_stacked_compound_heavy_rainfall_strong_earthquake_probability_250m.tif",
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def annual_base_stack(year: int) -> Path:
    return WORKSPACE / str(year) / "00_raw_exports_from_gee" / f"cpec_{year}_annual_dynamic_base_predictors_250m.tif"


def annual_seismic_raster(year: int, kind: str) -> Path:
    root = WORKSPACE / str(year) / "01_predictor_rasters_250m" / "annual_seismicity"
    names = {
        "mag_density": f"cpec_{year}_annual_magnitude_weighted_earthquake_kernel_density_m4plus_per_1000km2_250m.tif",
        "dist_m5": f"cpec_{year}_distance_to_nearest_annual_earthquake_m5plus_km_250m.tif",
        "pga": f"cpec_{year}_annual_max_shakemap_pga_percent_g_250m.tif",
    }
    return root / names[kind]


def alpha_tiles() -> list[Path]:
    tiles = sorted(RAW_DIR.glob(f"cpec_{YEAR}_alphaearth_embeddings_250m*.tif"))
    if len(tiles) != 6:
        raise RuntimeError(f"Expected 6 AlphaEarth tiles, found {len(tiles)}")
    return tiles


def load_models() -> tuple[dict[str, object], object]:
    base = {}
    for key in BASE_MODEL_KEYS:
        path = MODEL_DIR / f"model_fused_gem_usgs_seismic_{key}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        base[key] = joblib.load(path)
    meta = MODEL_DIR / "model_fused_gem_usgs_seismic_stacked_l2_logistic.joblib"
    if not meta.exists():
        raise FileNotFoundError(meta)
    return base, joblib.load(meta)


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


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def initialize_nodata(dst: rasterio.io.DatasetWriter, block_size: int = 256) -> None:
    for win in iter_windows(dst.width, dst.height, block_size):
        dst.write(np.full((int(win.height), int(win.width)), NODATA, dtype="float32"), 1, window=win)


def tile_offset(ref: rasterio.DatasetReader, tile: rasterio.DatasetReader) -> tuple[int, int]:
    col = int(round((tile.transform.c - ref.transform.c) / ref.transform.a))
    row = int(round((ref.transform.f - tile.transform.f) / abs(ref.transform.e)))
    return row, col


def ensure_aligned_gem_pga() -> Path:
    if GEM_PGA_ALIGNED.exists():
        return GEM_PGA_ALIGNED
    GEM_PGA_ALIGNED.parent.mkdir(parents=True, exist_ok=True)
    ref_path = annual_base_stack(YEAR)
    log("Creating aligned 250 m GEM PGA raster")
    with rasterio.open(ref_path) as ref, rasterio.open(GEM_PGA_SOURCE) as src:
        profile = output_profile(ref)
        with rasterio.open(GEM_PGA_ALIGNED, "w", **profile) as dst:
            data = np.full((ref.height, ref.width), NODATA, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )
            dst.write(data, 1)
            dst.set_band_description(1, "gem_pga_475yr_rock_g")
            dst.update_tags(source=str(GEM_PGA_SOURCE), unit="fraction_g")
    return GEM_PGA_ALIGNED


def fill_values() -> dict[str, float]:
    df = pd.read_csv(SAMPLE_TABLE, encoding="utf-8-sig")
    fills = {}
    for band in FUSED_SEISMIC_BANDS:
        if band in df.columns:
            fills[band] = float(pd.to_numeric(df[band], errors="coerce").median())
        else:
            fills[band] = 0.0
    return fills


def clean_matrix(flat: np.ndarray, bands: list[str], fills: dict[str, float]) -> pd.DataFrame:
    for idx, band in enumerate(bands):
        col = flat[:, idx]
        bad = ~np.isfinite(col) | np.isclose(col, NODATA)
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


def stress_rain_window(base_sources: dict[int, rasterio.io.DatasetReader], win: Window) -> tuple[np.ndarray, np.ndarray]:
    monsoon = None
    max1 = None
    for src in base_sources.values():
        arr = src.read(indexes=[4, 5], window=win, out_dtype="float32")
        if monsoon is None:
            monsoon = arr[0]
            max1 = arr[1]
        else:
            monsoon = np.maximum(monsoon, arr[0])
            max1 = np.maximum(max1, arr[1])
    return monsoon.astype("float32"), max1.astype("float32")


def stress_seismic_window(seis_sources: dict[str, dict[int, rasterio.io.DatasetReader]], win: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = None
    dist = None
    pga = None
    for year in range(2017, 2025):
        mag_arr = seis_sources["mag_density"][year].read(1, window=win, out_dtype="float32")
        dist_arr = seis_sources["dist_m5"][year].read(1, window=win, out_dtype="float32")
        pga_arr = seis_sources["pga"][year].read(1, window=win, out_dtype="float32")
        if mag is None:
            mag = mag_arr
            dist = dist_arr
            pga = pga_arr
        else:
            mag = np.maximum(mag, mag_arr)
            dist = np.minimum(dist, dist_arr)
            pga = np.maximum(pga, pga_arr)
    return mag.astype("float32"), dist.astype("float32"), pga.astype("float32")


def probability_for_window(
    arr: np.ndarray,
    mask_window: np.ndarray,
    fills: dict[str, float],
    base_models: dict[str, object],
    meta_model: object,
) -> np.ndarray:
    band_count, rows, cols = arr.shape
    out = np.full((rows, cols), NODATA, dtype="float32")
    flat_mask = mask_window.reshape(rows * cols)
    if not np.any(flat_mask):
        return out
    flat = arr.reshape(band_count, rows * cols).T
    x = clean_matrix(flat[flat_mask].copy(), FUSED_SEISMIC_BANDS, fills)
    out.reshape(rows * cols)[flat_mask] = stacked_probability(base_models, meta_model, x)
    return out


def probabilities_for_scenario_batch(
    scenario_arrays: dict[str, np.ndarray],
    mask_window: np.ndarray,
    fills: dict[str, float],
    base_models: dict[str, object],
    meta_model: object,
) -> dict[str, np.ndarray]:
    first = next(iter(scenario_arrays.values()))
    band_count, rows, cols = first.shape
    flat_mask = mask_window.reshape(rows * cols)
    outputs = {
        key: np.full((rows, cols), NODATA, dtype="float32")
        for key in scenario_arrays
    }
    if not np.any(flat_mask):
        return outputs

    valid_count = int(flat_mask.sum())
    matrices = []
    for arr in scenario_arrays.values():
        flat = arr.reshape(band_count, rows * cols).T
        matrices.append(flat[flat_mask].copy())
    x = clean_matrix(np.vstack(matrices), FUSED_SEISMIC_BANDS, fills)
    probs = stacked_probability(base_models, meta_model, x)

    start = 0
    for key in scenario_arrays:
        stop = start + valid_count
        outputs[key].reshape(rows * cols)[flat_mask] = probs[start:stop]
        start = stop
    return outputs


def raster_stats(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        valid = data.compressed()
        return {
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "valid_pixels": int(valid.size),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "mean": float(valid.mean()) if valid.size else None,
            "p90": float(np.percentile(valid, 90)) if valid.size else None,
        }


def create_difference_maps(prob_paths: dict[str, Path], profile: dict) -> list[Path]:
    outputs = []
    baseline = prob_paths["baseline"]
    with rasterio.open(baseline) as base_src:
        for key in ["heavy_rainfall", "strong_earthquake", "compound"]:
            out_path = OUT_DIR / f"cpec_2018_fused_seismic_stacked_{key}_minus_baseline_probability_250m.tif"
            with rasterio.open(prob_paths[key]) as scen_src, rasterio.open(out_path, "w", **profile) as dst:
                for win in iter_windows(base_src.width, base_src.height, 256):
                    base = base_src.read(1, window=win, out_dtype="float32")
                    scen = scen_src.read(1, window=win, out_dtype="float32")
                    valid = (~np.isclose(base, NODATA)) & (~np.isclose(scen, NODATA))
                    diff = np.where(valid, scen - base, NODATA).astype("float32")
                    dst.write(diff, 1, window=win)
                dst.set_band_description(1, f"delta_probability_{key}_minus_baseline")
            outputs.append(out_path)
    return outputs


def apply_scenarios(block_size: int = 256) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_aligned_gem_pga()
    base_models, meta_model = load_models()
    fills = fill_values()
    base_stack = annual_base_stack(YEAR)
    tiles = alpha_tiles()

    base_sources = {year: rasterio.open(annual_base_stack(year)) for year in range(2017, 2025)}
    seis_sources = {
        "mag_density": {year: rasterio.open(annual_seismic_raster(year, "mag_density")) for year in range(2017, 2025)},
        "dist_m5": {year: rasterio.open(annual_seismic_raster(year, "dist_m5")) for year in range(2017, 2025)},
        "pga": {year: rasterio.open(annual_seismic_raster(year, "pga")) for year in range(2017, 2025)},
    }
    out_datasets = {}
    try:
        with rasterio.open(base_stack) as base_src, rasterio.open(LOCAL_V3) as local_src, rasterio.open(
            TEMPLATE_PROB
        ) as mask_src, rasterio.open(GEM_PGA_ALIGNED) as gem_src, rasterio.open(
            annual_seismic_raster(YEAR, "mag_density")
        ) as mag2018_src, rasterio.open(
            annual_seismic_raster(YEAR, "dist_m5")
        ) as dist2018_src, rasterio.open(
            annual_seismic_raster(YEAR, "pga")
        ) as pga2018_src:
            profile = output_profile(base_src)
            for key, cfg in SCENARIOS.items():
                dst = rasterio.open(cfg["path"], "w", **profile)
                initialize_nodata(dst, block_size)
                dst.set_band_description(1, f"prob_{key}_fused_seismic_stacked")
                dst.update_tags(year=str(YEAR), model="Spatial-CV Stacked Ensemble", scenario=cfg["label"])
                out_datasets[key] = dst

            for tile_path in tiles:
                with rasterio.open(tile_path) as tile:
                    row0, col0 = tile_offset(base_src, tile)
                    log(f"Processing AlphaEarth tile {tile_path.name}")
                    written_windows = 0
                    for tile_win in iter_windows(tile.width, tile.height, block_size):
                        dst_win = Window(col0 + tile_win.col_off, row0 + tile_win.row_off, tile_win.width, tile_win.height)
                        mask = mask_src.read(1, window=dst_win) != mask_src.nodata
                        if not np.any(mask):
                            continue
                        base_arr = base_src.read(window=dst_win, out_dtype="float32")
                        local_arr = local_src.read(window=dst_win, out_dtype="float32")
                        alpha_arr = tile.read(window=tile_win, out_dtype="float32")
                        gem = gem_src.read(1, window=dst_win, out_dtype="float32")
                        mag2018 = mag2018_src.read(1, window=dst_win, out_dtype="float32")
                        dist2018 = dist2018_src.read(1, window=dst_win, out_dtype="float32")
                        pga2018 = pga2018_src.read(1, window=dst_win, out_dtype="float32")
                        rain_monsoon_max, rain_1day_max = stress_rain_window(base_sources, dst_win)
                        mag_strong, dist_strong, pga_strong = stress_seismic_window(seis_sources, dst_win)

                        scenario_arrays = {}
                        for key, cfg in SCENARIOS.items():
                            scenario_base = base_arr.copy()
                            if cfg["heavy_rain"]:
                                scenario_base[3] = rain_monsoon_max
                                scenario_base[4] = rain_1day_max
                            if cfg["strong_eq"]:
                                seismic = np.stack([gem, mag_strong, dist_strong, pga_strong]).astype("float32")
                            else:
                                seismic = np.stack([gem, mag2018, dist2018, pga2018]).astype("float32")
                            scenario_arrays[key] = np.concatenate([scenario_base, local_arr, alpha_arr, seismic], axis=0)

                        probabilities = probabilities_for_scenario_batch(
                            scenario_arrays,
                            mask,
                            fills,
                            base_models,
                            meta_model,
                        )
                        for key, prob in probabilities.items():
                            out_datasets[key].write(prob, 1, window=dst_win)
                        written_windows += 1
                        if written_windows % 25 == 0:
                            log(f"  wrote {written_windows} valid windows for {tile_path.name}")
                    log(f"Finished {tile_path.name}: {written_windows} valid windows")
            for dst in out_datasets.values():
                dst.close()
            out_datasets = {}
            diff_paths = create_difference_maps({k: Path(v["path"]) for k, v in SCENARIOS.items()}, profile)
    finally:
        for dst in out_datasets.values():
            dst.close()
        for src in base_sources.values():
            src.close()
        for group in seis_sources.values():
            for src in group.values():
                src.close()

    all_paths = [Path(cfg["path"]) for cfg in SCENARIOS.values()] + diff_paths
    qa = pd.DataFrame([raster_stats(path) for path in all_paths])
    qa.to_csv(QA_DIR / "cpec_2018_fused_seismic_rainfall_scenario_raster_qa.csv", index=False, encoding="utf-8-sig")
    (QA_DIR / "cpec_2018_fused_seismic_rainfall_scenario_raster_qa.json").write_text(
        json.dumps(qa.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    lines = [
        "# 2018 Fused Seismic-Rainfall Scenario Probability Rasters",
        "",
        "Model: fused Conventional + AlphaEarth Embeddings + screened seismic factors, Spatial-CV Stacked Ensemble.",
        "",
        "Scenario definitions:",
        "",
        "- Baseline: 2018 rainfall and 2018 annual seismic factors.",
        "- Heavy rainfall: 2018 baseline except monsoon rainfall and maximum 1-day rainfall are replaced by the per-pixel maximum across 2017-2024 annual predictor stacks.",
        "- Strong earthquake: 2018 baseline except annual magnitude-weighted earthquake density and annual maximum ShakeMap PGA are replaced by their per-pixel maxima across 2017-2024, and distance to annual M5+ earthquake is replaced by the per-pixel minimum across 2017-2024.",
        "- Compound: heavy-rainfall and strong-earthquake changes applied together.",
        "",
        "Outputs:",
        "",
    ]
    for path in all_paths:
        lines.append(f"- `{path}`")
    lines.extend(["", f"QA table: `{QA_DIR / 'cpec_2018_fused_seismic_rainfall_scenario_raster_qa.csv'}`"])
    report = QA_DIR / "cpec_2018_fused_seismic_rainfall_scenario_raster_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(str(report))
    log(qa.to_string(index=False))


if __name__ == "__main__":
    apply_scenarios()
