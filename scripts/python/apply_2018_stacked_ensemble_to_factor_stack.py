"""Apply the 2018 fused stacked ensemble to a local 73-band predictor stack.

Expected input:
    D:\DING PROJECT\04_maps\stacked_ensemble_inputs_250m
    └── cpec_2018_fused_predictor_stack_250m_no_nodata*.tif

Output:
    D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble
    └── cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(r"D:\DING PROJECT")
MODEL_DIR = PROJECT_ROOT / "03_models" / "meta_ensemble_2018_spatial_cv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "04_maps" / "stacked_ensemble_inputs_250m"
DEFAULT_OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
DEFAULT_OUTPUT = "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif"

CONVENTIONAL_REDUCED = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
]
ALPHA = [f"A{i:02d}" for i in range(64)]
FUSED_BANDS = CONVENTIONAL_REDUCED + ALPHA
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]


def find_stack(input_dir: Path) -> Path:
    matches = sorted(input_dir.glob("cpec_2018_fused_predictor_stack_250m_no_nodata*.tif"))
    if not matches:
        raise FileNotFoundError(f"No fused predictor stack GeoTIFF found in {input_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one predictor stack, found {len(matches)}: {matches}")
    return matches[0]


def load_models() -> tuple[dict[str, object], object]:
    base = {}
    for key in BASE_MODEL_KEYS:
        path = MODEL_DIR / f"model_fused_reduced_{key}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        base[key] = joblib.load(path)
    meta_path = MODEL_DIR / "model_fused_reduced_stacked_l2_logistic.joblib"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return base, joblib.load(meta_path)


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype="float32")


def band_order(src: rasterio.DatasetReader) -> list[int]:
    desc = list(src.descriptions)
    if all(b in desc for b in FUSED_BANDS):
        return [desc.index(b) + 1 for b in FUSED_BANDS]
    if src.count != len(FUSED_BANDS):
        raise ValueError(f"Input has {src.count} bands and no complete band descriptions; expected {len(FUSED_BANDS)}.")
    return list(range(1, len(FUSED_BANDS) + 1))


def iter_windows(width: int, height: int, block_size: int) -> list[Window]:
    windows = []
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            windows.append(Window(col_off, row_off, w, h))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    stack_path = args.input if args.input else find_stack(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / DEFAULT_OUTPUT
    base_models, meta_model = load_models()

    with rasterio.open(stack_path) as src:
        indexes = band_order(src)
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
        with rasterio.open(out_path, "w", **profile) as dst:
            for window in iter_windows(src.width, src.height, args.block_size):
                arr = src.read(indexes=indexes, window=window, out_dtype="float32")
                bands, rows, cols = arr.shape
                flat = arr.reshape(bands, rows * cols).T
                invalid = ~np.isfinite(flat).all(axis=1)
                if invalid.any():
                    means = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
                    flat[invalid] = np.where(np.isfinite(flat[invalid]), flat[invalid], means)
                x = pd.DataFrame(flat, columns=FUSED_BANDS)
                base_probs = pd.DataFrame(
                    {key: predict_positive(model, x) for key, model in base_models.items()},
                    columns=BASE_MODEL_KEYS,
                )
                prob = predict_positive(meta_model, base_probs).reshape(rows, cols).astype("float32")
                prob = np.clip(prob, 0.0, 1.0)
                dst.write(prob, 1, window=window)
            dst.update_tags(
                year="2018",
                model="Spatial-CV Stacked Ensemble",
                feature_set="Conventional + AlphaEarth Embeddings",
                output_type="continuous_probability_no_classes",
                source_predictor_stack=str(stack_path),
                base_models=",".join(BASE_MODEL_KEYS),
            )
            dst.set_band_description(1, "prob_conventional_alphaearth_embeddings_stacked_ensemble")

    print(out_path)


if __name__ == "__main__":
    main()
