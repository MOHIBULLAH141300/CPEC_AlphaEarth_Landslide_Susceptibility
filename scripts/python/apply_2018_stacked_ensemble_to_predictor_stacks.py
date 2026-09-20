"""Apply the three 2018 Spatial-CV Stacked Ensemble models to predictor stacks.

Inputs are GeoTIFF predictor stacks exported from Earth Engine:

- cpec_2018_conventional_predictor_stack_250m_no_nodata*.tif
- cpec_2018_alphaearth_embeddings_predictor_stack_250m_no_nodata*.tif
- cpec_2018_conventional_alphaearth_embeddings_predictor_stack_250m_no_nodata*.tif

Outputs are continuous probability rasters only; no susceptibility classes are
created.
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
MODEL_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "04_maps" / "stacked_ensemble_inputs_250m"
DEFAULT_OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
CONVENTIONAL_LIST = (
    PROJECT_ROOT
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)

ALPHA = [f"A{i:02d}" for i in range(64)]
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]

FEATURE_SETS = {
    "conventional": {
        "internal": "conventional_v3",
        "label": "Conventional",
        "output": "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
    },
    "alphaearth_embeddings": {
        "internal": "alphaearth_embeddings",
        "label": "AlphaEarth Embeddings",
        "output": "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    },
    "conventional_alphaearth_embeddings": {
        "internal": "conventional_v3_alphaearth_embeddings",
        "label": "Conventional + AlphaEarth Embeddings",
        "output": "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    },
}


def feature_list(name: str) -> list[str]:
    conventional = pd.read_csv(CONVENTIONAL_LIST)["factor"].tolist()
    if name == "conventional":
        return conventional
    if name == "alphaearth_embeddings":
        return ALPHA
    if name == "conventional_alphaearth_embeddings":
        return conventional + ALPHA
    raise ValueError(name)


def find_stack(input_dir: Path, name: str) -> Path:
    matches = sorted(input_dir.glob(f"cpec_2018_{name}_predictor_stack_250m_no_nodata*.tif"))
    if not matches:
        raise FileNotFoundError(f"No {name} predictor stack GeoTIFF found in {input_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one {name} predictor stack, found {len(matches)}: {matches}")
    return matches[0]


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


def band_order(src: rasterio.DatasetReader, bands: list[str]) -> list[int]:
    desc = list(src.descriptions)
    if all(b in desc for b in bands):
        return [desc.index(b) + 1 for b in bands]
    if src.count != len(bands):
        raise ValueError(f"Input has {src.count} bands and no complete band descriptions; expected {len(bands)}.")
    return list(range(1, len(bands) + 1))


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def apply_one(name: str, input_dir: Path, out_dir: Path, block_size: int) -> Path:
    cfg = FEATURE_SETS[name]
    internal = str(cfg["internal"])
    label = str(cfg["label"])
    bands = feature_list(name)
    stack_path = find_stack(input_dir, name)
    out_path = out_dir / str(cfg["output"])
    base_models, meta_model = load_models(internal)

    with rasterio.open(stack_path) as src:
        indexes = band_order(src, bands)
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
            for window in iter_windows(src.width, src.height, block_size):
                arr = src.read(indexes=indexes, window=window, out_dtype="float32")
                n_bands, rows, cols = arr.shape
                flat = arr.reshape(n_bands, rows * cols).T
                invalid = ~np.isfinite(flat).all(axis=1)
                if invalid.any():
                    means = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
                    flat[invalid] = np.where(np.isfinite(flat[invalid]), flat[invalid], means)
                x = pd.DataFrame(flat, columns=bands)
                base_probs = pd.DataFrame(
                    {key: predict_positive(model, x) for key, model in base_models.items()},
                    columns=BASE_MODEL_KEYS,
                )
                prob = predict_positive(meta_model, base_probs).reshape(rows, cols).astype("float32")
                dst.write(np.clip(prob, 0.0, 1.0), 1, window=window)
            dst.update_tags(
                year="2018",
                model="Spatial-CV Stacked Ensemble",
                feature_set=label,
                output_type="continuous_probability_no_classes",
                source_predictor_stack=str(stack_path),
                base_models=",".join(BASE_MODEL_KEYS),
            )
            dst.set_band_description(1, f"prob_{name}_stacked_ensemble")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = list(FEATURE_SETS) if args.feature_set == "all" else [args.feature_set]
    for name in names:
        print(apply_one(name, args.input_dir, args.out_dir, args.block_size))


if __name__ == "__main__":
    main()
