"""Apply the corrected manuscript-v3 stacks to baseline 250 m predictor rasters."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import shapefile


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


PROJECT_ROOT = Path(r"D:\DING PROJECT")
MODEL_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv" / "models"
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
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\New Folder\cpec1.shp"
)
OUT_DIR = PROJECT_ROOT / "04_maps" / "manuscript_v3_baseline_susceptibility_scores_250m"
NODATA = -9999.0
DEFAULT_BLOCK_SIZE = 512
ALPHA_COLUMNS = [f"A{i:02d}" for i in range(64)]

OUTPUTS = {
    "conventional": OUT_DIR / "cpec_baseline_conventional_stacked_susceptibility_score_250m.tif",
    "alphaearth_embeddings": OUT_DIR
    / "cpec_baseline_alphaearth_embeddings_stacked_susceptibility_score_250m.tif",
    "conventional_alphaearth_embeddings": OUT_DIR
    / "cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif",
}


def windows(width: int, height: int, block_size: int):
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def study_mask(reference: rasterio.io.DatasetReader) -> np.ndarray:
    reader = shapefile.Reader(str(BOUNDARY), encoding="latin1")
    geometries = [record.shape.__geo_interface__ for record in reader.iterShapeRecords()]
    return rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def verify_alignment(reference, *datasets) -> None:
    for dataset in datasets:
        if (
            dataset.width != reference.width
            or dataset.height != reference.height
            or dataset.crs != reference.crs
            or dataset.transform != reference.transform
        ):
            raise ValueError(f"Raster alignment mismatch: {dataset.name}")


def conventional_frame(base, local, terrain, lithology, window: Window, mask: np.ndarray) -> pd.DataFrame:
    b = base.read([1, 4, 5, 6, 7, 8], window=window, out_dtype="float32")
    l = local.read([1, 2, 3, 5, 6, 7, 8, 9, 10, 11], window=window, out_dtype="float32")
    t = terrain.read([1, 2], window=window, out_dtype="float32")
    rock = lithology.read(1, window=window)
    rr, cc = np.nonzero(mask)
    aspect = t[1, rr, cc]
    return pd.DataFrame(
        {
            "elevation_m": b[0, rr, cc],
            "slope_deg": t[0, rr, cc],
            "rain_monsoon_total": b[1, rr, cc],
            "rain_max_1day": b[2, rr, cc],
            "ndvi_median": b[3, rr, cc],
            "ndvi_amplitude": b[4, rr, cc],
            "log1p_dist_road_m": l[0, rr, cc],
            "log1p_dist_river_m": l[1, rr, cc],
            "log1p_dist_fault_m": l[2, rr, cc],
            "profile_curvature": l[3, rr, cc],
            "plan_curvature": l[4, rr, cc],
            "tri": l[5, rr, cc],
            "twi": l[6, rr, cc],
            "valley_depth": l[7, rr, cc],
            "aspect_sin": np.sin(np.deg2rad(aspect)),
            "aspect_cos": np.cos(np.deg2rad(aspect)),
            "modis_lc_type1": b[5, rr, cc],
            "lithology_code": rock[rr, cc],
            "soil_type": l[8, rr, cc],
            "eq_density_ms5": l[9, rr, cc],
        }
    )


def alpha_frame(tile, window: Window, mask: np.ndarray) -> pd.DataFrame:
    values = tile.read(window=window, out_dtype="float32")
    rr, cc = np.nonzero(mask)
    return pd.DataFrame(values[:, rr, cc].T, columns=ALPHA_COLUMNS)


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    base_scores = np.column_stack(
        [
            bundle["base_models"][key].predict_proba(frame)[:, 1]
            for key in bundle["base_model_order"]
        ]
    )
    score = bundle["meta_model"].predict_proba(base_scores)[:, 1]
    disagreement = np.std(base_scores, axis=1, ddof=0)
    return np.clip(score, 0.0, 1.0).astype("float32"), disagreement.astype("float32")


def output_profile(reference) -> dict:
    profile = reference.profile.copy()
    profile.update(
        count=2,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def tile_offsets(path: Path) -> tuple[int, int]:
    groups = re.findall(r"-(\d{10})", path.stem)
    if len(groups) < 2:
        raise ValueError(path.name)
    return int(groups[-2]), int(groups[-1])


def initialise_output(
    path: Path,
    reference,
    feature_set: str,
    block_size: int,
    overwrite: bool = False,
) -> bool:
    if path.exists() and not overwrite:
        with rasterio.open(path) as existing:
            verify_alignment(reference, existing)
            if existing.count != 2 or existing.nodata != NODATA:
                raise ValueError(f"Existing output is incompatible: {path}")
        print(f"Resuming existing output: {path.name}", flush=True)
        return False
    with rasterio.open(path, "w", **output_profile(reference)) as dst:
        for window in windows(reference.width, reference.height, block_size):
            shape = (int(window.height), int(window.width))
            dst.write(np.full(shape, NODATA, dtype="float32"), 1, window=window)
            dst.write(np.full(shape, NODATA, dtype="float32"), 2, window=window)
        dst.set_band_description(1, "stacked_susceptibility_score")
        dst.set_band_description(2, "base_model_disagreement_sd")
        dst.update_tags(
            analysis="manuscript_v3",
            baseline_year="2018",
            feature_set=feature_set,
            model="20 km buffered nested spatial-CV stacked ensemble",
            score_interpretation="case-control susceptibility score; not population occurrence probability",
            uncertainty="standard deviation across six base-learner scores",
            inside_boundary_nodata="none by design; missing predictors are handled by fitted training imputers",
        )
    return True


def window_is_complete(dst, window: Window, submask: np.ndarray) -> bool:
    score = dst.read(1, window=window)
    return bool(np.all(np.isfinite(score[submask]) & (score[submask] != NODATA)))


def apply_conventional(
    reference,
    mask: np.ndarray,
    base,
    local,
    terrain,
    lithology,
    block_size: int,
    overwrite: bool,
) -> None:
    key = "conventional"
    bundle = joblib.load(MODEL_DIR / f"nested_spatial_stack_{key}.joblib")
    path = OUTPUTS[key]
    initialise_output(path, reference, bundle["feature_set_name"], block_size, overwrite)
    with rasterio.open(path, "r+") as dst:
        for number, window in enumerate(
            windows(reference.width, reference.height, block_size), start=1
        ):
            submask = mask[
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ]
            if not submask.any():
                continue
            if window_is_complete(dst, window, submask):
                continue
            frame = conventional_frame(base, local, terrain, lithology, window, submask)
            score, disagreement = predict_bundle(bundle, frame)
            out_score = np.full(submask.shape, NODATA, dtype="float32")
            out_uncertainty = np.full(submask.shape, NODATA, dtype="float32")
            out_score[submask], out_uncertainty[submask] = score, disagreement
            dst.write(out_score, 1, window=window)
            dst.write(out_uncertainty, 2, window=window)
            if number % 200 == 0:
                print(f"Conventional window {number}", flush=True)


def apply_tiled(
    key: str,
    reference,
    mask: np.ndarray,
    base,
    local,
    terrain,
    lithology,
    block_size: int,
    overwrite: bool,
) -> None:
    bundle = joblib.load(MODEL_DIR / f"nested_spatial_stack_{key}.joblib")
    path = OUTPUTS[key]
    initialise_output(path, reference, bundle["feature_set_name"], block_size, overwrite)
    tiles = sorted(AE_DIR.glob("*.tif"))
    with rasterio.open(path, "r+") as dst:
        for tile_path in tiles:
            row_offset, col_offset = tile_offsets(tile_path)
            with rasterio.open(tile_path) as tile:
                for number, tile_window in enumerate(
                    windows(tile.width, tile.height, block_size), start=1
                ):
                    global_window = Window(
                        col_offset + tile_window.col_off,
                        row_offset + tile_window.row_off,
                        tile_window.width,
                        tile_window.height,
                    )
                    submask = mask[
                        int(global_window.row_off) : int(global_window.row_off + global_window.height),
                        int(global_window.col_off) : int(global_window.col_off + global_window.width),
                    ]
                    if not submask.any():
                        continue
                    if window_is_complete(dst, global_window, submask):
                        continue
                    alpha = alpha_frame(tile, tile_window, submask)
                    if key == "alphaearth_embeddings":
                        frame = alpha
                    else:
                        conventional = conventional_frame(
                            base, local, terrain, lithology, global_window, submask
                        )
                        frame = pd.concat(
                            [conventional.reset_index(drop=True), alpha.reset_index(drop=True)], axis=1
                        )
                    score, disagreement = predict_bundle(bundle, frame)
                    out_score = np.full(submask.shape, NODATA, dtype="float32")
                    out_uncertainty = np.full(submask.shape, NODATA, dtype="float32")
                    out_score[submask], out_uncertainty[submask] = score, disagreement
                    dst.write(out_score, 1, window=global_window)
                    dst.write(out_uncertainty, 2, window=global_window)
                    if number % 100 == 0:
                        print(f"{bundle['feature_set_name']} {tile_path.name}: window {number}", flush=True)


def raster_qa(path: Path, mask: np.ndarray, block_size: int) -> dict:
    values, uncertainty = [], []
    inside_nodata = 0
    with rasterio.open(path) as src:
        for window in windows(src.width, src.height, block_size):
            submask = mask[
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ]
            if not submask.any():
                continue
            score = src.read(1, window=window)
            disagreement = src.read(2, window=window)
            inside_nodata += int(np.sum((score == NODATA) & submask))
            valid_score = score[submask & np.isfinite(score) & (score != NODATA)]
            valid_uncertainty = disagreement[
                submask & np.isfinite(disagreement) & (disagreement != NODATA)
            ]
            if len(valid_score):
                values.append(valid_score)
                uncertainty.append(valid_uncertainty)
        result = {
            "path": str(path),
            "file_size_gb": path.stat().st_size / 1024**3,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "resolution": list(src.res),
            "inside_boundary_pixels": int(mask.sum()),
            "inside_boundary_nodata_pixels": inside_nodata,
        }
        if not values:
            result.update(
                score_min=None,
                score_p10=None,
                score_median=None,
                score_mean=None,
                score_p90=None,
                score_max=None,
                mean_base_model_disagreement=None,
                p90_base_model_disagreement=None,
            )
            return result
        score = np.concatenate(values)
        disagreement = np.concatenate(uncertainty)
        result.update(
            score_min=float(score.min()),
            score_p10=float(np.quantile(score, 0.10)),
            score_median=float(np.median(score)),
            score_mean=float(score.mean()),
            score_p90=float(np.quantile(score, 0.90)),
            score_max=float(score.max()),
            mean_base_model_disagreement=float(disagreement.mean()),
            p90_base_model_disagreement=float(np.quantile(disagreement, 0.90)),
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        choices=["conventional", "alphaearth_embeddings", "conventional_alphaearth_embeddings", "all"],
        default="all",
        help="Raster branch to run. Existing compatible outputs are resumed window by window.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Skip inference and create QA tables for every output that exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(BASE) as base,
        rasterio.open(LOCAL) as local,
        rasterio.open(TERRAIN) as terrain,
        rasterio.open(LITHOLOGY) as lithology,
    ):
        verify_alignment(base, local, terrain, lithology)
        mask = study_mask(base)
        print(f"Official boundary pixels: {int(mask.sum()):,}", flush=True)
        if not args.qa_only:
            selected = set(OUTPUTS) if args.feature_set == "all" else {args.feature_set}
            if "conventional" in selected:
                apply_conventional(
                    base,
                    mask,
                    base,
                    local,
                    terrain,
                    lithology,
                    args.block_size,
                    args.overwrite,
                )
            for key in ["alphaearth_embeddings", "conventional_alphaearth_embeddings"]:
                if key in selected:
                    apply_tiled(
                        key,
                        base,
                        mask,
                        base,
                        local,
                        terrain,
                        lithology,
                        args.block_size,
                        args.overwrite,
                    )
    existing_outputs = [path for path in OUTPUTS.values() if path.exists()]
    qa = [raster_qa(path, mask, args.block_size) for path in existing_outputs]
    pd.DataFrame(qa).to_csv(OUT_DIR / "baseline_susceptibility_score_raster_qa.csv", index=False)
    (OUT_DIR / "baseline_susceptibility_score_raster_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(qa).to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
