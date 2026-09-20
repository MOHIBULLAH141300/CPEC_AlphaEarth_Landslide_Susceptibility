"""Create compact temporal susceptibility products from annual fused maps.

This implements the agreed temporal-transfer strategy:
- use fixed 2018 V3 Spatial-CV Stacked Ensemble outputs,
- focus the main temporal products on the fused feature set,
- summarize 2017-2024 as mean, variability, trend, persistent hotspots, and
  year-to-year change maps rather than presenting eight independent models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(r"D:\DING PROJECT")
ANNUAL = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
OUT = ANNUAL / "00_multi_year_summary_outputs"
CHANGE_OUT = OUT / "01_year_to_year_change_maps"
YEARS = list(range(2017, 2025))
NODATA = -9999.0
YEAR_NODATA = 0

# Anchored to the validated 2018 fused probability P80 threshold from the
# existing high-impact 2018 output package.
FUSED_P80_THRESHOLD = 0.12633591890335083
PERSISTENT_HIGH_COUNT = 6


def fused_path(year: int) -> Path:
    return (
        ANNUAL
        / str(year)
        / "02_probability_maps_250m"
        / f"cpec_{year}_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m.tif"
    )


def iter_windows(width: int, height: int, block_size: int = 512):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def float_profile(ref: rasterio.DatasetReader) -> dict:
    profile = ref.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def int_profile(ref: rasterio.DatasetReader, dtype: str, nodata: int) -> dict:
    profile = ref.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def open_outputs(ref: rasterio.DatasetReader) -> dict[str, rasterio.io.DatasetWriter]:
    OUT.mkdir(parents=True, exist_ok=True)
    CHANGE_OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mean": rasterio.open(
            OUT / "cpec_2017_2024_fused_mean_probability_250m.tif",
            "w",
            **float_profile(ref),
        ),
        "std": rasterio.open(
            OUT / "cpec_2017_2024_fused_temporal_std_probability_250m.tif",
            "w",
            **float_profile(ref),
        ),
        "range": rasterio.open(
            OUT / "cpec_2017_2024_fused_temporal_range_probability_250m.tif",
            "w",
            **float_profile(ref),
        ),
        "theilsen": rasterio.open(
            OUT / "cpec_2017_2024_fused_theilsen_trend_slope_per_year_250m.tif",
            "w",
            **float_profile(ref),
        ),
        "high_count": rasterio.open(
            OUT / "cpec_2017_2024_fused_high_probability_year_count_p80_250m.tif",
            "w",
            **int_profile(ref, "uint8", 255),
        ),
        "high_fraction": rasterio.open(
            OUT / "cpec_2017_2024_fused_high_probability_fraction_p80_250m.tif",
            "w",
            **float_profile(ref),
        ),
        "persistent": rasterio.open(
            OUT / "cpec_2017_2024_fused_persistent_high_probability_mask_p80_count_ge6_250m.tif",
            "w",
            **int_profile(ref, "uint8", 255),
        ),
        "year_max": rasterio.open(
            OUT / "cpec_2017_2024_fused_year_of_max_probability_250m.tif",
            "w",
            **int_profile(ref, "uint16", YEAR_NODATA),
        ),
        "year_min": rasterio.open(
            OUT / "cpec_2017_2024_fused_year_of_min_probability_250m.tif",
            "w",
            **int_profile(ref, "uint16", YEAR_NODATA),
        ),
    }
    for key, dst in outputs.items():
        dst.update_tags(
            temporal_strategy="fixed_2018_v3_model_applied_to_annual_predictors",
            feature_set="Conventional + AlphaEarth Embeddings",
            model="Spatial-CV Stacked Ensemble",
            years="2017-2024",
            fused_p80_threshold=str(FUSED_P80_THRESHOLD),
            persistent_high_count=str(PERSISTENT_HIGH_COUNT),
            product=key,
        )
    return outputs


def close_outputs(outputs: dict[str, rasterio.io.DatasetWriter]) -> None:
    for dst in outputs.values():
        dst.close()


def theilsen_slope(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    slopes = []
    year_arr = np.array(YEARS, dtype="float32")
    for i in range(len(YEARS) - 1):
        for j in range(i + 1, len(YEARS)):
            slopes.append((stack[j] - stack[i]) / (year_arr[j] - year_arr[i]))
    slope_stack = np.stack(slopes, axis=0)
    slope_stack[:, ~valid] = np.nan
    out = np.nanmedian(slope_stack, axis=0).astype("float32")
    out[~valid] = NODATA
    return out


def raster_stats(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=False)
        nodata = src.nodata
        valid = arr[(arr != nodata) & np.isfinite(arr)]
        return {
            "file": str(path),
            "valid_pixels": int(valid.size),
            "nodata_pixels": int(np.sum(arr == nodata)),
            "min": float(np.min(valid)) if valid.size else None,
            "p50": float(np.percentile(valid, 50)) if valid.size else None,
            "p80": float(np.percentile(valid, 80)) if valid.size else None,
            "p90": float(np.percentile(valid, 90)) if valid.size else None,
            "max": float(np.max(valid)) if valid.size else None,
            "mean": float(np.mean(valid)) if valid.size else None,
            "std": float(np.std(valid)) if valid.size else None,
        }


def main() -> None:
    paths = [fused_path(year) for year in YEARS]
    missing = [p for p in paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing fused annual rasters:\n" + "\n".join(map(str, missing)))

    srcs = [rasterio.open(path) for path in paths]
    try:
        ref = srcs[0]
        for path, src in zip(paths[1:], srcs[1:]):
            if src.width != ref.width or src.height != ref.height:
                raise ValueError(f"Shape mismatch: {path}")
            if src.transform != ref.transform or src.crs != ref.crs:
                raise ValueError(f"Grid mismatch: {path}")

        outputs = open_outputs(ref)
        change_outputs = {
            (prev, curr): rasterio.open(
                CHANGE_OUT / f"cpec_{curr}_minus_{prev}_fused_probability_change_250m.tif",
                "w",
                **float_profile(ref),
            )
            for prev, curr in zip(YEARS[:-1], YEARS[1:])
        }
        for (prev, curr), dst in change_outputs.items():
            dst.update_tags(
                product="year_to_year_change",
                previous_year=str(prev),
                current_year=str(curr),
                feature_set="Conventional + AlphaEarth Embeddings",
                model="Spatial-CV Stacked Ensemble",
            )

        for win_index, win in enumerate(iter_windows(ref.width, ref.height), start=1):
            annual = np.stack([src.read(1, window=win, out_dtype="float32") for src in srcs], axis=0)
            valid = np.all((annual != NODATA) & np.isfinite(annual), axis=0)

            mean = np.full(valid.shape, NODATA, dtype="float32")
            std = np.full(valid.shape, NODATA, dtype="float32")
            rng = np.full(valid.shape, NODATA, dtype="float32")
            high_fraction = np.full(valid.shape, NODATA, dtype="float32")
            high_count = np.full(valid.shape, 255, dtype="uint8")
            persistent = np.full(valid.shape, 255, dtype="uint8")
            year_max = np.full(valid.shape, YEAR_NODATA, dtype="uint16")
            year_min = np.full(valid.shape, YEAR_NODATA, dtype="uint16")

            if np.any(valid):
                masked = np.where(valid, annual, np.nan)
                mean[valid] = np.nanmean(masked, axis=0)[valid]
                std[valid] = np.nanstd(masked, axis=0)[valid]
                rng[valid] = (np.nanmax(masked, axis=0) - np.nanmin(masked, axis=0))[valid]
                hc = np.sum(masked >= FUSED_P80_THRESHOLD, axis=0)
                high_count[valid] = hc[valid].astype("uint8")
                high_fraction[valid] = (hc / len(YEARS))[valid].astype("float32")
                persistent[valid] = (hc[valid] >= PERSISTENT_HIGH_COUNT).astype("uint8")
                year_max[valid] = np.array(YEARS, dtype="uint16")[np.nanargmax(masked[:, valid], axis=0)]
                year_min[valid] = np.array(YEARS, dtype="uint16")[np.nanargmin(masked[:, valid], axis=0)]

            outputs["mean"].write(mean, 1, window=win)
            outputs["std"].write(std, 1, window=win)
            outputs["range"].write(rng, 1, window=win)
            outputs["theilsen"].write(theilsen_slope(annual, valid), 1, window=win)
            outputs["high_count"].write(high_count, 1, window=win)
            outputs["high_fraction"].write(high_fraction, 1, window=win)
            outputs["persistent"].write(persistent, 1, window=win)
            outputs["year_max"].write(year_max, 1, window=win)
            outputs["year_min"].write(year_min, 1, window=win)

            for prev_idx, (prev, curr) in enumerate(zip(YEARS[:-1], YEARS[1:])):
                change = np.full(valid.shape, NODATA, dtype="float32")
                if np.any(valid):
                    change[valid] = (annual[prev_idx + 1] - annual[prev_idx])[valid]
                change_outputs[(prev, curr)].write(change, 1, window=win)

            if win_index % 50 == 0:
                print(f"Processed {win_index} temporal windows", flush=True)

        close_outputs(outputs)
        for dst in change_outputs.values():
            dst.close()
    finally:
        for src in srcs:
            src.close()

    annual_rows = []
    for year in YEARS:
        row = raster_stats(fused_path(year))
        row["year"] = year
        annual_rows.append(row)
    pd.DataFrame(annual_rows).to_csv(OUT / "annual_fused_probability_summary_2017_2024.csv", index=False)

    diagnostic_paths = list(OUT.glob("cpec_2017_2024_fused_*.tif")) + list(CHANGE_OUT.glob("*.tif"))
    diagnostics = [raster_stats(path) for path in diagnostic_paths]
    pd.DataFrame(diagnostics).to_csv(OUT / "temporal_summary_raster_diagnostics.csv", index=False)

    report = {
        "years": YEARS,
        "feature_set": "Conventional + AlphaEarth Embeddings",
        "model": "Spatial-CV Stacked Ensemble",
        "temporal_strategy": "fixed_2018_v3_model_applied_to_annual_predictors",
        "fused_p80_threshold_from_2018": FUSED_P80_THRESHOLD,
        "persistent_high_definition": f"probability >= P80 in at least {PERSISTENT_HIGH_COUNT} of 8 years",
        "outputs_folder": str(OUT),
        "change_maps_folder": str(CHANGE_OUT),
    }
    (OUT / "temporal_summary_methods_and_thresholds.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (OUT / "README_TEMPORAL_SUMMARY_OUTPUTS.md").write_text(
        "# Temporal Summary Outputs 2017-2024\n\n"
        "These products summarize annual fused susceptibility maps from the fixed 2018 V3 "
        "Spatial-CV Stacked Ensemble. They should be described as temporal-transfer / "
        "dynamic-covariate stress-test outputs, not as eight independently validated annual models.\n\n"
        f"High-probability threshold: 2018 fused P80 = `{FUSED_P80_THRESHOLD:.6f}`.\n\n"
        f"Persistent high susceptibility: probability >= P80 in at least {PERSISTENT_HIGH_COUNT} of 8 years.\n\n"
        "Main rasters:\n"
        "- mean probability\n"
        "- temporal standard deviation\n"
        "- temporal range\n"
        "- Theil-Sen trend slope per year\n"
        "- high-probability year count and fraction\n"
        "- persistent high-probability mask\n"
        "- year of maximum and minimum probability\n"
        "- consecutive year-to-year change maps\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
