"""Prepare the next annual temporal batch.

This script does three storage-safe things:
1. Places the already verified 2018 stacked probability rasters into the
   annual 2018 folder using hardlinks where possible.
2. Computes per-year probability QA from actual files on disk.
3. Writes a fresh status table for the 2017-2024 temporal-transfer workspace.

It does not delete or overwrite local raw data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio
from rasterio import features
from rasterio.windows import Window

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT_ROOT = Path(r"D:\DING PROJECT")
ANNUAL = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
CLEAN_2018 = (
    PROJECT_ROOT
    / "00_READ_ME_FIRST_2018_V3_RESULTS"
    / "04_probability_maps_250m"
    / "01_primary_stacked_maps"
)
NODATA = -9999.0
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)

FEATURE_SETS = {
    "conventional": "Conventional",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_alphaearth_embeddings": "Conventional + AlphaEarth Embeddings",
}

SOURCE_2018 = {
    "conventional": CLEAN_2018
    / "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
    "alphaearth_embeddings": CLEAN_2018
    / "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
    "conventional_alphaearth_embeddings": CLEAN_2018
    / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
}


def year_dirs(year: int) -> dict[str, Path]:
    root = ANNUAL / str(year)
    return {
        "root": root,
        "raw": root / "00_raw_exports_from_gee",
        "pred": root / "02_probability_maps_250m",
        "qa": root / "05_year_specific_qa",
        "logs": root / "06_processing_logs",
    }


def annual_probability_path(year: int, key: str) -> Path:
    return (
        year_dirs(year)["pred"]
        / f"cpec_{year}_{key}_stacked_ensemble_probability_250m.tif"
    )


def iter_windows(width: int, height: int, block_size: int = 1024):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


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


def write_boundary_masked_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with rasterio.open(src) as ref:
        mask = study_mask(ref)
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
        with rasterio.open(tmp, "w", **profile) as out:
            for win in iter_windows(ref.width, ref.height):
                arr = ref.read(1, window=win, out_dtype="float32")
                m = mask[
                    int(win.row_off) : int(win.row_off + win.height),
                    int(win.col_off) : int(win.col_off + win.width),
                ]
                arr[~m] = NODATA
                out.write(arr, 1, window=win)
            out.set_band_description(1, ref.descriptions[0] or "probability")
            out.update_tags(**ref.tags())
    os.replace(tmp, dst)
    return "boundary_masked_copy"


def raster_stats(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024**2, 2) if path.exists() else 0.0,
            "valid_pixels": 0,
            "nodata_pixels": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "width": None,
            "height": None,
            "crs": None,
        }
    with rasterio.open(path) as src:
        arr = src.read(1, masked=False)
        nodata = src.nodata if src.nodata is not None else NODATA
        valid = arr[(arr != nodata) & np.isfinite(arr)]
        return {
            "path": str(path),
            "exists": True,
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "valid_pixels": int(valid.size),
            "nodata_pixels": int(np.sum(arr == nodata)),
            "min": float(np.min(valid)) if valid.size else None,
            "max": float(np.max(valid)) if valid.size else None,
            "mean": float(np.mean(valid)) if valid.size else None,
            "std": float(np.std(valid)) if valid.size else None,
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
        }


def write_year_qa(year: int, rows: list[dict[str, object]]) -> None:
    dirs = year_dirs(year)
    dirs["qa"].mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(dirs["qa"] / f"cpec_{year}_annual_probability_qa.csv", index=False)
    (dirs["qa"] / f"cpec_{year}_annual_probability_qa.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Annual Probability Raster QA: {year}",
        "",
        "Model strategy: fixed 2018 V3 Spatial-CV Stacked Ensemble applied to annual predictors.",
        "",
        "| Feature set | Valid pixels | Nodata pixels | Min | Max | Mean | Std | File |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        def fmt(v):
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

        lines.append(
            f"| {row['feature_set']} | {row['valid_pixels']} | {row['nodata_pixels']} | "
            f"{fmt(row['min'])} | {fmt(row['max'])} | {fmt(row['mean'])} | {fmt(row['std'])} | "
            f"`{row['path']}` |"
        )
    (dirs["qa"] / f"cpec_{year}_annual_probability_qa.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def install_2018_baseline() -> list[dict[str, object]]:
    actions = []
    for key, src in SOURCE_2018.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = annual_probability_path(2018, key)
        action = write_boundary_masked_copy(src, dst)
        actions.append(
            {
                "feature_set": FEATURE_SETS[key],
                "source": str(src),
                "destination": str(dst),
                "action": action,
            }
        )
    return actions


def raw_input_status(year: int) -> tuple[int, int, str]:
    raw = year_dirs(year)["raw"]
    base = list(raw.glob(f"cpec_{year}_annual_dynamic_base_predictors_250m*.tif"))
    alpha = list(raw.glob(f"cpec_{year}_alphaearth_embeddings_250m*.tif"))
    if len(base) == 1 and len(alpha) >= 1:
        status = "complete"
    elif len(base) or len(alpha):
        status = "partial"
    else:
        status = "missing"
    return len(base), len(alpha), status


def probability_status(year: int) -> tuple[int, str]:
    existing = [
        annual_probability_path(year, key)
        for key in FEATURE_SETS
        if annual_probability_path(year, key).exists()
        and annual_probability_path(year, key).stat().st_size > 0
    ]
    if len(existing) == 3:
        status = "complete"
    elif existing:
        status = "partial"
    else:
        status = "missing"
    return len(existing), status


def refresh_status() -> None:
    summary_rows = []
    for year in range(2017, 2025):
        base_count, alpha_count, raw_status = raw_input_status(year)
        probability_count, prob_status = probability_status(year)
        qa_rows = []
        for key, label in FEATURE_SETS.items():
            p = annual_probability_path(year, key)
            st = raster_stats(p)
            st["year"] = year
            st["feature_set"] = label
            st["feature_key"] = key
            qa_rows.append(st)
        if prob_status != "missing":
            write_year_qa(year, qa_rows)
        summary_rows.append(
            {
                "year": year,
                "base_predictor_files": base_count,
                "alphaearth_tiles": alpha_count,
                "raw_input_status": raw_status,
                "probability_rasters": probability_count,
                "probability_status": prob_status,
            }
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(ANNUAL / "annual_dynamic_status_latest.csv", index=False)
    lines = [
        "# Annual Dynamic Susceptibility Status",
        "",
        "Updated from actual local files.",
        "",
        "| Year | Base predictor files | AlphaEarth tiles | Raw input status | Probability rasters | Probability status |",
        "| ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['year']} | {row['base_predictor_files']} | {row['alphaearth_tiles']} | "
            f"{row['raw_input_status']} | {row['probability_rasters']} | {row['probability_status']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Complete probability years are ready for temporal comparison.",
            "- Years with complete raw inputs but missing probability rasters are ready for one-year model application.",
            "- 2018 is the validated baseline year; its verified 2018 stacked rasters are boundary-masked and copied into the annual folder.",
            "",
            "Processing rule: run only one year at a time and keep all raw, probability, QA, and figure outputs inside that year folder.",
        ]
    )
    (ANNUAL / "ANNUAL_DYNAMIC_STATUS_LATEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    actions = install_2018_baseline()
    (ANNUAL / "2018" / "06_processing_logs").mkdir(parents=True, exist_ok=True)
    (ANNUAL / "2018" / "06_processing_logs" / "install_2018_baseline_links.json").write_text(
        json.dumps(actions, indent=2), encoding="utf-8"
    )
    refresh_status()
    print(json.dumps(actions, indent=2))
    print(ANNUAL / "ANNUAL_DYNAMIC_STATUS_LATEST.md")


if __name__ == "__main__":
    main()
