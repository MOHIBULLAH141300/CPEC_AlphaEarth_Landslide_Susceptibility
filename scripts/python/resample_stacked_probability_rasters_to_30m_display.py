"""Resample final stacked probability rasters to an approximate 30 m display grid.

These are cartographic display rasters, not true 30 m model predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


PROJECT_ROOT = Path(r"D:\DING PROJECT")
IN_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble_30m_display"
QA_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_stacked_ensemble_30m_display_qa"
REPORT = PROJECT_ROOT / "05_reports" / "stacked_probability_rasters_30m_display_2026-05-03.md"

INPUTS = {
    "Conventional": (
        "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
        "cpec_2018_conventional_stacked_ensemble_probability_30m_display.tif",
    ),
    "AlphaEarth Embeddings": (
        "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_30m_display.tif",
    ),
    "Conventional + AlphaEarth Embeddings": (
        "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_30m_display.tif",
    ),
}


def resample_one(label: str, src_name: str, out_name: str) -> dict[str, object]:
    src_path = IN_DIR / src_name
    out_path = OUT_DIR / out_name
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    with rasterio.open(src_path) as src:
        scale = 250.0 / 30.0
        dst_width = int(round(src.width * scale))
        dst_height = int(round(src.height * scale))
        dst_transform = src.transform * src.transform.scale(src.width / dst_width, src.height / dst_height)
        profile = src.profile.copy()
        profile.update(
            {
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",
                "nodata": None,
            }
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
            dst.set_band_description(1, "probability_30m_display_resample")
            dst.update_tags(
                year="2018",
                feature_set=label,
                model="Spatial-CV Stacked Ensemble",
                source_resolution="250m",
                display_resolution="approximately_30m",
                resampling="bilinear",
                caveat="display_resample_not_true_30m_model_prediction",
            )
    with rasterio.open(out_path) as dst:
        return {
            "feature_set": label,
            "source": str(src_path),
            "output": str(out_path),
            "size_gb": round(out_path.stat().st_size / 1024**3, 3),
            "height": dst.height,
            "width": dst.width,
            "x_res": dst.res[0],
            "y_res": dst.res[1],
            "crs": str(dst.crs),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, (src_name, out_name) in INPUTS.items():
        print(f"START {label}", flush=True)
        rows.append(resample_one(label, src_name, out_name))
        print(f"DONE {rows[-1]['output']}", flush=True)
    qa = pd.DataFrame(rows)
    qa.to_csv(QA_DIR / "stacked_probability_rasters_30m_display_qa.csv", index=False)
    lines = [
        "# 30 m Display Resampling Of Stacked Probability Rasters",
        "",
        "These rasters are bilinear resamples of the final 250 m Spatial-CV Stacked Ensemble probability rasters. They are intended for cartographic display only, not as true 30 m model predictions.",
        "",
        "| Feature set | Output | Size GB | Width | Height |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['feature_set']} | `{row['output']}` | {row['size_gb']:.3f} | {row['width']} | {row['height']} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
