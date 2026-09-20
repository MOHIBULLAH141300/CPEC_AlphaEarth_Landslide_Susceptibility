"""Resample 250 m probability rasters to an approximate 30 m grid for display.

This does not create new 30 m model information; it creates smoother,
publication-friendly display rasters from the downloaded continuous probability
maps. True 30 m stacked probabilities require applying the trained models to
30 m predictor rasters.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


PROJECT_ROOT = Path(r"D:\DING PROJECT")
IN_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability"
OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_30m_resampled_display"
REPORT = PROJECT_ROOT / "05_reports" / "probability_raster_30m_resampled_display_2026-05-03.md"

INPUTS = {
    "cpec_2018_conventional_probability_250m_no_nodata.tif": "cpec_2018_conventional_probability_30m_resampled_display.tif",
    "cpec_2018_alphaearth_embeddings_probability_250m_no_nodata.tif": "cpec_2018_alphaearth_embeddings_probability_30m_resampled_display.tif",
    "cpec_2018_conventional_alphaearth_embeddings_probability_250m_no_nodata.tif": "cpec_2018_conventional_alphaearth_embeddings_probability_30m_resampled_display.tif",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for src_name, out_name in INPUTS.items():
        src_path = IN_DIR / src_name
        out_path = OUT_DIR / out_name
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        with rasterio.open(src_path) as src:
            scale = 250.0 / 30.0
            dst_width = int(round(src.width * scale))
            dst_height = int(round(src.height * scale))
            dst_transform = src.transform * src.transform.scale(
                src.width / dst_width,
                src.height / dst_height,
            )
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
                    "BIGTIFF": "IF_SAFER",
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
                dst.set_band_description(1, "probability_resampled_display")
                dst.update_tags(
                    source_resolution="250m_probability_map",
                    display_resolution="approximately_30m",
                    resampling="bilinear",
                    caveat="display_resample_not_true_30m_model_prediction",
                )
        rows.append((src_name, out_name, dst_width, dst_height))

    lines = [
        "# 30 m Display Resampling Of 2018 Probability Rasters",
        "",
        "These rasters are bilinear resamples of the downloaded 250 m probability maps. They are suitable for smoother cartographic display, but they should be described as resampled display rasters, not true 30 m model predictions.",
        "",
        "For true 30 m stacked-ensemble probability rasters, the trained models must be applied to a complete 30 m predictor stack for each feature set.",
        "",
        "| Source raster | Output raster | Width | Height |",
        "|---|---|---:|---:|",
    ]
    for src_name, out_name, width, height in rows:
        lines.append(f"| {src_name} | {OUT_DIR / out_name} | {width} | {height} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
