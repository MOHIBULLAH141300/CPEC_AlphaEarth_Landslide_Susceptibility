"""QA downloaded 2018 probability rasters against the official study boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import shapefile


PROJECT_ROOT = Path(r"D:\DING PROJECT")
RASTER_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability"
BOUNDARY = Path(r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp")
OUT_DIR = PROJECT_ROOT / "04_maps" / "rasters_2018_probability_qa"
REPORT = PROJECT_ROOT / "05_reports" / "probability_raster_qa_2018.md"

DISPLAY_NAMES = {
    "cpec_2018_conventional_probability_250m_no_nodata.tif": "Conventional",
    "cpec_2018_alphaearth_embeddings_probability_250m_no_nodata.tif": "AlphaEarth Embeddings",
    "cpec_2018_conventional_alphaearth_embeddings_probability_250m_no_nodata.tif": "Conventional + AlphaEarth Embeddings",
    "cpec_2018_alphaearth_only_probability_250m_no_nodata.tif": "AlphaEarth Embeddings",
}


def read_boundary_geometries() -> list[dict]:
    reader = shapefile.Reader(str(BOUNDARY))
    return [shape.__geo_interface__ for shape in reader.shapes()]


def qa_raster(path: Path, boundary_geoms: list[dict]) -> dict[str, object]:
    with rasterio.open(path) as src:
        bnd = [transform_geom("EPSG:32642", src.crs, geom) for geom in boundary_geoms]
        inside = rasterize(
            [(geom, 1) for geom in bnd if geom is not None],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        data = src.read(1, masked=False)
        nodata = src.nodata
        invalid = ~np.isfinite(data)
        if nodata is not None and np.isfinite(nodata):
            invalid |= data == nodata
        inside_count = int(inside.sum())
        inside_invalid = invalid & inside
        valid_inside = data[inside & ~invalid]
        outside_valid_count = int((~inside & ~invalid).sum())
        return {
            "file_name": path.name,
            "display_name": DISPLAY_NAMES.get(path.name, path.stem),
            "file_size_mb": path.stat().st_size / (1024 * 1024),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "pixel_width": float(src.transform.a),
            "pixel_height": float(abs(src.transform.e)),
            "bounds_left": float(src.bounds.left),
            "bounds_bottom": float(src.bounds.bottom),
            "bounds_right": float(src.bounds.right),
            "bounds_top": float(src.bounds.top),
            "nodata_value": nodata,
            "inside_boundary_pixels": inside_count,
            "inside_nodata_pixels": int(inside_invalid.sum()),
            "inside_nodata_percent": float(inside_invalid.sum() / inside_count * 100) if inside_count else float("nan"),
            "outside_valid_pixels": outside_valid_count,
            "min_probability_inside": float(np.min(valid_inside)) if valid_inside.size else float("nan"),
            "max_probability_inside": float(np.max(valid_inside)) if valid_inside.size else float("nan"),
            "mean_probability_inside": float(np.mean(valid_inside)) if valid_inside.size else float("nan"),
            "probability_values_in_0_1": bool(valid_inside.size and np.min(valid_inside) >= -1e-6 and np.max(valid_inside) <= 1 + 1e-6),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary_geoms = read_boundary_geometries()
    raster_paths = sorted(RASTER_DIR.glob("*.tif"))
    clean_alpha = RASTER_DIR / "cpec_2018_alphaearth_embeddings_probability_250m_no_nodata.tif"
    old_alpha = RASTER_DIR / "cpec_2018_alphaearth_only_probability_250m_no_nodata.tif"
    if clean_alpha.exists() and old_alpha.exists():
        raster_paths = [path for path in raster_paths if path != old_alpha]
    rows = [qa_raster(path, boundary_geoms) for path in raster_paths]
    qa = pd.DataFrame(rows)
    csv_path = OUT_DIR / "probability_raster_qa_2018.csv"
    json_path = OUT_DIR / "probability_raster_qa_2018.json"
    qa.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# 2018 Probability Raster QA",
        "",
        f"- Raster folder: `{RASTER_DIR}`",
        f"- Boundary: `{BOUNDARY}`",
        "",
        "| Raster | Size MB | CRS | Width x height | Inside no-data pixels | Inside no-data % | Min | Max | Mean | Values 0-1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in qa.iterrows():
        lines.append(
            f"| {row['display_name']} | {row['file_size_mb']:.1f} | {row['crs']} | "
            f"{int(row['width'])} x {int(row['height'])} | {int(row['inside_nodata_pixels'])} | "
            f"{row['inside_nodata_percent']:.4f} | {row['min_probability_inside']:.4f} | "
            f"{row['max_probability_inside']:.4f} | {row['mean_probability_inside']:.4f} | "
            f"{row['probability_values_in_0_1']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Correct rasters should have probability values between 0 and 1.",
            "- Correct no-nodata rasters should have zero no-data pixels inside the official study boundary.",
            "",
            "## Saved Outputs",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(qa.to_string(index=False))


if __name__ == "__main__":
    main()
