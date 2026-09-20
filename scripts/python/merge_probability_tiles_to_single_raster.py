"""Merge Google Earth Engine GeoTIFF tile exports into one raster per model.

Use this after downloading the Drive folder `GEE_CPEC_LSM_RASTERS_30M` to a
local folder. Earth Engine split the 30 m CPEC-wide rasters into many tiles
because the study area is large.

Example:
python merge_probability_tiles_to_single_raster.py ^
  --tiles-dir "D:\\DING PROJECT\\04_maps\\drive_tiles_30m" ^
  --out-dir "D:\\DING PROJECT\\04_maps\\probability_rasters_30m_merged"
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import rasterio
from rasterio.merge import merge


PATTERN = re.compile(r"^(?P<prefix>.+?_probability_30m)-\d{10}-\d{10}\.tif$", re.IGNORECASE)

MODEL_LABELS = {
    "cpec_2018_conventional_probability_30m": "Conventional",
    "cpec_2018_alphaearth_embeddings_probability_30m": "AlphaEarth Embeddings",
    "cpec_2018_conventional_alphaearth_embeddings_probability_30m": "Conventional + AlphaEarth Embeddings",
    "cpec_2018_conventional_reduced_probability_30m": "Conventional",
    "cpec_2018_alphaearth_only_probability_30m": "AlphaEarth Embeddings",
    "cpec_2018_fused_reduced_probability_30m": "Conventional + AlphaEarth Embeddings",
}


def group_tiles(tiles_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for tif in sorted(tiles_dir.glob("*.tif")):
        match = PATTERN.match(tif.name)
        if match:
            groups[match.group("prefix")].append(tif)
    return dict(groups)


def merge_group(prefix: str, files: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_merged_single_30m.tif"
    sources = [rasterio.open(path) for path in files]
    try:
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
            nodata=sources[0].nodata,
        )
        merge(sources, dst_path=out_path, dst_kwds=profile)
    finally:
        for src in sources:
            src.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 30 m GEE probability GeoTIFF tiles.")
    parser.add_argument("--tiles-dir", required=True, help="Local folder containing downloaded GEE GeoTIFF tiles.")
    parser.add_argument("--out-dir", default=r"D:\DING PROJECT\04_maps\probability_rasters_30m_merged")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    out_dir = Path(args.out_dir)
    groups = group_tiles(tiles_dir)
    if not groups:
        raise SystemExit(f"No GEE tile files matching '*_probability_30m-##########-##########.tif' found in {tiles_dir}")

    for prefix, files in sorted(groups.items()):
        label = MODEL_LABELS.get(prefix, prefix)
        print(f"Merging {len(files)} tiles for {label}...")
        out_path = merge_group(prefix, files, out_dir)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
