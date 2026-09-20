"""Create a manifest for the 30 m Google Drive raster tiles.

The Drive connector can list private GeoTIFFs, but this environment may not be
able to download private TIFF bytes directly. Save the listing from Drive into a
CSV manifest so downloaded local tiles can be checked before merging.
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(r"D:\DING PROJECT")
OUT = PROJECT_ROOT / "04_maps" / "drive_probability_tiles_30m_manifest.csv"

# Generated from the visible Drive listing on 2026-05-02.
TILES = [
    # Conventional, visible tiles.
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000065536-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000057344-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000057344-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000057344-0000016384.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000057344-0000008192.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000057344-0000000000.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000016384.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000008192.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000049152-0000000000.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000016384.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000008192.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000040960-0000000000.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000032768-0000049152.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000032768-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000032768-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000032768-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000032768-0000016384.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000024576-0000057344.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000024576-0000049152.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000024576-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000024576-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000024576-0000024576.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000016384-0000057344.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000016384-0000049152.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000016384-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000016384-0000032768.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000008192-0000065536.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000008192-0000057344.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000008192-0000049152.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000008192-0000040960.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000000000-0000065536.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000000000-0000057344.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000000000-0000049152.tif"),
    ("cpec_2018_conventional_reduced_probability_30m", "cpec_2018_conventional_reduced_probability_30m-0000000000-0000040960.tif"),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_prefix", "file_name"])
        writer.writerows(TILES)
    print(f"Saved {len(TILES)} visible tile records to {OUT}")


if __name__ == "__main__":
    main()

