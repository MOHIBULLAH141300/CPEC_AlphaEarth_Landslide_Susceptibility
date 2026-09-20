"""Export completed GEE probability rasters to Google Drive as GeoTIFFs."""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
FOLDER = "GEE_CPEC_LSM_RASTERS"
STATUS = Path(r"D:\DING PROJECT\04_maps\drive_probability_raster_export_plan.json")

ASSETS = {
    "cpec_2018_conventional_reduced_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_conventional_reduced_probability_250m",
    "cpec_2018_alphaearth_only_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_alphaearth_only_probability_250m",
    "cpec_2018_fused_reduced_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_fused_reduced_probability_250m",
}


def main() -> None:
    ee.Initialize(project=PROJECT)
    plan = []
    for name, asset in ASSETS.items():
        image = ee.Image(asset)
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name,
            folder=FOLDER,
            fileNamePrefix=name,
            scale=250,
            crs="EPSG:4326",
            region=image.geometry(),
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        plan.append({"name": name, "asset": asset, "drive_folder": FOLDER, "task_id": task.id})
    STATUS.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
