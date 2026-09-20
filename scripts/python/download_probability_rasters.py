"""Download completed GEE probability map assets as local GeoTIFFs."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import ee


PROJECT = "ee-mohibullah141300"
OUT_DIR = Path(r"D:\DING PROJECT\04_maps\rasters_2018_probability")
STATUS = OUT_DIR / "download_status.json"

ASSETS = {
    "cpec_2018_conventional_reduced_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_conventional_reduced_probability_250m",
    "cpec_2018_alphaearth_only_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_alphaearth_only_probability_250m",
    "cpec_2018_fused_reduced_probability_250m": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_fused_reduced_probability_250m",
}


def main() -> None:
    ee.Initialize(project=PROJECT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for name, asset in ASSETS.items():
        image = ee.Image(asset)
        url = image.getDownloadURL(
            {
                "name": name,
                "scale": 250,
                "crs": "EPSG:4326",
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
        )
        zip_path = OUT_DIR / f"{name}.zip"
        tif_path = OUT_DIR / f"{name}.tif"
        print(f"Downloading {name}")
        urlretrieve(url, zip_path)
        if zip_path.suffix.lower() == ".zip":
            with ZipFile(zip_path) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))]
                if not members:
                    raise RuntimeError(f"No GeoTIFF found in {zip_path}")
                extracted = zf.extract(members[0], OUT_DIR)
                Path(extracted).replace(tif_path)
        records.append({"name": name, "asset": asset, "zip": str(zip_path), "tif": str(tif_path), "bytes": tif_path.stat().st_size})
    STATUS.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
