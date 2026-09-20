"""Export the GEE-available 2018 predictor stacks needed for stacked rasters.

These are not complete v3 stacks. They contain only bands present in the clean
GEE public factor asset. The local full-CPEC v3 factors are generated locally
and merged afterward.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
DRIVE_FOLDER = "GEE_CPEC_LSM_STACKED_ENSEMBLE_GEE_INPUTS_250M"
PLAN = Path(r"D:\DING PROJECT\04_maps\stacked_ensemble_gee_available_stacks_250m_export_plan.json")

GEE_CONVENTIONAL_BASE = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "modis_lc_type1",
]
ALPHA = [f"A{i:02d}" for i in range(64)]


def export_stack(name: str, bands: list[str], study_area: ee.FeatureCollection) -> dict[str, object]:
    stack = (
        ee.Image(FACTORS_2018)
        .select(bands)
        .toFloat()
        .clip(study_area)
        .set(
            {
                "year": 2018,
                "resolution_m": 250,
                "feature_set": name,
                "output_type": "gee_available_predictor_stack_for_local_stacked_ensemble",
                "feature_count": len(bands),
                "factor_asset": FACTORS_2018,
            }
        )
    )
    prefix = f"cpec_2018_{name}_gee_available_predictor_stack_250m"
    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=prefix,
        region=study_area.geometry(),
        scale=250,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return {
        "name": name,
        "drive_folder": DRIVE_FOLDER,
        "file_prefix": prefix,
        "task_id": task.id,
        "resolution_m": 250,
        "band_count": len(bands),
        "bands": bands,
    }


def main() -> None:
    ee.Initialize(project=PROJECT)
    study_area = ee.FeatureCollection(BOUNDARY)
    outputs = {
        "conventional_base": GEE_CONVENTIONAL_BASE,
        "alphaearth_embeddings": ALPHA,
        "conventional_base_alphaearth_embeddings": GEE_CONVENTIONAL_BASE + ALPHA,
    }
    plan = [export_stack(name, bands, study_area) for name, bands in outputs.items()]
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
