"""Export local-deployment predictor stack for the 2018 fused stacked model.

This exports the exact 73 predictors needed by the local
Conventional + AlphaEarth Embeddings stacked ensemble:
- 9 selected conventional predictors
- 64 AlphaEarth embedding bands

The stack is exported at 250 m with predictor masks filled from training-sample
means, matching the no-internal-NoData policy used for the probability rasters.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
DRIVE_FOLDER = "GEE_CPEC_LSM_STACKED_ENSEMBLE_INPUTS_250M"
PLAN = Path(r"D:\DING PROJECT\04_maps\stacked_ensemble_factor_stack_250m_export_plan.json")

CONVENTIONAL_REDUCED = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
]

ALPHA = [f"A{i:02d}" for i in range(64)]
FUSED_BANDS = CONVENTIONAL_REDUCED + ALPHA


def fill_predictor_masks(image: ee.Image, bands: list[str]) -> ee.Image:
    samples = ee.FeatureCollection(SAMPLES)
    filled = []
    for band in bands:
        fill_value = ee.Number(samples.aggregate_mean(band))
        filled.append(image.select(band).unmask(fill_value, False).rename(band))
    return ee.Image.cat(filled)


def main() -> None:
    ee.Initialize(project=PROJECT)
    study_area = ee.FeatureCollection(BOUNDARY)
    raw = ee.Image(FACTORS_2018).select(FUSED_BANDS)
    stack = (
        fill_predictor_masks(raw, FUSED_BANDS)
        .toFloat()
        .clip(study_area)
        .set(
            {
                "year": 2018,
                "resolution_m": 250,
                "output_type": "fused_predictor_stack_for_local_stacked_ensemble",
                "feature_count": len(FUSED_BANDS),
                "sample_asset": SAMPLES,
                "factor_asset": FACTORS_2018,
                "nodata_policy": "predictor masks filled from training sample means using unmask(..., sameFootprint=False)",
            }
        )
    )

    prefix = "cpec_2018_fused_predictor_stack_250m_no_nodata"
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
    plan = {
        "drive_folder": DRIVE_FOLDER,
        "file_prefix": prefix,
        "task_id": task.id,
        "resolution_m": 250,
        "band_count": len(FUSED_BANDS),
        "bands": FUSED_BANDS,
        "next_local_folder": r"D:\DING PROJECT\04_maps\stacked_ensemble_inputs_250m",
        "next_script": r"D:\DING PROJECT\06_scripts\python\apply_2018_stacked_ensemble_to_factor_stack.py",
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
