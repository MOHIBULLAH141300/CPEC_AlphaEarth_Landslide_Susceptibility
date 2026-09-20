"""Export 250 m 2018 probability maps with no internal no-data pixels.

This replaces the earlier 250 m probability raster export where edge pixels
inside the official study boundary could become masked because one or more
predictor bands had missing values near the boundary. Here every predictor band
is unmasked from training-sample means before classification.

No susceptibility classes are produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
DRIVE_FOLDER = "GEE_CPEC_LSM_RASTERS_250M_NO_NODATA"
PLAN = Path(r"D:\DING PROJECT\04_maps\drive_probability_raster_250m_no_nodata_export_plan.json")

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


def classifier(seed: int) -> ee.Classifier:
    return ee.Classifier.smileGradientTreeBoost(
        numberOfTrees=500,
        shrinkage=0.03,
        samplingRate=0.85,
        maxNodes=32,
        loss="Logistic",
        seed=seed,
    ).setOutputMode("PROBABILITY")


def fill_predictor_masks(image: ee.Image, bands: list[str]) -> ee.Image:
    """Fill missing predictor pixels and expand footprint before clipping.

    The second argument to unmask is `sameFootprint=False`, so fill values can
    extend into clipped edge pixels inside the study area where a source raster
    footprint was slightly smaller than the boundary.
    """
    samples = ee.FeatureCollection(SAMPLES)
    filled = []
    for band in bands:
        fill_value = ee.Number(samples.aggregate_mean(band))
        filled.append(image.select(band).unmask(fill_value, False).rename(band))
    return ee.Image.cat(filled)


def predictor_stack(bands: list[str]) -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    raw = ee.Image(FACTORS_2018).select(bands)
    return fill_predictor_masks(raw, bands).clip(study_area)


def probability_image(name: str, bands: list[str], seed: int) -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    samples = ee.FeatureCollection(SAMPLES).filter(ee.Filter.inList("label", [0, 1]))
    model = classifier(seed).train(features=samples, classProperty="label", inputProperties=bands)
    probability = predictor_stack(bands).classify(model).rename(f"prob_{name}").toFloat()

    return probability.clip(study_area).set(
        {
            "year": 2018,
            "resolution_m": 250,
            "model_family": "GEE_smileGradientTreeBoost",
            "feature_set": name,
            "output_type": "continuous_probability_no_classes",
            "nodata_policy": "predictor masks filled from training sample means using unmask(..., sameFootprint=False)",
            "sample_asset": SAMPLES,
            "factor_asset": FACTORS_2018,
            "seed": seed,
        }
    )


def main() -> None:
    ee.Initialize(project=PROJECT)
    study_area = ee.FeatureCollection(BOUNDARY)
    seed = 141300
    outputs = {
        "conventional": CONVENTIONAL_REDUCED,
        "alphaearth_embeddings": ALPHA,
        "conventional_alphaearth_embeddings": CONVENTIONAL_REDUCED + ALPHA,
    }
    plan = []
    for name, bands in outputs.items():
        prefix = f"cpec_2018_{name}_probability_250m_no_nodata"
        task = ee.batch.Export.image.toDrive(
            image=probability_image(name, bands, seed),
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
        plan.append(
            {
                "name": name,
                "drive_folder": DRIVE_FOLDER,
                "file_prefix": prefix,
                "resolution_m": 250,
                "task_id": task.id,
                "bands": len(bands),
                "nodata_policy": "predictor masks filled from training sample means before classification",
            }
        )
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
