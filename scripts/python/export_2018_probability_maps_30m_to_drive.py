"""Export 30 m continuous 2018 probability maps to Google Drive.

This corrects the earlier 250 m deployment rasters. The 30 m workflow rebuilds
the 2018 predictor stack from source datasets in GEE, fills predictor masks
from training-sample means to avoid edge holes, and exports continuous
probability rasters over the official CPEC boundary.

No susceptibility classes are produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
DRIVE_FOLDER = "GEE_CPEC_LSM_RASTERS_30M"
PLAN = Path(r"D:\DING PROJECT\04_maps\drive_probability_raster_30m_export_plan.json")

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


def annual_rainfall_2018() -> ee.Image:
    start = ee.Date.fromYMD(2018, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(study_area)
        .select("precipitation")
    )
    return ee.Image.cat(
        [
            chirps.filterDate(ee.Date.fromYMD(2018, 6, 1), ee.Date.fromYMD(2018, 10, 1))
            .sum()
            .rename("rain_monsoon_total"),
            chirps.max().rename("rain_max_1day"),
        ]
    ).resample("bilinear")


def vegetation_2018() -> ee.Image:
    start = ee.Date.fromYMD(2018, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).filterBounds(study_area)
    ndvi = modis.select("NDVI").map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    return ee.Image.cat(
        [
            ndvi.median().rename("ndvi_median"),
            ndvi.max().subtract(ndvi.min()).rename("ndvi_amplitude"),
        ]
    ).resample("bilinear")


def thermal_2018() -> ee.Image:
    start = ee.Date.fromYMD(2018, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .filterBounds(study_area)
        .select("LST_Day_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"]))
    )
    return lst.mean().rename("lst_day_mean_c").resample("bilinear")


def landcover_2018() -> ee.Image:
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filter(ee.Filter.calendarRange(2018, 2018, "year")).first()
    return ee.Image(lc).select("LC_Type1").rename("modis_lc_type1")


def terrain() -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(study_area).select("DEM").mosaic()
    terrain_products = ee.Terrain.products(dem)
    return ee.Image.cat(
        [
            dem.rename("elevation_m"),
            terrain_products.select("slope").rename("slope_deg"),
            terrain_products.select("aspect").rename("aspect_deg"),
        ]
    )


def alphaearth_2018() -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate("2018-01-01", "2019-01-01")
        .filterBounds(study_area)
        .mosaic()
        .select("A.*")
        .resample("bilinear")
    )


def fill_from_sample_means(image: ee.Image, bands: list[str]) -> ee.Image:
    samples = ee.FeatureCollection(SAMPLES)
    filled = []
    for band in bands:
        fill = ee.Number(samples.aggregate_mean(band))
        filled.append(image.select(band).unmask(fill).rename(band))
    return ee.Image.cat(filled)


def predictor_stack_30m(bands: list[str]) -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    stack = (
        terrain()
        .addBands(annual_rainfall_2018())
        .addBands(vegetation_2018())
        .addBands(thermal_2018())
        .addBands(landcover_2018())
        .addBands(alphaearth_2018())
        .select(bands)
    )
    return fill_from_sample_means(stack, bands).clip(study_area)


def classifier(seed: int) -> ee.Classifier:
    return ee.Classifier.smileGradientTreeBoost(
        numberOfTrees=500,
        shrinkage=0.03,
        samplingRate=0.85,
        maxNodes=32,
        loss="Logistic",
        seed=seed,
    ).setOutputMode("PROBABILITY")


def probability_image(name: str, bands: list[str], seed: int) -> ee.Image:
    samples = ee.FeatureCollection(SAMPLES).filter(ee.Filter.inList("label", [0, 1]))
    model = classifier(seed).train(features=samples, classProperty="label", inputProperties=bands)
    return (
        predictor_stack_30m(bands)
        .classify(model)
        .rename(f"prob_{name}")
        .toFloat()
        .set(
            {
                "year": 2018,
                "resolution_m": 30,
                "model_family": "GEE_smileGradientTreeBoost",
                "feature_set": name,
                "output_type": "continuous_probability_no_classes",
                "mask_fill": "predictor_missing_values_filled_from_training_sample_means",
                "sample_asset": SAMPLES,
                "seed": seed,
            }
        )
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
        image = probability_image(name, bands, seed)
        prefix = f"cpec_2018_{name}_probability_30m"
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=prefix,
            folder=DRIVE_FOLDER,
            fileNamePrefix=prefix,
            region=study_area.geometry(),
            scale=30,
            crs="EPSG:4326",
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            fileDimensions=8192,
            skipEmptyTiles=True,
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        plan.append(
            {
                "name": name,
                "drive_folder": DRIVE_FOLDER,
                "file_prefix": prefix,
                "resolution_m": 30,
                "task_id": task.id,
                "bands": len(bands),
                "note": "continuous probability; no classes; missing predictor masks filled from training-sample means",
            }
        )
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
