r"""Export annual model-compatible GEE predictor stacks at 250 m.

This script follows the storage-safe temporal strategy:
- process one year at a time,
- export only compact, non-duplicated annual stacks,
- keep 250 m modelling resolution,
- do not create 30 m display products.

For each year it exports two GeoTIFFs to a year-specific Google Drive folder:
1. annual_dynamic_base_predictors: 8 model-compatible GEE bands
2. alphaearth_embeddings: 64 AlphaEarth/Satellite Embedding bands

The 11 static/semi-static local V3 factors are reused locally from:
D:\DING PROJECT\04_maps\stacked_ensemble_local_v3_factors_250m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
WORKSPACE = Path(r"D:\DING PROJECT\04_maps\annual_dynamic_2017_2024")
PACKAGE_TEMPORAL = Path(
    r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\09_temporal_extension_plan"
)

BASE_BANDS = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "modis_lc_type1",
]
ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    years = sorted(set(years))
    bad = [y for y in years if y < 2017 or y > 2024]
    if bad:
        raise ValueError(f"AlphaEarth annual workflow supports 2017-2024 only. Bad years: {bad}")
    return years


def study_area() -> ee.FeatureCollection:
    return ee.FeatureCollection(BOUNDARY)


def annual_rainfall_factors(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    area = study_area()
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(area)
        .select("precipitation")
    )
    monsoon = chirps.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 10, 1))
    return ee.Image.cat(
        [
            monsoon.sum().rename("rain_monsoon_total"),
            chirps.max().rename("rain_max_1day"),
        ]
    )


def annual_vegetation_factors(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    area = study_area()
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).filterBounds(area)
    ndvi = modis.select("NDVI").map(
        lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"])
    )
    return ee.Image.cat(
        [
            ndvi.median().rename("ndvi_median"),
            ndvi.max().subtract(ndvi.min()).rename("ndvi_amplitude"),
        ]
    )


def annual_land_cover_factor(year: int) -> ee.Image:
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    fallback = ee.ImageCollection("MODIS/061/MCD12Q1").filter(
        ee.Filter.calendarRange(year - 1, year - 1, "year")
    )
    img = ee.Image(ee.Algorithms.If(lc.size().gt(0), lc.first(), fallback.first()))
    return img.select("LC_Type1").rename("modis_lc_type1")


def terrain_factors() -> ee.Image:
    area = study_area()
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .filterBounds(area)
        .select("DEM")
        .mosaic()
        .clip(area)
    )
    terrain = ee.Terrain.products(dem)
    return ee.Image.cat(
        [
            dem.rename("elevation_m"),
            terrain.select("slope").rename("slope_deg"),
            terrain.select("aspect").rename("aspect_deg"),
        ]
    )


def annual_dynamic_base_stack(year: int) -> ee.Image:
    area = study_area()
    stack = (
        terrain_factors()
        .addBands(annual_rainfall_factors(year))
        .addBands(annual_vegetation_factors(year))
        .addBands(annual_land_cover_factor(year))
        .select(BASE_BANDS)
        .toFloat()
        .clip(area)
        .set(
            {
                "year": year,
                "feature_set": "annual_dynamic_base_predictors",
                "band_count": len(BASE_BANDS),
                "bands": ",".join(BASE_BANDS),
                "resolution_m": 250,
                "boundary_asset": BOUNDARY,
                "temporal_strategy": "fixed_2018_v3_model_applied_to_annual_predictors",
            }
        )
    )
    return stack


def annual_alphaearth_stack(year: int) -> ee.Image:
    area = study_area()
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(start, end)
        .filterBounds(area)
        .mosaic()
        .select(ALPHA_BANDS)
        .toFloat()
        .clip(area)
        .set(
            {
                "year": year,
                "feature_set": "alphaearth_embeddings",
                "band_count": len(ALPHA_BANDS),
                "bands": ",".join(ALPHA_BANDS),
                "resolution_m": 250,
                "boundary_asset": BOUNDARY,
                "temporal_strategy": "fixed_2018_v3_model_applied_to_annual_predictors",
            }
        )
    )


def dataset_counts(year: int) -> dict[str, int]:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return {
        "chirps_daily": int(
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).size().getInfo()
        ),
        "mod13q1": int(
            ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).size().getInfo()
        ),
        "mcd12q1": int(
            ee.ImageCollection("MODIS/061/MCD12Q1")
            .filter(ee.Filter.calendarRange(year, year, "year"))
            .size()
            .getInfo()
        ),
        "alphaearth_tiles": int(
            ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(start, end)
            .size()
            .getInfo()
        ),
    }


def export_image(
    image: ee.Image, year: int, product: str, drive_folder: str, start_task: bool
) -> dict[str, object]:
    file_prefix = f"cpec_{year}_{product}_250m"
    record: dict[str, object] = {
        "year": year,
        "product": product,
        "drive_folder": drive_folder,
        "file_prefix": file_prefix,
        "scale": 250,
        "crs": "EPSG:4326",
        "status": "planned",
    }
    if start_task:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=file_prefix,
            folder=drive_folder,
            fileNamePrefix=file_prefix,
            region=study_area().geometry(),
            scale=250,
            crs="EPSG:4326",
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        record["task_id"] = task.id
        record["status"] = "started"
    return record


def write_year_log(year: int, records: list[dict[str, object]]) -> None:
    log_dir = WORKSPACE / str(year) / "06_processing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"gee_annual_export_plan_{year}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (log_dir / f"gee_annual_export_plan_{year}.md").write_text(
        "# GEE Annual Export Plan\n\n"
        f"Year: {year}\n\n"
        "Products:\n"
        + "\n".join(
            f"- {r['product']}: Drive folder `{r['drive_folder']}`, prefix `{r['file_prefix']}`, status `{r['status']}`"
            for r in records
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017-2019", help="Comma/range list, e.g. 2017-2019")
    parser.add_argument("--start", action="store_true", help="Start Earth Engine export tasks")
    parser.add_argument(
        "--drive-prefix",
        default="GEE_CPEC_ANNUAL_DYNAMIC",
        help="Prefix for one Drive export folder per year",
    )
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    years = parse_years(args.years)
    all_records: list[dict[str, object]] = []
    for year in years:
        drive_folder = f"{args.drive_prefix}_{year}_250M"
        counts = dataset_counts(year)
        year_records = [
            export_image(
                annual_dynamic_base_stack(year),
                year,
                "annual_dynamic_base_predictors",
                drive_folder,
                args.start,
            ),
            export_image(
                annual_alphaearth_stack(year),
                year,
                "alphaearth_embeddings",
                drive_folder,
                args.start,
            ),
        ]
        for record in year_records:
            record["dataset_counts"] = counts
            record["local_year_folder"] = str(WORKSPACE / str(year))
            record["expected_local_raw_export_folder"] = str(
                WORKSPACE / str(year) / "00_raw_exports_from_gee"
            )
        write_year_log(year, year_records)
        all_records.extend(year_records)

    aggregate = WORKSPACE / "gee_annual_export_plan_latest.json"
    aggregate.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    PACKAGE_TEMPORAL.mkdir(parents=True, exist_ok=True)
    (PACKAGE_TEMPORAL / aggregate.name).write_bytes(aggregate.read_bytes())
    print(json.dumps(all_records, indent=2))
    print(f"Saved aggregate plan to: {aggregate}")


if __name__ == "__main__":
    main()
