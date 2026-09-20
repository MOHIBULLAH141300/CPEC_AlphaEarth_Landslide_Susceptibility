"""Start managed Earth Engine exports for public annual CPEC factor stacks.

This script exports only newly built public-data factors clipped to the official
CPEC boundary. It does not use previous model outputs. By default it runs in
dry-run mode so the planned asset IDs can be reviewed before starting tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
ASSET_ROOT = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public"
LOG_PATH = Path(r"D:\DING PROJECT\04_maps\gee_public_factor_export_plan.json")


def annual_rainfall_factors(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(study_area)
        .select("precipitation")
    )
    day_count = end.difference(start, "day")

    def rolling_sum(days: int) -> ee.ImageCollection:
        offsets = ee.List.sequence(0, day_count.subtract(days))
        return ee.ImageCollection.fromImages(
            offsets.map(
                lambda day_offset: chirps.filterDate(
                    start.advance(ee.Number(day_offset), "day"),
                    start.advance(ee.Number(day_offset).add(days), "day"),
                ).sum()
            )
        )

    return ee.Image.cat(
        [
            chirps.sum().rename("rain_annual_total"),
            chirps.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 10, 1))
            .sum()
            .rename("rain_monsoon_total"),
            chirps.max().rename("rain_max_1day"),
            rolling_sum(3).max().rename("rain_max_3day"),
            rolling_sum(7).max().rename("rain_max_7day"),
        ]
    )


def annual_vegetation_factors(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).filterBounds(study_area)
    ndvi = modis.select("NDVI").map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    evi = modis.select("EVI").map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    return ee.Image.cat(
        [
            ndvi.median().rename("ndvi_median"),
            ndvi.max().rename("ndvi_max"),
            ndvi.max().subtract(ndvi.min()).rename("ndvi_amplitude"),
            evi.median().rename("evi_median"),
        ]
    )


def annual_thermal_factors(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .filterBounds(study_area)
        .select("LST_Day_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"]))
    )
    return ee.Image.cat([lst.mean().rename("lst_day_mean_c"), lst.max().rename("lst_day_max_c")])


def annual_land_cover_factors(year: int) -> ee.Image:
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filter(ee.Filter.calendarRange(year, year, "year")).first()
    return ee.Image(lc).select("LC_Type1").rename("modis_lc_type1")


def terrain_factors() -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(study_area).select("DEM").mosaic().clip(study_area)
    terrain = ee.Terrain.products(dem)
    return ee.Image.cat(
        [
            dem.rename("elevation_m"),
            terrain.select("slope").rename("slope_deg"),
            terrain.select("aspect").rename("aspect_deg"),
        ]
    )


def alphaearth_factors(year: int) -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(start, end)
        .filterBounds(study_area)
        .mosaic()
        .select("A.*")
    )


def build_stack(year: int, include_alphaearth: bool) -> ee.Image:
    study_area = ee.FeatureCollection(BOUNDARY)
    stack = (
        terrain_factors()
        .addBands(annual_rainfall_factors(year))
        .addBands(annual_vegetation_factors(year))
        .addBands(annual_thermal_factors(year))
        .addBands(annual_land_cover_factors(year))
    )
    if include_alphaearth and year >= 2017:
        stack = stack.addBands(alphaearth_factors(year))
    return (
        stack.clip(study_area)
        .toFloat()
        .set("year", year)
        .set("includes_alphaearth", include_alphaearth and year >= 2017)
        .set("boundary_asset", BOUNDARY)
        .set("method", "new_clean_public_factor_stack_no_old_model_outputs")
    )


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2018", help="Comma/range list, e.g. 2013-2024 or 2018,2024")
    parser.add_argument("--scale", type=int, default=250)
    parser.add_argument("--start", action="store_true", help="Actually start Earth Engine export tasks")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    years = parse_years(args.years)
    plan = []
    for year in years:
        include_alphaearth = year >= 2017
        suffix = "alphaearth" if include_alphaearth else "conventional"
        asset_id = f"{ASSET_ROOT}/cpec_public_factors_{year}_{suffix}_{args.scale}m"
        plan.append({"year": year, "include_alphaearth": include_alphaearth, "asset_id": asset_id, "scale": args.scale})
        if args.start:
            task = ee.batch.Export.image.toAsset(
                image=build_stack(year, include_alphaearth),
                description=f"cpec_public_factors_{year}_{suffix}_{args.scale}m",
                assetId=asset_id,
                region=ee.FeatureCollection(BOUNDARY).geometry(),
                scale=args.scale,
                crs="EPSG:4326",
                maxPixels=1e13,
            )
            task.start()
            plan[-1]["task_id"] = task.id

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    print(f"Saved plan to {LOG_PATH}")


if __name__ == "__main__":
    main()
