"""Dry-run annual CPEC factor bands in Earth Engine.

No exports are started. This only checks band names/counts for representative
years so the factor design is verifiable before heavy processing.
"""

import json

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"


def annual_rainfall_factors(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(study_area)
        .select("precipitation")
    )
    annual_total = chirps.sum().rename("rain_annual_total")
    monsoon_total = (
        chirps.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 10, 1))
        .sum()
        .rename("rain_monsoon_total")
    )
    max_1_day = chirps.max().rename("rain_max_1day")
    day_count = end.difference(start, "day")

    def rolling_sum(days):
        offsets = ee.List.sequence(0, day_count.subtract(days))
        return ee.ImageCollection.fromImages(
            offsets.map(
                lambda day_offset: chirps.filterDate(
                    start.advance(ee.Number(day_offset), "day"),
                    start.advance(ee.Number(day_offset).add(days), "day"),
                )
                .sum()
                .set("system:time_start", start.advance(ee.Number(day_offset), "day").millis())
            )
        )

    max_3_day = rolling_sum(3).max().rename("rain_max_3day")
    max_7_day = rolling_sum(7).max().rename("rain_max_7day")
    return ee.Image.cat([annual_total, monsoon_total, max_1_day, max_3_day, max_7_day])


def annual_vegetation_factors(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    study_area = ee.FeatureCollection(BOUNDARY)
    modis = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start, end)
        .filterBounds(study_area)
    )
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


def annual_thermal_factors(year):
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


def annual_land_cover_factors(year):
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filter(ee.Filter.calendarRange(year, year, "year")).first()
    return ee.Image(lc).select("LC_Type1").rename("modis_lc_type1")


def terrain_factors():
    study_area = ee.FeatureCollection(BOUNDARY)
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .filterBounds(study_area)
        .select("DEM")
        .mosaic()
        .clip(study_area)
    )
    terrain = ee.Terrain.products(dem)
    return ee.Image.cat(
        [
            dem.rename("elevation_m"),
            terrain.select("slope").rename("slope_deg"),
            terrain.select("aspect").rename("aspect_deg"),
        ]
    )


def alphaearth_factors(year):
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


def build_stack(year, include_alphaearth):
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
    return stack.clip(study_area).toFloat()


def main():
    ee.Initialize(project=PROJECT)
    results = []
    for year, include_alphaearth in [(2016, False), (2018, True)]:
        stack = build_stack(year, include_alphaearth)
        bands = stack.bandNames().getInfo()
        results.append(
            {
                "year": year,
                "include_alphaearth": include_alphaearth,
                "band_count": len(bands),
                "bands": bands,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
