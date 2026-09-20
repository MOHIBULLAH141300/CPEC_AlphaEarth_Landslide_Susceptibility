# Annual Dynamic Susceptibility Extension Plan, 2017-2024

Date: 2026-05-04

## Recommended Label

For the current 2018 V3 result, use:

**2018 dynamic-factor landslide susceptibility baseline**

For the next multi-year outputs, use:

**annual dynamic-factor landslide susceptibility maps, 2017-2024**

Only call the study fully **temporal landslide susceptibility modelling** after dated event validation or space-time validation is added.

## Why Annual Maps Will Differ

Static terrain/geology factors stay fixed, while annual rainfall, vegetation, land cover, and AlphaEarth embeddings change by year.

Manifest summary:

- Static factors: `11`
- Semi-static or semi-dynamic fixed-baseline factors: `3`
- Annual or annual-candidate factors: `14`

## Model-Compatible Annual V3 Core

The first annual extension should keep the 2018 V3 model structure stable:

- Conventional branch: keep the final 19 V3 factors.
- AlphaEarth branch: use A00-A63 for each year from 2017-2024.
- Fused branch: Conventional + AlphaEarth Embeddings.
- Grid: 250 m modelling grid.
- Boundary: official CPEC study area.

## Factors To Update Every Year

| field_name                  | readable_name                          | recommended_source                   | usable_years                      | decision_for_next_model                           |
| --------------------------- | -------------------------------------- | ------------------------------------ | --------------------------------- | ------------------------------------------------- |
| modis_lc_type1              | Land-cover class                       | MODIS/061/MCD12Q1                    | 2017-2024, with 2024 verification | Keep in annual maps                               |
| rain_monsoon_total          | Monsoon rainfall total                 | UCSB-CHG/CHIRPS/DAILY                | 2017-2024                         | Keep in annual maps                               |
| rain_max_1day               | Maximum 1-day rainfall                 | UCSB-CHG/CHIRPS/DAILY                | 2017-2024                         | Keep in annual maps                               |
| rain_annual_total           | Annual rainfall total                  | UCSB-CHG/CHIRPS/DAILY                | 2017-2024                         | Candidate; VIF/ablation before final annual model |
| rain_max_3day               | Maximum 3-day rainfall                 | UCSB-CHG/CHIRPS/DAILY                | 2017-2024                         | Candidate; VIF/ablation before final annual model |
| rain_max_7day               | Maximum 7-day rainfall                 | UCSB-CHG/CHIRPS/DAILY                | 2017-2024                         | Candidate; VIF/ablation before final annual model |
| ndvi_median                 | Median NDVI                            | MODIS/061/MOD13Q1                    | 2017-2024                         | Keep in annual maps                               |
| ndvi_amplitude              | NDVI amplitude                         | MODIS/061/MOD13Q1                    | 2017-2024                         | Keep in annual maps                               |
| ndvi_max                    | Maximum NDVI                           | MODIS/061/MOD13Q1                    | 2017-2024                         | Candidate; VIF/ablation before final annual model |
| evi_median                  | Median EVI                             | MODIS/061/MOD13Q1                    | 2017-2024                         | Candidate; VIF/ablation before final annual model |
| A00-A63                     | AlphaEarth Embeddings                  | GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL | 2017-2024                         | Keep in AlphaEarth and fused branches             |
| dynamic_world_probabilities | Dynamic World land-cover probabilities | GOOGLE/DYNAMICWORLD/V1               | 2017-2024                         | Optional V4/ablation, not in current V3 model     |
| lst_day_mean_c              | Mean daytime LST                       | MODIS/061/MOD11A2                    | 2017-2024                         | Do not add blindly; V3 dropped LST due high VIF   |
| lst_day_max_c               | Maximum daytime LST                    | MODIS/061/MOD11A2                    | 2017-2024                         | Do not add blindly; V3 dropped LST due high VIF   |

## Static Or Fixed-Baseline Factors

| field_name         | readable_name              | temporal_role | decision_for_next_model                                                  |
| ------------------ | -------------------------- | ------------- | ------------------------------------------------------------------------ |
| elevation_m        | Elevation                  | Static        | Keep in annual maps                                                      |
| slope_deg          | Slope                      | Static        | Keep in annual maps                                                      |
| aspect_deg         | Aspect                     | Static        | Keep in annual maps                                                      |
| profile_curvature  | Profile curvature          | Static        | Keep in annual maps                                                      |
| plan_curvature     | Plan curvature             | Static        | Keep in annual maps                                                      |
| tri                | Terrain ruggedness index   | Static        | Keep in annual maps                                                      |
| twi                | Topographic wetness index  | Static        | Keep in annual maps                                                      |
| valley_depth       | Valley depth               | Static        | Keep in annual maps                                                      |
| lithology_code     | Lithology                  | Static        | Keep in annual maps                                                      |
| soil_type          | Soil type                  | Static        | Keep in annual maps                                                      |
| log1p_dist_fault_m | Distance to active faults  | Static        | Keep in annual maps                                                      |
| log1p_dist_river_m | Distance to rivers/streams | Semi-static   | Keep fixed initially                                                     |
| log1p_dist_road_m  | Distance to roads          | Semi-static   | Keep fixed initially; run road-distance ablation later                   |
| eq_density_ms5     | Earthquake density         | Semi-dynamic  | Keep fixed initially; update later if post-2015 seismic catalog is added |

## Year Coverage Matrix

| field_name                  | readable_name                          | temporal_role         | source                                             | 2017                                | 2018                                | 2019                                | 2020                                | 2021                                | 2022                                | 2023                                | 2024                                |
| --------------------------- | -------------------------------------- | --------------------- | -------------------------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| elevation_m                 | Elevation                              | Static                | Existing V3/local stack                            | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| slope_deg                   | Slope                                  | Static                | Existing V3/local stack                            | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| aspect_deg                  | Aspect                                 | Static                | Existing V3/local stack                            | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| profile_curvature           | Profile curvature                      | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| plan_curvature              | Plan curvature                         | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| tri                         | Terrain ruggedness index               | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| twi                         | Topographic wetness index              | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| valley_depth                | Valley depth                           | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| lithology_code              | Lithology                              | Static                | Local CPEC vector/raster                           | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| soil_type                   | Soil type                              | Static                | Local CPEC raster                                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| log1p_dist_fault_m          | Distance to active faults              | Static                | Local official fault vector                        | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| log1p_dist_river_m          | Distance to rivers/streams             | Semi-static           | Local official hydrography vector                  | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| log1p_dist_road_m           | Distance to roads                      | Semi-static           | Local official 2018 road vector                    | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               | fixed                               |
| modis_lc_type1              | Land-cover class                       | Annual                | MODIS/061/MCD12Q1                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | verify in GEE                       |
| rain_monsoon_total          | Monsoon rainfall total                 | Annual                | UCSB-CHG/CHIRPS/DAILY                              | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| rain_max_1day               | Maximum 1-day rainfall                 | Annual                | UCSB-CHG/CHIRPS/DAILY                              | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| rain_annual_total           | Annual rainfall total                  | Annual                | UCSB-CHG/CHIRPS/DAILY                              | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| rain_max_3day               | Maximum 3-day rainfall                 | Annual                | UCSB-CHG/CHIRPS/DAILY                              | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| rain_max_7day               | Maximum 7-day rainfall                 | Annual                | UCSB-CHG/CHIRPS/DAILY                              | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| ndvi_median                 | Median NDVI                            | Annual                | MODIS/061/MOD13Q1                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| ndvi_amplitude              | NDVI amplitude                         | Annual                | MODIS/061/MOD13Q1                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| ndvi_max                    | Maximum NDVI                           | Annual                | MODIS/061/MOD13Q1                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| evi_median                  | Median EVI                             | Annual                | MODIS/061/MOD13Q1                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| eq_density_ms5              | Earthquake density                     | Semi-dynamic          | Local 1970-2015 earthquake data; USGS/GEM optional | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated | fixed until seismic catalog updated |
| A00-A63                     | AlphaEarth Embeddings                  | Annual                | GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL               | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| dynamic_world_probabilities | Dynamic World land-cover probabilities | Annual/near-real-time | GOOGLE/DYNAMICWORLD/V1                             | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| lst_day_mean_c              | Mean daytime LST                       | Annual                | MODIS/061/MOD11A2                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |
| lst_day_max_c               | Maximum daytime LST                    | Annual                | MODIS/061/MOD11A2                                  | available                           | available                           | available                           | available                           | available                           | available                           | available                           | available                           |

## Immediate Next Steps

1. Create annual predictor-stack export scripts for 2017-2024 using this manifest.
2. First export only 2017, 2018, and 2019 as a pilot.
3. Confirm that 2018 annual outputs reproduce the current V3 2018 factor values closely.
4. Apply the final 2018 stacked models to annual stacks.
5. Produce annual probability maps and change maps:
   - yearly probability
   - year minus 2018 baseline
   - persistent high susceptibility
   - increasing susceptibility
   - KKH/CPEC road exposure change
6. Add dated-event temporal validation only after inventory date enrichment.

Important validation note:

- The local HMA catalog contributes dated events inside the study area through 2018.
- For 2019-2024, true temporal validation will require additional dated inventory sources or manual event-date enrichment.

## Annual Output Structure And Storage Rule

Annual processing must be organized by year:

- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2017`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2018`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2019`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2020`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2021`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2022`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2023`
- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\2024`

Each year folder contains raw exports, 250 m predictor stacks, 250 m probability maps, previous-year change products, annual tables, QA, and logs for that year only.

Multi-year products are stored separately in:

- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\00_multi_year_summary_outputs`

## Required Multi-Year Outputs

The temporal paper should emphasize these products:

1. Year-to-year change maps:
   - 2018 minus 2017
   - 2019 minus 2018
   - 2020 minus 2019
   - 2021 minus 2020
   - 2022 minus 2021
   - 2023 minus 2022
   - 2024 minus 2023
2. Pixel-wise trend from 2017-2024.
3. Mean susceptibility from 2017-2024.
4. Persistent high-susceptibility zones.
5. Annual CPEC/KKH road exposure table and change summary.

## Resolution Rule

All temporal modelling outputs must remain at **250 m**.

Do not create 30 m display-resampled annual maps in this temporal workflow. This avoids unnecessary storage, prevents confusion between modelling resolution and display resolution, and keeps yearly processing computationally manageable.

## Literature/Data Support

- AlphaEarth/Satellite Embedding V1 is annual, 64-band, 10 m, and available from 2017-2024 in Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- CHIRPS Daily supports annual and extreme rainfall metrics from 1981 onward: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
- MODIS MOD13Q1 supports NDVI/EVI annual summaries at 250 m: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
- MODIS MOD11A2 supports annual LST candidates at 1 km: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
- MODIS MCD12Q1 supports annual land-cover class products: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
- Dynamic World offers optional 10 m land-cover probability features for Sentinel-2 era years: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- NASA HMA landslide catalog provides dated landslide records useful for temporal enrichment: https://nsidc.org/data/hma_ls_cat/versions/2

## Outputs Created

- `D:\DING PROJECT\02_methods\temporal_extension_2017_2024\annual_dynamic_factor_manifest_2017_2024.csv`
- `D:\DING PROJECT\02_methods\temporal_extension_2017_2024\annual_factor_year_coverage_matrix_2017_2024.csv`
- `D:\DING PROJECT\02_methods\temporal_extension_2017_2024\temporal_inventory_source_audit.csv`
- `D:\DING PROJECT\02_methods\temporal_extension_2017_2024\temporal_extension_decision_2017_2024.json`
- `D:\DING PROJECT\05_reports\temporal_inventory_audit_2017_2024_2026-05-04.md`
- `D:\DING PROJECT\05_reports\annual_dynamic_susceptibility_extension_plan_2017_2024_2026-05-04.md`
