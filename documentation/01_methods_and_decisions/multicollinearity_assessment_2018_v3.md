# 2018 v3 Multicollinearity Assessment

## Literature Gate

The v3 diagnostic set adds standard landslide-conditioning factors supported by recent literature: road proximity, river/stream proximity, fault proximity, lithology, terrain morphology, TWI, seismic background, vegetation, rainfall, thermal, land-cover, and AlphaEarth embedding branches.

## Pre-Modelling Exclusions

- Night lights (`night_lights`): Excluded before modelling because local samples are 100% missing.
- Legacy river-distance raster (`dist_river_raster_m`): Excluded because 60% of samples are missing and vector-derived distance to rivers is complete.
- Population density (`population_density`): Excluded from susceptibility modelling because it is better treated as exposure/vulnerability, not a physical instability control.

## Seed Selected Conventional v3 Factors

- Elevation (`elevation_m`)
- Slope (`slope_deg`)
- Aspect (`aspect_deg`)
- Monsoon rainfall total (`rain_monsoon_total`)
- Maximum 1-day rainfall (`rain_max_1day`)
- NDVI median (`ndvi_median`)
- NDVI amplitude (`ndvi_amplitude`)
- Mean daytime LST (`lst_day_mean_c`)
- MODIS land cover (`modis_lc_type1`)
- Distance to roads (`log1p_dist_road_m`)
- Distance to rivers/streams (`log1p_dist_river_m`)
- Distance to active faults (`log1p_dist_fault_m`)
- Lithology/geology class (`lithology_code`)
- Profile curvature (`profile_curvature`)
- Plan curvature (`plan_curvature`)
- Terrain ruggedness index (`tri`)
- Topographic wetness index (`twi`)
- Valley depth (`valley_depth`)
- Relief (`relief`)
- Soil type (`soil_type`)
- Earthquake density > Ms5 (`eq_density_ms5`)

## Top High-Correlation Pairs

| Factor A | Factor B | Spearman r |
|---|---|---:|
| Distance to roads | Distance to roads | 1.000 |
| Distance to rivers/streams | Distance to rivers/streams | 1.000 |
| Distance to active faults | Distance to active faults | 1.000 |
| Terrain ruggedness index | Relief | 0.962 |
| Maximum 3-day rainfall | Maximum 7-day rainfall | 0.959 |
| Mean daytime LST | Maximum daytime LST | 0.952 |
| NDVI median | EVI median | 0.940 |
| Maximum 1-day rainfall | Maximum 3-day rainfall | 0.901 |
| Annual rainfall total | Maximum 7-day rainfall | 0.898 |
| Annual rainfall total | Maximum 3-day rainfall | 0.886 |
| Elevation | Mean daytime LST | -0.879 |
| Relief | LS factor | 0.864 |
| Maximum 1-day rainfall | Maximum 7-day rainfall | 0.857 |

## Seed Selected VIF

| Factor | VIF |
|---|---:|
| Relief | 31.33 |
| Terrain ruggedness index | 28.82 |
| Mean daytime LST | 12.32 |
| Elevation | 12.21 |
| Monsoon rainfall total | 4.03 |
| Maximum 1-day rainfall | 3.66 |
| NDVI median | 2.81 |
| NDVI amplitude | 1.94 |
| Valley depth | 1.90 |
| Lithology/geology class | 1.89 |
| Distance to rivers/streams | 1.58 |
| Slope | 1.52 |
| Topographic wetness index | 1.51 |
| MODIS land cover | 1.48 |
| Plan curvature | 1.44 |
| Aspect | 1.39 |
| Distance to active faults | 1.35 |
| Soil type | 1.32 |
| Distance to roads | 1.31 |
| Earthquake density > Ms5 | 1.25 |
| Profile curvature | 1.22 |

## Saved Tables

- Correlation matrix: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_spearman_correlation_matrix.csv`
- High-correlation pairs: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_high_correlation_pairs_abs_ge_0_85.csv`
- Candidate VIF: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_candidate_vif.csv`
- Seed selected VIF: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_seed_selected_vif.csv`
- Initial decisions: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_initial_keep_drop_decisions.csv`
- Seed selected factor list: `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3\v3_seed_selected_factor_list.csv`

## Next Step

Review high-VIF factors and finalize the v3 conventional feature set before rerunning models. If several terrain factors remain collinear, keep the most physically interpretable and best-performing subset.
