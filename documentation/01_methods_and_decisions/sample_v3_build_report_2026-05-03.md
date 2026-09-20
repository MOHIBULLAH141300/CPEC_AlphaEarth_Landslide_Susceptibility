# cpec_2018_lsm_samples_v3 Build Report

## Literature Gate

The added factors pass the landslide-susceptibility literature gate: road proximity, river/stream proximity, fault proximity, lithology, terrain morphology, hydrological terrain indices, and seismic background are standard conditioning variables in recent ML/ensemble LSM studies.

## Output

- Input: `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v2.csv`
- Output: `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v3.csv`
- Rows: 3316
- Columns: 112

## Added Factors

| Factor | Type | Missing % | Min | Max | Mean |
|---|---|---:|---:|---:|---:|
| dist_road_m | vector_distance_m | 0.00 | 0.604 | 165490.789 | 28749.241 |
| dist_river_m | vector_distance_m | 0.00 | 2.141 | 67084.788 | 13919.345 |
| dist_fault_m | vector_distance_m | 0.00 | 5.102 | 223803.402 | 23157.954 |
| lithology_code | polygon_class_factorized | 0.15 | 0.000 | 23.000 | 13.594 |
| profile_curvature | raster_sample | 0.12 | 0.000 | 30.976 | 5.217 |
| plan_curvature | raster_sample | 0.12 | 0.000 | 81.324 | 27.099 |
| tri | raster_sample | 2.29 | 0.447 | 135.813 | 7.913 |
| twi | raster_sample | 0.12 | 3.424 | 24.220 | 7.500 |
| valley_depth | raster_sample | 0.12 | -1.500 | 1374.808 | 337.912 |
| relief | raster_sample | 0.12 | 0.000 | 528.000 | 30.812 |
| ls_factor | raster_sample | 0.12 | 0.000 | 302.905 | 26.624 |
| soil_type | raster_sample | 0.36 | 1.000 | 35.000 | 16.462 |
| eq_density_ms5 | raster_sample | 0.12 | 1.000 | 5.000 | 1.932 |
| population_density | raster_sample | 0.78 | 0.000 | 34269.891 | 96.306 |
| night_lights | raster_sample | 100.00 | nan | nan | nan |
| dist_river_raster_m | raster_sample | 60.01 | 2.000 | 8.000 | 3.882 |

## Saved QA

- Sources: `D:\DING PROJECT\03_models\sample_v3_factor_audit\sample_v3_sources.json`
- Added-factor QA: `D:\DING PROJECT\03_models\sample_v3_factor_audit\sample_v3_added_factor_qa.csv`

## Next Step

Run v3 multicollinearity diagnostics before modelling. Factors with excessive missingness, extreme redundancy, or weak justification should be dropped before final model training.
