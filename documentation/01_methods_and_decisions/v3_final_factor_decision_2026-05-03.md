# v3 Final Factor Decision Before Modelling

Date: 2026-05-03

## Literature Gate

The final v3 factor decision follows recent landslide susceptibility literature and the teacher's comments. The corrected model now includes the missing standard conditioning factors:

- road proximity
- river/stream proximity
- active fault proximity
- lithology/geology
- additional terrain morphology
- hydrological terrain index
- seismic background

## Data Gate

The v3 factors were sampled from full-CPEC sources in:

- `D:\CPEC data`

The archived KKH-subset folders were not used.

## Output Table

New sample table:

- `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v3.csv`

Rows:

- 3,316

Columns:

- 112

## Excluded Before Modelling

These were excluded for clear methodological reasons:

- `night_lights`: 100% missing in the current sample extraction.
- `dist_river_raster_m`: 60% missing; vector-derived `log1p_dist_river_m` is complete.
- `population_density`: better treated as exposure/vulnerability, not physical susceptibility.

## Dropped After Multicollinearity Diagnostics

- `relief`: highly redundant with TRI and LS factor; Spearman r with TRI = 0.962.
- `lst_day_mean_c`: strongly confounded with elevation in the CPEC-wide sample; VIF stayed above 12 when both were retained.
- raw distance fields `dist_road_m`, `dist_river_m`, `dist_fault_m`: replaced by log-transformed distance fields to reduce skew and avoid duplicate encoding.
- `rain_annual_total`, `rain_max_3day`, `rain_max_7day`: redundant with monsoon rainfall and 1-day extreme rainfall in this baseline; can be revisited for annual dynamic rainfall experiments.
- `ndvi_max`, `evi_median`, `lst_day_max_c`: redundant with selected vegetation/thermal variables.
- `ls_factor`: redundant with selected terrain morphology and slope/valley descriptors.

## Final v3 Conventional Factor Set

The selected v3 conventional factor set is:

1. `elevation_m`
2. `slope_deg`
3. `aspect_deg`
4. `rain_monsoon_total`
5. `rain_max_1day`
6. `ndvi_median`
7. `ndvi_amplitude`
8. `modis_lc_type1`
9. `log1p_dist_road_m`
10. `log1p_dist_river_m`
11. `log1p_dist_fault_m`
12. `lithology_code`
13. `profile_curvature`
14. `plan_curvature`
15. `tri`
16. `twi`
17. `valley_depth`
18. `soil_type`
19. `eq_density_ms5`

All selected v3 VIF values are below 4.

## AlphaEarth Branch

AlphaEarth embeddings remain as the fixed annual foundation-model representation:

- `A00` to `A63`

## Official v3 Model Groups

The next modelling step should compare:

- Conventional v3
- AlphaEarth Embeddings
- Conventional v3 + AlphaEarth Embeddings

## Saved Diagnostics

- `D:\DING PROJECT\05_reports\sample_v3_build_report_2026-05-03.md`
- `D:\DING PROJECT\05_reports\multicollinearity_assessment_2018_v3.md`
- `D:\DING PROJECT\03_models\multicollinearity_assessment_2018_v3`

## Next Step

Rerun the full model family using the v3 feature set:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Spatial-CV stacked ensemble

Then regenerate SHAP/TreeSHAP and probability maps only after v3 evaluation is complete.

