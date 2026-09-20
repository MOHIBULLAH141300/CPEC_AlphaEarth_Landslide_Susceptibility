# Spatial-CV Stacked Ensemble Probability Raster Workflow

Date: 2026-05-03

## Required Probability Rasters

The paper-facing raster outputs must be produced from the Spatial-CV Stacked Ensemble model for:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

These are different from the earlier GEE Gradient Tree Boost probability rasters. The stacked ensemble is a local Python model, so Earth Engine cannot directly classify the rasters with it.

## Correct Workflow

1. Export GEE-available predictor stacks from GEE at 250 m:
   - `cpec_2018_conventional_base_gee_available_predictor_stack_250m`
   - `cpec_2018_alphaearth_embeddings_gee_available_predictor_stack_250m`
   - `cpec_2018_conventional_base_alphaearth_embeddings_gee_available_predictor_stack_250m`

2. Generate the missing v3 predictors from local full-CPEC data:
   - `log1p_dist_road_m`
   - `log1p_dist_river_m`
   - `log1p_dist_fault_m`
   - `lithology_code`
   - `profile_curvature`
   - `plan_curvature`
   - `tri`
   - `twi`
   - `valley_depth`
   - `soil_type`
   - `eq_density_ms5`

3. Download/assemble the GeoTIFFs into:
   - `D:\DING PROJECT\04_maps\stacked_ensemble_inputs_250m`

4. Apply the saved local base models and stacked meta-learner:
   - `D:\DING PROJECT\06_scripts\python\apply_2018_stacked_ensemble_to_predictor_stacks.py`

5. Save continuous probability rasters here:
   - `D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble`

## Scripts

- GEE export script:
  - `D:\DING PROJECT\06_scripts\python\export_2018_stacked_ensemble_predictor_stacks_250m_to_drive.py`

- Corrected GEE-available export script:
  - `D:\DING PROJECT\06_scripts\python\export_2018_gee_available_predictor_stacks_250m_to_drive.py`

- Local stacked raster prediction script:
  - `D:\DING PROJECT\06_scripts\python\apply_2018_stacked_ensemble_to_predictor_stacks.py`

## GEE Status Check

The attempted direct complete-stack exports failed for Conventional and Conventional + AlphaEarth Embeddings because the GEE public factor image does not contain the local v3 factors:

- `log1p_dist_road_m`
- `log1p_dist_river_m`
- `log1p_dist_fault_m`
- `lithology_code`
- `profile_curvature`
- `plan_curvature`
- `tri`
- `twi`
- `valley_depth`
- `soil_type`
- `eq_density_ms5`

Corrected replacement GEE export tasks were started for the GEE-available bands:

| Export | Task ID | Status at last check |
|---|---|---|
| Conventional base GEE stack | `SDNF67JGVWJBPP4RRYYTGQD2` | RUNNING |
| AlphaEarth Embeddings GEE stack | `3JXU4LD7CLVPW7VPR5IUHDSJ` | RUNNING |
| Conventional base + AlphaEarth GEE stack | `Y2UCP2WOOKQB7EFDBKVZ675H` | RUNNING |

## Output Rasters

Expected final rasters:

- `cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif`
- `cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif`
- `cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif`

## 30 m Display Products

A 30 m resampled display version can be created after the 250 m stacked rasters exist. This improves cartographic appearance but must be described as display resampling, not true 30 m model prediction.

For true 30 m stacked predictions, the same workflow must be repeated with 30 m predictor stacks, which will create very large rasters.
