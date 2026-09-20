# Annual Dynamic Pilot Export Status: 2017-2019

Date checked: 2026-05-04

Purpose: start the temporal extension using the fixed 2018 Spatial-CV Stacked Ensemble model and annual predictor values. This pilot exports 2017-2019 first, so the workflow can be checked before launching all 2017-2024 years.

## Current Earth Engine Export Status

| Year | Product | Earth Engine state | Drive folder | File prefix |
| --- | --- | --- | --- | --- |
| 2017 | Annual dynamic base predictors | RUNNING | `GEE_CPEC_ANNUAL_DYNAMIC_2017_250M` | `cpec_2017_annual_dynamic_base_predictors_250m` |
| 2017 | AlphaEarth Embeddings | RUNNING | `GEE_CPEC_ANNUAL_DYNAMIC_2017_250M` | `cpec_2017_alphaearth_embeddings_250m` |
| 2018 | Annual dynamic base predictors | RUNNING | `GEE_CPEC_ANNUAL_DYNAMIC_2018_250M` | `cpec_2018_annual_dynamic_base_predictors_250m` |
| 2018 | AlphaEarth Embeddings | RUNNING | `GEE_CPEC_ANNUAL_DYNAMIC_2018_250M` | `cpec_2018_alphaearth_embeddings_250m` |
| 2019 | Annual dynamic base predictors | READY | `GEE_CPEC_ANNUAL_DYNAMIC_2019_250M` | `cpec_2019_annual_dynamic_base_predictors_250m` |
| 2019 | AlphaEarth Embeddings | READY | `GEE_CPEC_ANNUAL_DYNAMIC_2019_250M` | `cpec_2019_alphaearth_embeddings_250m` |

`RUNNING` means Earth Engine is actively processing the export. `READY` means the task is queued and waiting for an available Earth Engine worker.

## Local Google Drive Check

Google Drive sync root checked:

`C:\Users\Administrator\My Drive`

No annual GeoTIFF files had appeared in the local Google Drive folder at this check. This is expected while the Earth Engine tasks are still running or queued.

## Managed Local Destination

When exports complete and Google Drive syncs them locally, the files will be copied into year-specific managed folders:

`D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\<year>\00_raw_exports_from_gee`

The annual model outputs will then be written to:

`D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\<year>\02_probability_maps_250m`

## Reproducible Scripts

The following scripts are now saved in both the main scripts folder and the clean V3 package:

| Script | Role |
| --- | --- |
| `export_annual_dynamic_predictor_stacks_250m_to_drive.py` | Starts compact annual GEE exports: base predictors plus AlphaEarth Embeddings |
| `check_annual_gee_export_status.py` | Checks Earth Engine task states |
| `copy_annual_gee_exports_from_drive.py` | Copies completed Google Drive exports into the managed annual folders |
| `apply_fixed_2018_v3_stacked_model_to_annual_year.py` | Applies the fixed 2018 stacked models to one annual predictor year |

All four scripts passed Python syntax compilation after QA.

## Next Step After Exports Finish

1. Re-check task status.
2. Copy completed exports from Google Drive into the managed annual folders.
3. Apply the fixed 2018 stacked ensemble model year by year.
4. Inspect QA for nodata, probability range, raster alignment, and boundary coverage.
5. Only after the pilot years are clean, launch the remaining annual exports for 2020-2024.

