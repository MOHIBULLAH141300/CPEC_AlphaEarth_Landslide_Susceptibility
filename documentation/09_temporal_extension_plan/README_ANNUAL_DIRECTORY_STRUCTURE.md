# Annual Dynamic Susceptibility Workspace, 2017-2024

This workspace is for the Option 1 temporal strategy:

**Use the same trained 2018 V3 model and apply it separately to each annual predictor stack from 2017 to 2024.**

## Core Rules

1. Process one year at a time.
2. Store raw and processed materials inside that year's folder.
3. Use 250 m as the modelling and output resolution.
4. Do not create 30 m display-resampled maps for this temporal workflow.
5. Keep only necessary raw exports; final outputs should be compressed GeoTIFFs and compact tables.
6. Store multi-year summaries separately in `00_multi_year_summary_outputs`.

## Annual Folder Layout

Each year has:

- `00_raw_exports_from_gee`
- `01_predictor_stack_250m`
- `02_probability_maps_250m`
- `03_previous_year_change_250m`
- `04_year_specific_tables`
- `05_year_specific_qa`
- `06_processing_logs`

## Multi-Year Outputs

Multi-year outputs should go in `00_multi_year_summary_outputs`:

- `01_year_to_year_change_250m`: 2018-2017, 2019-2018, ..., 2024-2023.
- `02_trend_2017_2024_250m`: pixel-wise trend/slope over 2017-2024.
- `03_mean_susceptibility_2017_2024_250m`: mean annual probability.
- `04_persistent_high_susceptibility_250m`: consistently high zones across years.
- `05_annual_road_exposure_tables`: road exposure per year and annual change.
- `06_summary_figures`: publication figures.
- `07_summary_qa`: QA for all multi-year products.

## Change Products

Use previous-year change as the main temporal difference:

- `2018_minus_2017`
- `2019_minus_2018`
- `2020_minus_2019`
- `2021_minus_2020`
- `2022_minus_2021`
- `2023_minus_2022`
- `2024_minus_2023`

The change raster should also be copied into the later year's `03_previous_year_change_250m` folder.

## Why No 30 m Display Products Here

The temporal workflow must stay computationally efficient and scientifically clean. The model operates at 250 m, so all annual modelling outputs and change products remain at 250 m. Cartographic 30 m resampling is skipped for this phase.
