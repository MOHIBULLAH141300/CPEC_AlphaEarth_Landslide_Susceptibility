# Dynamic susceptibility results, 2017-2024

This folder is the clean supervisor-facing package for the temporal extension of the CPEC landslide susceptibility project.

## Modelling rule

The 2018 trained Spatial-CV Stacked Ensemble models were kept fixed and applied to year-specific annual predictors from 2017 to 2024. This means the annual maps show dynamic predictor effects under the same validated model structure. They should be interpreted as temporal-transfer or dynamic-covariate susceptibility maps, not as independently retrained annual models.

## Completed outputs

- Annual 250 m probability maps for 2017-2024 are complete for Conventional, AlphaEarth Embeddings, and Conventional + AlphaEarth Embeddings.
- Multi-year fused temporal products are complete: mean probability, temporal standard deviation, temporal range, Theil-Sen trend, high-probability year count, persistent high-probability mask, year of maximum probability, year of minimum probability, and year-to-year change maps.
- Large rasters are not duplicated in this clean folder to save disk space. Their exact locations are listed in 04_raster_indexes_large_files_not_duplicated.

## First interpretation

- Highest mean fused susceptibility years: 2021, 2022.
- Lowest mean fused susceptibility years: 2024, 2019.
- Persistent high-susceptibility area, using the 2018 fused P80 threshold and occurrence in at least 6 of 8 years: 18.1 percent of valid study-area pixels.
- The temporal products support discussion of persistent hotspots, dynamic annual changes, and years/areas where annual environmental conditions intensify or reduce susceptibility.

## What to open first

1. 01_publication_figures for visual summary figures.
2. 02_tables_and_diagnostics\annual_fused_probability_summary_2017_2024.csv for yearly statistics.
3. 04_raster_indexes_large_files_not_duplicated to locate the actual GeoTIFF rasters.
4. 03_methods_and_interpretation\temporal_summary_methods_and_thresholds.json for thresholds and processing rules.
