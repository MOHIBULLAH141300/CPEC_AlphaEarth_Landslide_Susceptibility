# Annual Dynamic Susceptibility Status

## Current organization

Annual work is now organized one year at a time under:

`D:\DING PROJECT\04_maps\annual_dynamic_2017_2024`

Each year contains:

- `00_raw_exports_from_gee`
- `01_processed_predictor_stacks_250m`
- `02_probability_maps_250m`
- `03_change_from_previous_year`
- `04_figures`
- `05_year_specific_qa`
- `06_processing_logs`
- `99_background_raw_downloads`

Loose Google Drive ZIP downloads from the project root were moved into:

`D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\99_background_loose_drive_downloads`

## Data currently available locally

| Year | Base annual predictors | AlphaEarth embeddings | Local raw folder status |
|---:|---:|---:|---|
| 2017 | Available | Available | Complete |
| 2018 | Available | Available | Complete |
| 2019 | Available | Available | Complete |
| 2020 | Not yet local | Not yet local | Pending GEE export/download |
| 2021 | Not yet local | Not yet local | Pending GEE export/download |
| 2022 | Not yet local | Not yet local | Pending GEE export/download |
| 2023 | Not yet local | Not yet local | Pending GEE export/download |
| 2024 | Not yet local | Not yet local | Pending GEE export/download |

Each complete local year currently has seven raw TIFF files:

- one annual base-predictor GeoTIFF,
- six AlphaEarth embedding GeoTIFF tiles.

## Annual model-output status

No annual year should be treated as a finished temporal susceptibility result
yet. A 2017 run was interrupted: it produced a Conventional probability file,
but the AlphaEarth output is a zero-byte incomplete file and the fused output
is missing. Those interrupted files are background only and should be rerun
cleanly before any temporal result is used.

## Next processing rule

Run the fixed 2018 V3 stacked models one year at a time. Do not merge all
years into one temporary workspace. The probability outputs for each year
should stay in that year's `02_probability_maps_250m` folder.

Suggested order:

1. Process 2017.
2. QA 2017.
3. Process 2018.
4. QA 2018.
5. Process 2019.
6. Export/download 2020, then repeat the same one-year workflow through 2024.

## Important note

The dynamic annual maps should use raw stacked probabilities as the main
susceptibility products. Reliability/confidence-adjusted products can be
created afterward as secondary maps if annual Area-of-Applicability products
are generated.
