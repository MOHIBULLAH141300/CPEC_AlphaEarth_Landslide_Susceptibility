"""Create the annual dynamic susceptibility workspace for 2017-2024.

The structure keeps raw exports, processed stacks, probability maps, QA, and
logs inside each year-specific folder so annual products do not mix. Multi-year
products such as trend, mean susceptibility, and persistent high-susceptibility
zones are stored separately.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(r"D:\DING PROJECT")
WORKSPACE = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024"
PACKAGE_ROOT = PROJECT_ROOT / "00_READ_ME_FIRST_2018_V3_RESULTS"
PACKAGE_TEMPORAL = PACKAGE_ROOT / "09_temporal_extension_plan"
README = WORKSPACE / "README_ANNUAL_DIRECTORY_STRUCTURE.md"

YEARS = range(2017, 2025)
YEAR_SUBDIRS = [
    "00_raw_exports_from_gee",
    "01_predictor_stack_250m",
    "02_probability_maps_250m",
    "03_previous_year_change_250m",
    "04_year_specific_tables",
    "05_year_specific_qa",
    "06_processing_logs",
]
SUMMARY_SUBDIRS = [
    "01_year_to_year_change_250m",
    "02_trend_2017_2024_250m",
    "03_mean_susceptibility_2017_2024_250m",
    "04_persistent_high_susceptibility_250m",
    "05_annual_road_exposure_tables",
    "06_summary_figures",
    "07_summary_qa",
]


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        ydir = WORKSPACE / str(year)
        for subdir in YEAR_SUBDIRS:
            (ydir / subdir).mkdir(parents=True, exist_ok=True)
        (ydir / f"README_{year}.md").write_text(
            f"""# Annual Dynamic Susceptibility Workspace: {year}

This folder contains all raw and processed materials for the {year} annual dynamic-factor susceptibility map.

## Folder Rules

- `00_raw_exports_from_gee`: temporary annual GEE exports for {year}. Keep only if needed for QA.
- `01_predictor_stack_250m`: compressed final 250 m annual predictor stack for {year}, if retained.
- `02_probability_maps_250m`: final 250 m model probability rasters for {year}.
- `03_previous_year_change_250m`: year-to-year change raster stored in the later year folder. For example, 2019 minus 2018 belongs inside the 2019 folder.
- `04_year_specific_tables`: annual metrics, thresholds, and exposure tables for {year}.
- `05_year_specific_qa`: raster QA, missing-data checks, and model-input checks for {year}.
- `06_processing_logs`: task logs and GEE/local processing notes for {year}.

## Naming Rule

Use this label in public outputs:

`CPEC {year} annual dynamic-factor susceptibility map`

No 30 m display-resampled products should be created for this temporal workflow unless explicitly requested later.
""",
            encoding="utf-8",
        )

    summary = WORKSPACE / "00_multi_year_summary_outputs"
    for subdir in SUMMARY_SUBDIRS:
        (summary / subdir).mkdir(parents=True, exist_ok=True)

    README.write_text(
        """# Annual Dynamic Susceptibility Workspace, 2017-2024

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
""",
        encoding="utf-8",
    )

    PACKAGE_TEMPORAL.mkdir(parents=True, exist_ok=True)
    (PACKAGE_TEMPORAL / README.name).write_bytes(README.read_bytes())

    print(f"Created annual dynamic workspace: {WORKSPACE}")
    print(f"Copied structure README to: {PACKAGE_TEMPORAL / README.name}")


if __name__ == "__main__":
    main()
