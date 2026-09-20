# Reproducibility guide

## Environment

Use Python 3.10 or newer. Install the packages listed in `requirements.txt` in an isolated environment. Google Earth Engine scripts additionally require an authenticated Earth Engine account and access to the source collections named in the manuscript.

## Data layout

The executed scripts were developed against a project root containing these logical locations:

```text
01_clean_data/     Harmonised inventories, controls, and matched-control samples
03_models/         Input sample tables and generated model outputs
04_maps/           Predictor stacks, probability rasters, AoA rasters, and road outputs
05_reports/        Figures and analysis reports
06_scripts/        Python, PowerShell, and Earth Engine scripts
```

The public repository uses `model_outputs/`, `documentation/`, and `scripts/` to make the archived results easier to inspect. Before rerunning a script, set its `PROJECT_ROOT` or `PROJECT` constant to a local working directory that follows the logical layout above. Scripts that use local boundary, road, or raster paths must also be pointed to locally downloaded copies of those inputs.

## Primary execution order

1. Prepare the harmonised sample table and final conventional factor list.
2. Run `run_manuscript_v3_nested_spatial_cv.py` for pooled predictive skill.
3. Run `run_manuscript_v3_leave_one_domain_out.py` for comparative cross-domain transfer.
4. Run `run_manuscript_v3_harmonised_aoa.py` for source-only predictor-domain support.
5. Run `run_manuscript_v3_robustness_experiments.py` and `run_manuscript_v3_spatial_block_model_comparisons.py`.
6. Run `run_manuscript_v6_partition_sensitivity.py` and `run_manuscript_v6_spatial_design_sensitivity.py`.
7. Run the road-exposure and road-distance-ablation scripts after producing the fused susceptibility and support rasters.
8. Generate manuscript figures from the archived CSV outputs and rasters.

## Deterministic design choices

- Random seed: `141300`
- Primary outer folds: five
- Primary inner folds: five
- Primary spatial block size: 1 degree
- Primary exclusion buffer: 20 km
- Bootstrap repetitions: 2,000 spatial-block repetitions
- Primary AoA setting: 15 retained dimensions and the 95th-percentile dissimilarity threshold

The sensitivity outputs in `model_outputs/manuscript_v6_audit_resolutions/`, `model_outputs/manuscript_v6_partition_sensitivity/`, and `model_outputs/manuscript_v6_spatial_design_sensitivity/` document the alternative settings reported in the manuscript and supplement.

## Limits of redistribution

The repository does not grant new rights to third-party datasets. Users must obtain source products under their original licences and terms. Model outputs and sample-level tables should be interpreted as research products for reproducing the published analyses, not as operational hazard or risk estimates.

