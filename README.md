# CPEC AlphaEarth landslide-susceptibility analysis

Data, code, derived results, and supplementary material for:

> **Evaluating AlphaEarth embeddings for cross-regional landslide susceptibility mapping: A transferability-aware framework for the China-Pakistan Economic Corridor**

Mohib Ullah, Mingtao Ding, Qiang Xue, Ying Dong, and Zhenhong Li
Submitted to *Geoscience Frontiers*

## Study purpose

The study evaluates three predictor configurations for landslide-susceptibility mapping across the China-Pakistan Economic Corridor (CPEC):

- 19 conventional conditioning factors;
- 64 AlphaEarth embedding axes; and
- their fusion.

The PPS-CDT-PDS framework distinguishes three questions that should not be conflated:

1. **Pooled predictive skill (PPS):** performance under fully nested, buffered spatial cross-validation.
2. **Cross-domain transferability (CDT):** performance when a complete geographic domain is excluded from model fitting.
3. **Predictor-domain support (PDS):** whether target conditions are represented by the source-domain predictor space.

The main scientific result is methodological rather than a large accuracy gain. Fusion increased pooled ROC-AUC from 0.960 to 0.967, while statistically resolved transfer improvement was geographically selective and predictor support was narrower than for conventional predictors.

## Repository map

```text
Supplementary_Data_S1.xlsx     Submission supplement and complete sensitivity grids
scripts/
  python/                      Analysis and figure-generation scripts
  gee/                         Google Earth Engine preparation/export scripts
model_outputs/                 Derived metrics, predictions, manifests, and diagnostics
documentation/                 Organised methods, figures, tables, and supporting outputs
```

The most relevant result directories are:

- `model_outputs/manuscript_v3_nested_spatial_cv/`
- `model_outputs/manuscript_v3_leave_one_domain_out/`
- `model_outputs/manuscript_v3_harmonised_aoa/`
- `model_outputs/manuscript_v3_robustness_experiments/`
- `model_outputs/manuscript_v3_spatial_block_comparisons/`
- `model_outputs/manuscript_v6_audit_resolutions/`
- `model_outputs/manuscript_v6_partition_sensitivity/`
- `model_outputs/manuscript_v6_spatial_design_sensitivity/`

Core analysis entry points include:

- `scripts/python/run_manuscript_v3_nested_spatial_cv.py`
- `scripts/python/run_manuscript_v3_leave_one_domain_out.py`
- `scripts/python/run_manuscript_v3_harmonised_aoa.py`
- `scripts/python/run_manuscript_v3_robustness_experiments.py`
- `scripts/python/run_manuscript_v3_spatial_block_model_comparisons.py`
- `scripts/python/run_manuscript_v6_partition_sensitivity.py`
- `scripts/python/run_manuscript_v6_spatial_design_sensitivity.py`
- `scripts/python/run_manuscript_v3_road_exposure.py`

## Reproducibility scope

This repository is a publication archive of the executed workflow and its derived outputs. The analysis used a deterministic random seed of `141300`, fixed spatial-block assignments, five outer and five inner spatial folds, and a 20 km exclusion buffer in the primary design.

Some archived scripts retain absolute workstation paths from the executed analysis. These paths do not contain credentials, but they must be replaced with local paths before rerunning the workflow. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the expected directory layout, dependency groups, and execution order.

Raw satellite, terrain, climate, land-cover, geological, seismic, road, and inventory datasets are not all redistributed here because their licences and download services differ. The manuscript and Table 1 identify the authoritative sources. The primary CPEC slope-failure inventory is available from Science Data Bank (Yi et al., 2021), and the other public predictors are available through their respective services.

## Supplementary data

`Supplementary_Data_S1.xlsx` contains the deterministic alternative partition, all alternative-domain leave-one-domain-out results and paired contrasts, and the complete area-of-applicability sensitivity grid used in the manuscript.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The article DOI will be added after publication.

## Licence

Unless a file states otherwise, the original repository content is distributed under the [Creative Commons Attribution 4.0 International licence](LICENSE). Third-party datasets remain subject to their source licences and are not relicensed by this repository.

## Contact

Zhenhong Li: `zhenhong.li@chd.edu.cn`
Qiang Xue: `xueqiang_79@163.com`
