# CPEC AlphaEarth Landslide Susceptibility Mapping - Data and Code Repository

This repository contains the analysis code, data, and supplementary materials for the manuscript:

**"A transferability-aware framework for evaluating AlphaEarth embeddings in cross-regional landslide susceptibility mapping"**

**Authors:** Mohib Ullah, Mingtao Ding, Qiang Xue, Ying Dong, Zhenhong Li  
**Journal:** Geoscience Frontiers  
**DOI:** [To be assigned upon publication]

---

## Overview

This study develops a three-dimensional evaluation framework (PPS–CDT–PDS) to assess AlphaEarth embeddings for landslide susceptibility mapping across the China–Pakistan Economic Corridor (CPEC). The repository includes:

- Analysis code for spatial cross-validation and stacked ensemble modeling
- Fixed fold assignments for reproducibility
- Derived sample tables and model manifests
- Bootstrap summaries and figure-generation scripts
- Supplementary data (domain partition rules, LODO estimates, AoA sensitivity grid)

---

## Directory Structure

```
├── 06_scripts/                    # Analysis scripts
│   ├── python/                    # Python modeling scripts
│   │   └── run_2018_v3_model_family.py
│   └── [other analysis scripts]
├── 03_models/                     # Model outputs and results
│   ├── v3_model_family_2018_spatial_cv/
│   │   ├── v3_model_family_summary.csv
│   │   ├── v3_model_family_meta_coefficients.csv
│   │   ├── v3_model_family_fold_metrics.csv
│   │   └── [other model outputs]
│   └── stacked_ensemble_paper_outputs/
│       └── stacked_ensemble_metrics_pooled.csv
├── 00_READ_ME_FIRST_2018_V3_RESULTS/  # Comprehensive results documentation
│   ├── 06_samples_factors_and_vif/
│   │   └── cpec_2018_lsm_samples_v3.csv
│   ├── 10_domain_specific_2018_results/
│   └── [other results directories]
└── Supplementary_Data_S1.xlsx      # Domain partition rules and LODO results
```

---

## Key Files

### Data Files
- `cpec_2018_lsm_samples_v3.csv` - Final sample table with 3,316 observations (45.5% failures, 54.5% controls)
- `Supplementary_Data_S1.xlsx` - Deterministic partition rules, LODO estimates, AoA sensitivity grid

### Model Outputs
- `v3_model_family_summary.csv` - Pooled performance metrics across feature sets and models
- `v3_model_family_meta_coefficients.csv` - Stacked ensemble meta-learner coefficients per fold
- `v3_model_family_fold_metrics.csv` - Detailed fold-level performance metrics

### Analysis Scripts
- `run_2018_v3_model_family.py` - Main script for spatial cross-validation and ensemble training

---

## Reproducibility

### Fixed Fold Assignments
Spatial 5-fold cross-validation uses deterministic block assignment:
- Block assignment: `(|31 × longitude_block + 17 × latitude_block| mod 5) + 1`
- 20 km exclusion buffer enforced between training and test folds
- Random seed: 141300 (for sample reproducibility)

### Model Configuration
- Base learners: Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost
- Meta-learner: L2-regularized logistic regression
- Hyperparameters: See `run_2018_v3_model_family.py` for complete specifications

---

## Citation

If you use this code or data, please cite our manuscript:

```
Ullah, M., Ding, M., Xue, Q., Dong, Y., & Li, Z. (2026). 
A transferability-aware framework for evaluating AlphaEarth embeddings 
in cross-regional landslide susceptibility mapping. 
Geoscience Frontiers.
```

---

## License

This repository is made available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

**Corresponding Author:**  
Zhenhong Li  
Email: zhenhong.li@chd.edu.cn  
Institution: Chang'an University, Xi'an 710054, China

---

## Acknowledgements

This work was supported by:
- Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (Grant JYB2025XDXM104)
- National Natural Science Foundation of China (Grant 42374027)
- Fundamental Research Funds for the Central Universities (Grant 300112266411)

---

## Data Sources

The primary CPEC landslide and rockfall inventory is available from Science Data Bank (Yi et al., 2021).  
Copernicus DEM, CHIRPS, MODIS, and Google Satellite Embedding products are available through their respective public data services.
