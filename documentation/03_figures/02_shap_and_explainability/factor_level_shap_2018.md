# Factor-Level SHAP For 2018 Feature Sets

These figures explain the XGBoost base learner for each official feature set. They complement the stacked-ensemble meta-SHAP figures: meta-SHAP explains base-model contributions, while these plots explain original conditioning-factor contributions.

## Top Factors

| Feature set | Rank | Factor | Mean |SHAP| |
|---|---:|---|---:|
| Conventional | 1 | Distance to roads | 1.71784 |
| Conventional | 2 | Elevation | 1.03887 |
| Conventional | 3 | Valley depth | 0.86468 |
| Conventional | 4 | Terrain ruggedness | 0.77382 |
| Conventional | 5 | Profile curvature | 0.47774 |
| Conventional | 6 | Distance to active faults | 0.36144 |
| Conventional | 7 | Distance to rivers/streams | 0.35791 |
| Conventional | 8 | NDVI median | 0.25431 |
| Conventional | 9 | Aspect | 0.18944 |
| Conventional | 10 | Plan curvature | 0.17322 |
| Conventional | 11 | Max 1-day rainfall | 0.15040 |
| Conventional | 12 | NDVI amplitude | 0.13499 |
| Conventional | 13 | Lithology/geology | 0.11782 |
| Conventional | 14 | Monsoon rainfall | 0.10379 |
| Conventional | 15 | TWI | 0.09667 |
| AlphaEarth Embeddings | 1 | AlphaEarth A16 | 1.16726 |
| AlphaEarth Embeddings | 2 | AlphaEarth A55 | 0.52441 |
| AlphaEarth Embeddings | 3 | AlphaEarth A44 | 0.46959 |
| AlphaEarth Embeddings | 4 | AlphaEarth A60 | 0.24700 |
| AlphaEarth Embeddings | 5 | AlphaEarth A25 | 0.23183 |
| AlphaEarth Embeddings | 6 | AlphaEarth A17 | 0.21972 |
| AlphaEarth Embeddings | 7 | AlphaEarth A15 | 0.21318 |
| AlphaEarth Embeddings | 8 | AlphaEarth A34 | 0.20921 |
| AlphaEarth Embeddings | 9 | AlphaEarth A45 | 0.18141 |
| AlphaEarth Embeddings | 10 | AlphaEarth A13 | 0.16489 |
| AlphaEarth Embeddings | 11 | AlphaEarth A46 | 0.16346 |
| AlphaEarth Embeddings | 12 | AlphaEarth A56 | 0.16311 |
| AlphaEarth Embeddings | 13 | AlphaEarth A47 | 0.14853 |
| AlphaEarth Embeddings | 14 | AlphaEarth A06 | 0.13932 |
| AlphaEarth Embeddings | 15 | AlphaEarth A22 | 0.13339 |
| Conventional + AlphaEarth Embeddings | 1 | Distance to roads | 1.57756 |
| Conventional + AlphaEarth Embeddings | 2 | AlphaEarth A16 | 0.74453 |
| Conventional + AlphaEarth Embeddings | 3 | AlphaEarth A55 | 0.43724 |
| Conventional + AlphaEarth Embeddings | 4 | Valley depth | 0.42210 |
| Conventional + AlphaEarth Embeddings | 5 | Profile curvature | 0.32459 |
| Conventional + AlphaEarth Embeddings | 6 | Elevation | 0.29584 |
| Conventional + AlphaEarth Embeddings | 7 | Terrain ruggedness | 0.28872 |
| Conventional + AlphaEarth Embeddings | 8 | AlphaEarth A44 | 0.26166 |
| Conventional + AlphaEarth Embeddings | 9 | Distance to rivers/streams | 0.22738 |
| Conventional + AlphaEarth Embeddings | 10 | Distance to active faults | 0.21925 |
| Conventional + AlphaEarth Embeddings | 11 | Plan curvature | 0.18851 |
| Conventional + AlphaEarth Embeddings | 12 | AlphaEarth A25 | 0.16277 |
| Conventional + AlphaEarth Embeddings | 13 | AlphaEarth A13 | 0.13415 |
| Conventional + AlphaEarth Embeddings | 14 | AlphaEarth A46 | 0.12601 |
| Conventional + AlphaEarth Embeddings | 15 | AlphaEarth A09 | 0.12466 |

## Saved Figures

- SHAP beeswarm, Conventional: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_beeswarm_2018_conventional.png`
- SHAP beeswarm, AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_beeswarm_2018_alphaearth_embeddings.png`
- SHAP beeswarm, Conventional + AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_beeswarm_2018_conventional_alphaearth_embeddings.png`
- SHAP importance, Conventional: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_importance_2018_conventional.png`
- SHAP importance, AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_importance_2018_alphaearth_embeddings.png`
- SHAP importance, Conventional + AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_factor_shap_importance_2018_conventional_alphaearth_embeddings.png`
