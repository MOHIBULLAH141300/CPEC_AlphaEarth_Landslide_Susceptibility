# 2018 v3 Model Family Report

## Literature And Diagnostics Gate

- v3 uses the literature-supported corrected factor set.
- Missing road, river/stream, fault, lithology, terrain morphology, soil, and seismic-background factors were added before modelling.
- Final conventional v3 factors were selected after multicollinearity reduction; all final selected VIF values are below 4.

## Conventional v3 Factors

- `elevation_m`
- `slope_deg`
- `aspect_deg`
- `rain_monsoon_total`
- `rain_max_1day`
- `ndvi_median`
- `ndvi_amplitude`
- `modis_lc_type1`
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

## Top Results By Mean PR-AUC

| Rank | Feature set | Model | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Conventional v3 + AlphaEarth Embeddings | XGBoost | 0.960 +/- 0.035 | 0.973 +/- 0.017 | 0.907 +/- 0.059 | 0.890 +/- 0.079 | 0.062 +/- 0.025 |
| 2 | Conventional v3 + AlphaEarth Embeddings | CatBoost | 0.959 +/- 0.041 | 0.972 +/- 0.019 | 0.900 +/- 0.071 | 0.880 +/- 0.098 | 0.062 +/- 0.028 |
| 3 | Conventional v3 + AlphaEarth Embeddings | LightGBM | 0.957 +/- 0.039 | 0.971 +/- 0.018 | 0.822 +/- 0.088 | 0.772 +/- 0.150 | 0.066 +/- 0.029 |
| 4 | Conventional v3 + AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.957 +/- 0.037 | 0.972 +/- 0.018 | 0.914 +/- 0.056 | 0.897 +/- 0.076 | 0.060 +/- 0.026 |
| 5 | Conventional v3 + AlphaEarth Embeddings | Random Forest | 0.950 +/- 0.040 | 0.968 +/- 0.018 | 0.898 +/- 0.052 | 0.881 +/- 0.072 | 0.077 +/- 0.017 |
| 6 | Conventional v3 + AlphaEarth Embeddings | Extra Trees | 0.948 +/- 0.041 | 0.966 +/- 0.020 | 0.847 +/- 0.064 | 0.814 +/- 0.088 | 0.085 +/- 0.018 |
| 7 | Conventional v3 | Extra Trees | 0.948 +/- 0.037 | 0.962 +/- 0.018 | 0.798 +/- 0.070 | 0.740 +/- 0.109 | 0.105 +/- 0.027 |
| 8 | Conventional v3 | CatBoost | 0.946 +/- 0.032 | 0.960 +/- 0.017 | 0.837 +/- 0.056 | 0.802 +/- 0.078 | 0.089 +/- 0.029 |
| 9 | Conventional v3 | Spatial-CV Stacked Ensemble | 0.945 +/- 0.035 | 0.961 +/- 0.016 | 0.895 +/- 0.026 | 0.878 +/- 0.041 | 0.075 +/- 0.015 |
| 10 | Conventional v3 | Random Forest | 0.944 +/- 0.033 | 0.961 +/- 0.013 | 0.858 +/- 0.035 | 0.830 +/- 0.057 | 0.089 +/- 0.016 |
| 11 | Conventional v3 | XGBoost | 0.942 +/- 0.035 | 0.958 +/- 0.018 | 0.853 +/- 0.047 | 0.825 +/- 0.067 | 0.092 +/- 0.031 |
| 12 | Conventional v3 + AlphaEarth Embeddings | Logistic Regression (L2) | 0.940 +/- 0.058 | 0.959 +/- 0.029 | 0.877 +/- 0.072 | 0.854 +/- 0.102 | 0.078 +/- 0.032 |
| 13 | Conventional v3 | LightGBM | 0.938 +/- 0.041 | 0.956 +/- 0.017 | 0.801 +/- 0.055 | 0.748 +/- 0.089 | 0.099 +/- 0.022 |
| 14 | AlphaEarth Embeddings | XGBoost | 0.923 +/- 0.056 | 0.948 +/- 0.025 | 0.876 +/- 0.043 | 0.853 +/- 0.065 | 0.087 +/- 0.020 |
| 15 | AlphaEarth Embeddings | LightGBM | 0.922 +/- 0.058 | 0.946 +/- 0.025 | 0.830 +/- 0.057 | 0.794 +/- 0.079 | 0.090 +/- 0.024 |
| 16 | AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.922 +/- 0.059 | 0.951 +/- 0.025 | 0.879 +/- 0.038 | 0.860 +/- 0.067 | 0.082 +/- 0.020 |
| 17 | AlphaEarth Embeddings | CatBoost | 0.921 +/- 0.063 | 0.946 +/- 0.029 | 0.873 +/- 0.048 | 0.851 +/- 0.070 | 0.085 +/- 0.020 |
| 18 | AlphaEarth Embeddings | Logistic Regression (L2) | 0.919 +/- 0.065 | 0.947 +/- 0.025 | 0.875 +/- 0.043 | 0.852 +/- 0.068 | 0.085 +/- 0.020 |

## Saved Outputs

- Model folder: `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv`
- Summary: `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv\v3_model_family_summary.csv`
- Fold metrics: `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv\v3_model_family_fold_metrics.csv`
- OOF predictions: `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv\v3_model_family_spatial_cv_predictions.csv`
- Meta coefficients: `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv\v3_model_family_meta_coefficients.csv`
