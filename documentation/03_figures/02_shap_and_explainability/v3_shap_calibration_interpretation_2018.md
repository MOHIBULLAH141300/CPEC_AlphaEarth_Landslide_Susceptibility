# v3 SHAP And Calibration Interpretation

## Literature/Methods Gate

After model evaluation, the next standard step is explainability and calibration before final raster export. TreeSHAP supports interpretation of the best tree model, and calibration/Brier score checks whether probabilities are reliable enough for map use.

## Top v3 Fused XGBoost SHAP Variables

| Rank | Feature | Mean |SHAP| | Mean SHAP |
|---:|---|---:|---:|
| 1 | Distance to roads | 1.57983 | -0.32784 |
| 2 | AlphaEarth A16 | 0.74423 | -0.16085 |
| 3 | AlphaEarth A55 | 0.43706 | 0.00133 |
| 4 | Valley depth | 0.42266 | -0.07731 |
| 5 | Profile curvature | 0.32541 | -0.07365 |
| 6 | Elevation | 0.29520 | 0.00324 |
| 7 | Terrain ruggedness | 0.28826 | -0.04004 |
| 8 | AlphaEarth A44 | 0.26057 | -0.07507 |
| 9 | Distance to rivers/streams | 0.22966 | 0.03295 |
| 10 | Distance to active faults | 0.21665 | -0.02298 |
| 11 | Plan curvature | 0.18861 | 0.00230 |
| 12 | AlphaEarth A25 | 0.16354 | -0.00089 |
| 13 | AlphaEarth A13 | 0.13418 | -0.03411 |
| 14 | AlphaEarth A46 | 0.12555 | -0.00214 |
| 15 | AlphaEarth A26 | 0.12500 | -0.00848 |

## Calibration Ranking By Brier Score

| Model | Brier score |
|---|---:|
| Fused v3 Stacked Ensemble | 0.06098 |
| Fused v3 XGBoost | 0.06314 |
| Fused v3 CatBoost | 0.06368 |

## Stacked Ensemble Meta-Coefficients

| Base model | Mean coefficient |
|---|---:|
| Extra Trees | 1.2639 |
| XGBoost | 0.8269 |
| CatBoost | 0.5532 |
| Logistic Regression (L2) | 0.4050 |
| Random Forest | 0.2913 |
| LightGBM | 0.1104 |

## Saved Figures

- SHAP importance: `D:\DING PROJECT\05_reports\figures\figure_v3_fused_xgboost_shap_importance.png`
- SHAP beeswarm: `D:\DING PROJECT\05_reports\figures\figure_v3_fused_xgboost_shap_beeswarm.png`
- Calibration: `D:\DING PROJECT\05_reports\figures\figure_v3_fused_model_calibration.png`
- Meta-coefficients: `D:\DING PROJECT\05_reports\figures\figure_v3_fused_stacked_meta_coefficients.png`
