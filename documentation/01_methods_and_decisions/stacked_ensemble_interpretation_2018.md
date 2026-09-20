# Stacked Ensemble Interpretation For Three Feature Sets

Public labels used in figures and reports:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

SHAP is computed at the stacked meta-learner level. The SHAP variables are the base-model probability outputs, not the original terrain/rainfall/embedding predictors.

## Top Meta-SHAP Base Models

| Feature set | Rank | Base model | Mean |SHAP| |
|---|---:|---|---:|
| Conventional | 1 | Random Forest | 1.52143 |
| Conventional | 2 | Extra Trees | 1.51262 |
| Conventional | 3 | CatBoost | 0.14289 |
| Conventional | 4 | XGBoost | 0.11056 |
| Conventional | 5 | Logistic Regression | 0.10854 |
| Conventional | 6 | LightGBM | 0.08259 |
| AlphaEarth Embeddings | 1 | Logistic Regression | 1.14698 |
| AlphaEarth Embeddings | 2 | Extra Trees | 1.04738 |
| AlphaEarth Embeddings | 3 | CatBoost | 0.80133 |
| AlphaEarth Embeddings | 4 | Random Forest | 0.55292 |
| AlphaEarth Embeddings | 5 | XGBoost | 0.33822 |
| AlphaEarth Embeddings | 6 | LightGBM | 0.09827 |
| Conventional + AlphaEarth Embeddings | 1 | Extra Trees | 1.23381 |
| Conventional + AlphaEarth Embeddings | 2 | XGBoost | 0.87860 |
| Conventional + AlphaEarth Embeddings | 3 | CatBoost | 0.53575 |
| Conventional + AlphaEarth Embeddings | 4 | Logistic Regression | 0.37607 |
| Conventional + AlphaEarth Embeddings | 5 | Random Forest | 0.24684 |
| Conventional + AlphaEarth Embeddings | 6 | LightGBM | 0.06509 |

## Saved Figures

- SHAP importance, Conventional: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_importance_2018_conventional.png`
- SHAP beeswarm, Conventional: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_beeswarm_2018_conventional.png`
- Meta-coefficients, Conventional: `D:\DING PROJECT\05_reports\figures\figure_stacked_meta_coefficients_2018_conventional.png`
- SHAP importance, AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_importance_2018_alphaearth_embeddings.png`
- SHAP beeswarm, AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_beeswarm_2018_alphaearth_embeddings.png`
- Meta-coefficients, AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_meta_coefficients_2018_alphaearth_embeddings.png`
- SHAP importance, Conventional + AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_importance_2018_conventional_alphaearth_embeddings.png`
- SHAP beeswarm, Conventional + AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_shap_beeswarm_2018_conventional_alphaearth_embeddings.png`
- Meta-coefficients, Conventional + AlphaEarth Embeddings: `D:\DING PROJECT\05_reports\figures\figure_stacked_meta_coefficients_2018_conventional_alphaearth_embeddings.png`
- Calibration curve: `D:\DING PROJECT\05_reports\figures\figure_stacked_calibration_2018_all_feature_sets.png`
