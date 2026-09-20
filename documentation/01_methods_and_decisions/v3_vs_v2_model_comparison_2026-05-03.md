# v3 vs v2 Model Comparison

Date: 2026-05-03

## Purpose

This comparison checks whether the literature-corrected v3 factor set improved the 2018 CPEC landslide susceptibility models relative to the previous v2 workflow.

## Key result

The corrected v3 factors substantially improve the fused model results.

| Case | Feature set | Model | ROC-AUC | PR-AUC | Balanced accuracy | F1 | Brier |
|---|---|---|---:|---:|---:|---:|---:|
| v2 fused stacked | Conventional + AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.952 | 0.927 | 0.884 | 0.863 | 0.082 |
| v2 fused XGBoost | Conventional + AlphaEarth Embeddings | XGBoost | 0.950 | 0.927 | 0.878 | 0.857 | 0.086 |
| v3 fused stacked | Conventional v3 + AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.972 | 0.957 | 0.914 | 0.897 | 0.060 |
| v3 fused XGBoost | Conventional v3 + AlphaEarth Embeddings | XGBoost | 0.973 | 0.960 | 0.907 | 0.890 | 0.062 |

## Interpretation

The added literature-supported factors improved the model strongly:

- road proximity
- river/stream proximity
- fault proximity
- lithology/geology
- terrain morphology
- TWI
- valley depth
- soil type
- earthquake density

The best v3 model by PR-AUC is:

- Conventional v3 + AlphaEarth Embeddings
- XGBoost

The v3 stacked ensemble has slightly lower PR-AUC than v3 XGBoost, but better:

- balanced accuracy
- F1
- Brier score

This means v3 XGBoost is the strongest ranking model by PR-AUC, while the stacked ensemble is better calibrated and more balanced. The final map-model choice should be made after SHAP, calibration, and raster QA.

## Saved comparison table

- `D:\DING PROJECT\03_models\v3_model_family_2018_spatial_cv\v2_v3_key_model_comparison.csv`

## Current recommended next step

Run SHAP/TreeSHAP for the v3 fused XGBoost and v3 fused stacked/meta-ensemble candidates, then choose the final mapping model.

