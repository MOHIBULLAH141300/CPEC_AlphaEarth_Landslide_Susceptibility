# Supervisor Summary: 2018 V3 CPEC Landslide Susceptibility Workflow

## Study Goal

The 2018 V3 workflow builds a new landslide susceptibility baseline for the CPEC study area, with special relevance to the KKH corridor. It replaces the older reduced/preliminary workflow and should be treated as the current final 2018 result package.

## Study Area And Data

- Study area: official CPEC boundary containing Pakistan and Kashgar, Xinjiang.
- Sample table: `3316` observations from the 2018 landslide susceptibility sample dataset.
- Validation: 5-fold spatial block cross-validation using `spatial_fold_5`.
- Main output grid: 250 m probability rasters, clipped/masked to the official study area.

## Factor Sets

Three feature sets were evaluated:

| Feature set | Number of predictors | Purpose |
| --- | ---: | --- |
| Conventional | 19 | Physically interpretable landslide conditioning factors |
| AlphaEarth Embeddings | 64 | Foundation-model remote-sensing representation from annual 2018 embeddings |
| Conventional + AlphaEarth Embeddings | 83 | Fused physical + foundation-model feature set |

The final Conventional factor list includes:

- Topography: elevation, slope, aspect, valley depth, terrain ruggedness index, TWI, profile curvature, plan curvature.
- Climate/vegetation: monsoon rainfall total, maximum 1-day rainfall, median NDVI, NDVI amplitude.
- Proximity/geology: distance to roads, distance to rivers/streams, distance to active faults, lithology, soil type.
- Land cover and seismicity: land-cover class, earthquake density.

## Multicollinearity Decision

The V3 multicollinearity check removed problematic redundancy before modelling. The final 19 Conventional factors all had acceptable VIF values, with the largest final VIF below 4. This is supervisor/reviewer friendly because it shows that highly redundant factors were not blindly kept.

Main VIF file:

- `06_samples_factors_and_vif\v3_final_selected_vif.csv`

Readable factor labels:

- `06_samples_factors_and_vif\readable_conventional_factor_list.csv`

## Models Evaluated

For each feature set, these models were tested:

- Logistic Regression (L2)
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Spatial-CV Stacked Ensemble

The stacked model used out-of-fold predictions from spatial CV base learners, so it is more defensible than a simple random split ensemble.

## Key Model Results

Best pooled AUC results for the stacked-model comparison:

| Feature set | Model | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: |
| Conventional + AlphaEarth Embeddings | XGBoost | 0.9733 | 0.9684 |
| Conventional + AlphaEarth Embeddings | CatBoost | 0.9726 | 0.9683 |
| Conventional + AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.9689 | 0.9612 |
| Conventional | Spatial-CV Stacked Ensemble | 0.9607 | 0.9544 |
| AlphaEarth Embeddings | Spatial-CV Stacked Ensemble | 0.9510 | 0.9410 |

Interpretation:

- The fused feature set performs best because conventional factors provide physical landslide controls while AlphaEarth adds annual remote-sensing/context information.
- AlphaEarth alone is useful but weaker than the fused set because embeddings do not explicitly encode every geomorphic or proximity factor.
- XGBoost/CatBoost slightly outperform the stacked model for the fused set, but the stacked model remains valuable for the paper because it provides a systematic meta-learning comparison across feature sets.

Clean performance tables:

- `02_model_performance_tables\clean_model_performance_summary_for_supervisor.csv`
- `02_model_performance_tables\clean_pooled_auc_summary_for_supervisor.csv`

## Probability Maps

Primary 250 m stacked probability rasters are in:

- `04_probability_maps_250m\01_primary_stacked_maps`

The three final map products are:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

The fused map, Conventional + AlphaEarth Embeddings, is the main planning raster.

## High-Impact Added Results

The package also includes 250 m diagnostic outputs:

- Mean probability map.
- Model disagreement / uncertainty proxy.
- Reliability score map.
- AlphaEarth added-value map.
- Reliable high-probability mask.
- Uncertain high-probability mask.
- Optional quantile priority-zone map.
- CPEC/KKH road exposure and hotspot segments.

Main report:

- `01_methods_and_decisions\stacked_250m_uncertainty_added_value_road_exposure_2026-05-03.md`

Key road-exposure result:

- All clipped roads: about 17,066 km.
- KKH proxy routes, identified as N35 / 314: about 1,448 km.
- KKH proxy weighted length above fused P90 probability: about 1,077.5 km.

## Explainability Outputs

SHAP outputs are stored in:

- `03_figures\02_shap_and_explainability`

Use the factor-level SHAP figures for reporting because they are easier to explain than raw embedding-band plots. The figures group many embedding bands into readable total/remaining embedding contribution summaries.

## Recommended Story For Supervisor

The 2018 V3 workflow is no longer just a model-comparison exercise. It now has:

- Literature-supported conditioning factors.
- VIF-tested factor selection.
- Spatial cross-validation.
- Multiple machine-learning and ensemble models.
- AlphaEarth foundation embedding comparison.
- Explainability using SHAP.
- Uncertainty and reliability maps.
- CPEC/KKH road exposure hotspot outputs.

This gives the study a stronger publication story: physical modelling, foundation-model innovation, spatial robustness, explainability, and corridor infrastructure relevance.

## Temporal Extension Status

The current 2018 V3 result should be called a **2018 dynamic-factor landslide susceptibility baseline**.

The next planned extension is **annual dynamic-factor susceptibility mapping for 2017-2024**. The temporal extension plan is stored in:

- `09_temporal_extension_plan`

Important caution:

- The current CPEC inventory is mostly geometry-only.
- Only 1 of 1508 V3 positive samples has an explicit event year.
- The local NASA HMA catalog provides 185 dated events inside the official study area through 2018, so it can help enrich/validate early annual work.
- For 2019-2024, true temporal validation will require more dated inventory data or manual event-date enrichment.

## Important Caution

Road distance is included as a conditioning factor, so road exposure outputs should be described as planning-priority overlays rather than independent validation. If needed later, a road-distance ablation model can be added to test how much the hotspot map depends on the road-proximity predictor.
