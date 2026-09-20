# Teacher Suggestions Compliance Check

Date checked: 2026-05-02

Teacher suggestion file:

- `C:\Users\Administrator\Desktop\cpec landslides\suggestions-renew.docx`

Extracted text copy:

- `D:\DING PROJECT\05_reports\teacher_suggestions_extracted_text.txt`

## Short conclusion

The current results are a strong and clean 2018 baseline experiment, especially for comparing:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

However, the current result package is not yet the full study requested by the teacher. The teacher is asking for a broader Q1-level study on spatial-temporal evolution of landslide susceptibility along CPEC, especially annual mapping after 2013, dynamic factors, scenario testing, and deeper methodological framing.

## Compliance checklist

| Teacher suggestion | Current status | Evidence in project | Remaining work |
|---|---|---|---|
| Upgrade topic toward spatial-temporal evolution of LSM along CPEC using ensemble machine learning with AlphaEarth Embeddings | Partially complete | 2018 baseline model comparison exists | Need annual post-2013 susceptibility maps and temporal evolution analysis |
| Explain motivation for ensemble machine learning from study-area challenges and method advantages | Partially complete | `cpec_kkh_data_literature_audit.md`, `reduced_strategy_2018_modelling_report.md` | Need stronger manuscript-ready motivation text linking CPEC scale, complex conditioning environment, AlphaEarth high-dimensional data, noisy inventory, and ensemble methods |
| Compare conventional factors, AlphaEarth-only, and fused AlphaEarth + conventional factors | Complete for 2018 | Metrics, ROC/PR curves, SHAP outputs | Extend yearly if annual study is pursued |
| Address multicollinearity and noisy factors | Complete for 2018 | `multicollinearity_assessment_2018.md`; VIF/correlation tables | None for 2018 baseline; repeat/justify if annual factor sets change |
| Clarify why Extra Trees and other base learners are used, and what information each model extracts | Partially complete | Models include Extra Trees, Random Forest, LightGBM, XGBoost, Logistic L2 | Need manuscript section explaining each model role; current final headline uses XGBoost rather than a formal stacked/meta-learning ensemble |
| Use ensemble machine learning / meta-learning to combine base models | Partially complete | Multiple ensemble learners trained and evaluated | A true stacking/blending meta-learner is not yet implemented as an official final result |
| Draw method framework / flowchart | Not complete | No framework/flowchart file found | Need create publication-ready workflow diagram |
| Use official data and correctly define study boundary | Mostly complete | GEE official boundary asset used: `projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area` | Confirm citation/source text in manuscript; optionally document geodata source more explicitly |
| Build dynamic susceptibility dataset since CPEC concept year 2013 | Not complete | Only annual-factor scripts/dry-run exist | Need annual samples/factors and maps for 2013 onward |
| Add event timestamps to landslide inventory | Not complete / unknown | Current sample table supports 2018 modelling; inventory timestamp handling not confirmed in current outputs | Need build/verify temporal landslide inventory with dates/year |
| Add PGA as a dynamic factor | Not complete | No PGA/earthquake files found | Need obtain/derive PGA or seismic hazard/scenario factor |
| Use rainfall dynamic factor construction based on recent literature | Partially complete | Rainfall factors included in 2018 conventional set | Need annual/extreme rainfall dynamic factors and literature-linked justification |
| Analyze spatial-temporal rules/patterns of susceptibility | Not complete | Only 2018 outputs found | Need annual maps, change analysis, regional/geomorphic unit summaries |
| Use Spatial Cross-Validation to prove large-area robustness across geomorphic units | Partially complete | Spatial CV exists | Need geomorphic-unit-specific robustness/evolution analysis |
| Scenario experiment for key infrastructure under 100-year earthquake and 100-year rainfall | Not complete | No infrastructure/scenario outputs found | Need infrastructure layers and extreme PGA/rainfall scenario maps |
| Discuss robustness vs generalization carefully | Not yet manuscript-level | Evaluation reports exist | Need text revision in manuscript/report |
| Read high-impact journals and improve writing style | Partially complete | Literature audit exists | Need update manuscript narrative and add current high-impact references |

## Current completed 2018 result components

Completed outputs include:

- Multicollinearity assessment
- Spatial cross-validation
- Evaluation metrics
- ROC-AUC and PR-AUC curves
- 250 m no-NoData probability rasters
- SHAP / TreeSHAP interpretation figures
- Raster QA report

Important folders:

- `D:\DING PROJECT\03_models\multicollinearity_assessment_2018`
- `D:\DING PROJECT\03_models\reduced_strategy_2018_spatial_cv`
- `D:\DING PROJECT\03_models\reduced_strategy_2018_metrics`
- `D:\DING PROJECT\03_models\reduced_strategy_2018_curves`
- `D:\DING PROJECT\03_models\shap_2018_tree_importance`
- `D:\DING PROJECT\04_maps\rasters_2018_probability`
- `D:\DING PROJECT\05_reports\figures`

## Recommended next steps

1. Create a publication-ready methodology framework diagram.
2. Convert the current 2018 baseline into the official baseline chapter/section.
3. Build annual dynamic factor stacks from 2013 onward.
4. Prepare/verify timestamped landslide inventory for temporal modelling.
5. Add PGA/seismic factor and extreme rainfall scenario factors.
6. Implement a formal stacked ensemble or clearly justify XGBoost as the final best-performing tree ensemble.
7. Produce annual susceptibility maps and spatial-temporal evolution analysis.
8. Add key infrastructure exposure/risk analysis for KKH, railway/oil-gas corridor, hydropower, and Gwadar-related assets if data are available.

## Housekeeping

The duplicate old raster is still locked by Windows and could not be removed:

- `D:\DING PROJECT\04_maps\rasters_2018_probability\cpec_2018_alphaearth_only_probability_250m_no_nodata.tif`

The correct current raster already exists:

- `D:\DING PROJECT\04_maps\rasters_2018_probability\cpec_2018_alphaearth_embeddings_probability_250m_no_nodata.tif`

This does not affect the clean checked results.

