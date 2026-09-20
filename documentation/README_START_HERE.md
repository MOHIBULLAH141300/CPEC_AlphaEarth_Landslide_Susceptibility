# Start Here: 2018 V3 Landslide Susceptibility Results

This is the clean, supervisor-facing package for the 2018 V3 CPEC landslide susceptibility workflow.

Use this folder first. The rest of `D:\DING PROJECT` remains as background working material and should only be opened when you need to trace an intermediate file.

## What This Package Contains

1. `00_SUPERVISOR_SUMMARY_2018_V3.md`
   - Plain-language summary of the final 2018 V3 method, model design, factors, and key results.

2. `01_methods_and_decisions`
   - Teacher suggestions, literature checks, factor decisions, VIF/multicollinearity report, model method reports, raster workflow, transferability reports, area-of-applicability report, and high-impact output report.

3. `02_model_performance_tables`
   - Clean supervisor tables plus full model metrics, fold metrics, ROC/PR curve coordinates, calibration table, and stacked ensemble meta-coefficients.

4. `03_figures`
   - Publication-ready PNG figures:
   - ROC, PR, calibration, stacked meta-coefficients.
   - SHAP importance and beeswarm figures.
   - 250 m uncertainty, AlphaEarth added-value, priority-zone, and road hotspot maps.
   - Administrative transferability and area-of-applicability/transfer-confidence figures.
   - Final reliability-aware susceptibility and road hotspot figures.

5. `04_probability_maps_250m`
   - Primary 250 m stacked probability GeoTIFFs.
   - Diagnostic 250 m rasters: mean probability, disagreement, reliability, AlphaEarth added-value, priority zones, reliable/uncertain high-probability masks.
   - Transfer-confidence and area-of-applicability GeoTIFFs for Conventional, AlphaEarth Embeddings, and Conventional + AlphaEarth Embeddings.
   - Final reliability-aware susceptibility, reliable high, uncertain high, and field-verification priority rasters.

6. `05_road_exposure_outputs`
   - Road exposure summary, all road segments, top 50 hotspot segments, GeoJSON, and GPKG outputs.

7. `06_samples_factors_and_vif`
   - V3 sample table, final factor list, readable factor labels, VIF tables, study-area boundary copy, and the local V3 factor stack QA.

8. `07_reproducible_scripts`
   - Main scripts used to build the V3 sample table, VIF diagnostics, model family, maps, and high-impact 250 m outputs.

9. `08_model_files_optional_background`
   - Final trained V3 model files. You usually do not need to open these unless reproducing predictions.

10. `99_background_index`
   - Pointers to original project folders.

11. `09_temporal_extension_plan`
   - Next-step plan for extending the 2018 V3 baseline into 2017-2024 annual dynamic-factor susceptibility maps.
   - Includes the temporal inventory audit, annual factor manifest, and year coverage matrix.

12. `10_domain_specific_2018_results`
   - One clean folder per transfer domain: Kashgar, Gilgit-Baltistan, KP-AJK, Balochistan, and Punjab-Sindh lowland corridor.
   - Each domain includes clipped 2018 probability rasters, transfer-confidence/reliability rasters, leave-one-domain-out metrics, road exposure tables, domain-wise TreeSHAP/factor-importance figures, VIF diagnostics, figures, and a domain README.

## Clean Public Names To Use

Use these names in the report and discussion:

| Internal idea | Public name |
| --- | --- |
| `conventional_v3` | Conventional |
| `alphaearth_embeddings` | AlphaEarth Embeddings |
| `conventional_v3_alphaearth_embeddings` | Conventional + AlphaEarth Embeddings |
| `stacked_l2_logistic` | Spatial-CV Stacked Ensemble |

Do not use "Reduced" in final names.

## Most Important Files

- Supervisor summary: `00_SUPERVISOR_SUMMARY_2018_V3.md`
- Clean performance summary: `02_model_performance_tables\clean_model_performance_summary_for_supervisor.csv`
- Clean pooled AUC summary: `02_model_performance_tables\clean_pooled_auc_summary_for_supervisor.csv`
- Readable factor list: `06_samples_factors_and_vif\readable_conventional_factor_list.csv`
- Main high-impact output report: `01_methods_and_decisions\stacked_250m_uncertainty_added_value_road_exposure_2026-05-03.md`
- Teacher-sent literature methodology review: `01_methods_and_decisions\teacher_sent_literature_methodology_review_2026-05-10.md`
- Subdomain definition / reviewer justification: `01_methods_and_decisions\subdomain_definition_and_reviewer_justification_2026-05-10.md`
- Administrative transferability report: `01_methods_and_decisions\v3_admin_transferability_domain_tests_report_2026-05-10.md`
- Area-of-applicability / transfer-confidence report: `01_methods_and_decisions\cpec_2018_area_of_applicability_transfer_confidence_report.md`
- Reliability-aware final output report: `01_methods_and_decisions\cpec_2018_reliability_aware_susceptibility_and_exposure_report.md`
- Conventional reliability-weighted raster diagnosis: `01_methods_and_decisions\conventional_reliability_weighted_raster_diagnosis_2026-05-10.md`
- Temporal extension plan: `09_temporal_extension_plan\annual_dynamic_susceptibility_extension_plan_2017_2024_2026-05-04.md`
- Better temporal evaluation strategy: `09_temporal_extension_plan\better_temporal_evaluation_strategy_2026-05-10.md`
- Domain-specific 2018 result index: `10_domain_specific_2018_results\README_DOMAIN_RESULTS_START_HERE.md`
- Domain comparison and mechanism report: `10_domain_specific_2018_results\00_domain_comparison_summary\domainwise_mechanism_and_transferability_interpretation_report.md`
- Primary probability rasters: `04_probability_maps_250m\01_primary_stacked_maps`
- Uncertainty and road-exposure rasters/tables: `04_probability_maps_250m\02_uncertainty_added_value_reliability` and `05_road_exposure_outputs`
- Transfer-confidence rasters: `04_probability_maps_250m\03_transfer_confidence_area_of_applicability`
- Final reliability-aware rasters: `04_probability_maps_250m\04_reliability_aware_final_outputs`

## Storage Note

Large raster and model files in this package are hardlinks where possible. They appear here like normal files but do not duplicate disk space on the D drive. Do not delete the original project folders until the full thesis/paper workflow is finished.
