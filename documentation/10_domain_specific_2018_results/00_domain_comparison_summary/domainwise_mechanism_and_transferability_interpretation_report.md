# Domain-Wise Mechanism and Transferability Interpretation

## What Was Done

The CPEC-wide 2018 V3 Spatial-CV Stacked Ensemble remains the main prediction model. For domain-wise factor mechanisms, TreeSHAP was computed for the XGBoost base learner inside each official feature set. This keeps the modelling strategy comparable across domains while still showing why the model behaves differently in each domain.

## What This Adds Scientifically

- It tests model generalization using leave-one-domain-out transferability.
- It shows where the model is inside or outside its Area of Applicability.
- It identifies which controlling mechanisms dominate each CPEC subdomain.
- It avoids weak separate regional retraining while preserving domain-level interpretation.
- It adds domain-wise VIF diagnostics as an appendix check, while keeping global VIF as the final factor-selection gate.

## Domain Mechanism Summary

- **Kashgar (Xinjiang, China)**: fused leave-one-domain-out AUC = 0.965; fused AOA coverage = 98.9%. Dominant mechanism groups are AlphaEarth embeddings (53.8%), Proximity controls (20.7%). Top fused factors are Distance to roads, AlphaEarth A16, AlphaEarth A55, Valley depth, Elevation.
- **Gilgit-Baltistan**: fused leave-one-domain-out AUC = 0.979; fused AOA coverage = 97.8%. Dominant mechanism groups are AlphaEarth embeddings (55.5%), Proximity controls (19.6%). Top fused factors are Distance to roads, AlphaEarth A16, Valley depth, AlphaEarth A55, Elevation.
- **KP-AJK**: fused leave-one-domain-out AUC = 0.976; fused AOA coverage = 90.7%. Dominant mechanism groups are AlphaEarth embeddings (57.3%), Proximity controls (16.9%). Top fused factors are Distance to roads, AlphaEarth A16, AlphaEarth A55, Valley depth, AlphaEarth A44.
- **Balochistan**: fused leave-one-domain-out AUC = 0.913; fused AOA coverage = 98.9%. Dominant mechanism groups are AlphaEarth embeddings (57.8%), Topography/morphometry (18.4%). Top fused factors are Distance to roads, AlphaEarth A16, Valley depth, Profile curvature, AlphaEarth A55.
- **Punjab-Sindh lowland corridor**: fused leave-one-domain-out AUC = 0.990; fused AOA coverage = 98.2%. Dominant mechanism groups are AlphaEarth embeddings (60.6%), Topography/morphometry (21.7%). Top fused factors are AlphaEarth A16, Distance to roads, AlphaEarth A55, Profile curvature, Terrain ruggedness.

## Main Tables

- `domain_transferability_mechanism_comparison_table.csv`
- `domainwise_treeshap_importance_all_feature_sets.csv`
- `domainwise_treeshap_group_importance_all_feature_sets.csv`
- `domainwise_conventional_vif_diagnostics.csv`

## Main Figures

- `figure_domain_transferability_reliability_comparison_heatmap.png`
- `figure_domain_top_factor_shap_heatmap_fused.png`
- `figure_domain_mechanism_group_heatmap_fused.png`
- `figure_domainwise_conventional_vif_diagnostics.png`

## VIF Interpretation

The final factor set is still controlled by the global V3 multicollinearity
gate. Domain-wise VIF is included only to diagnose local redundancy after
subsetting the study area. Local VIF can rise in smaller domains because the
environmental range is narrower; it should not be used to select a different
factor set for each domain.

## Suggested Paper Wording

A domain-wise transferability and explanation analysis was conducted to test whether the CPEC-wide model generalized across politically and physiographically contrasting subdomains. Instead of fitting independent regional models, which would reduce sample size and weaken comparability, we used leave-one-domain-out validation, Area of Applicability mapping, and domain-specific TreeSHAP summaries of the XGBoost base learner. This design quantifies both predictive transferability and spatially varying conditioning mechanisms.
