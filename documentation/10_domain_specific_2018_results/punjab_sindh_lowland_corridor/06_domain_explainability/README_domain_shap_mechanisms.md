# Domain-Wise SHAP and Mechanism Outputs: Punjab-Sindh lowland corridor

These outputs explain original conditioning factors for the XGBoost base learner inside each feature set. They complement, but do not replace, the main Spatial-CV Stacked Ensemble results.

fused leave-one-domain-out AUC = 0.990; fused AOA coverage = 98.2%. Dominant mechanism groups are AlphaEarth embeddings (60.6%), Topography/morphometry (21.7%). Top fused factors are AlphaEarth A16, Distance to roads, AlphaEarth A55, Profile curvature, Terrain ruggedness.

## Files

- `domain_treeshap_importance_all_feature_sets.csv`
- `domain_treeshap_group_importance_all_feature_sets.csv`
- `domain_treeshap_importance_*.png`
- `domain_treeshap_beeswarm_*.png`
- `domain_mechanism_group_shap_heatmap.png`
