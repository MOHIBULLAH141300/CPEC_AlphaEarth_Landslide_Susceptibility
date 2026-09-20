# Domain-Wise SHAP and Mechanism Outputs: Kashgar (Xinjiang, China)

These outputs explain original conditioning factors for the XGBoost base learner inside each feature set. They complement, but do not replace, the main Spatial-CV Stacked Ensemble results.

fused leave-one-domain-out AUC = 0.965; fused AOA coverage = 98.9%. Dominant mechanism groups are AlphaEarth embeddings (53.8%), Proximity controls (20.7%). Top fused factors are Distance to roads, AlphaEarth A16, AlphaEarth A55, Valley depth, Elevation.

## Files

- `domain_treeshap_importance_all_feature_sets.csv`
- `domain_treeshap_group_importance_all_feature_sets.csv`
- `domain_treeshap_importance_*.png`
- `domain_treeshap_beeswarm_*.png`
- `domain_mechanism_group_shap_heatmap.png`
