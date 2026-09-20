# Domain-Wise SHAP and Mechanism Outputs: KP-AJK

These outputs explain original conditioning factors for the XGBoost base learner inside each feature set. They complement, but do not replace, the main Spatial-CV Stacked Ensemble results.

fused leave-one-domain-out AUC = 0.976; fused AOA coverage = 90.7%. Dominant mechanism groups are AlphaEarth embeddings (57.3%), Proximity controls (16.9%). Top fused factors are Distance to roads, AlphaEarth A16, AlphaEarth A55, Valley depth, AlphaEarth A44.

## Files

- `domain_treeshap_importance_all_feature_sets.csv`
- `domain_treeshap_group_importance_all_feature_sets.csv`
- `domain_treeshap_importance_*.png`
- `domain_treeshap_beeswarm_*.png`
- `domain_mechanism_group_shap_heatmap.png`
