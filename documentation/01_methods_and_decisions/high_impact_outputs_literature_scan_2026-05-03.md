# High-Impact Publishable Outputs From The CPEC 2018 Landslide Susceptibility Results

Date: 2026-05-03

## Literature Signals Checked

- Recent LSM papers increasingly combine ensemble learning, SHAP/XAI, uncertainty analysis, and calibration rather than reporting only ROC-AUC.
- Recent work emphasizes spatial cross-validation and non-landslide sampling uncertainty because random splits can inflate performance.
- Foundation-model embeddings are becoming a new research frontier for geospatial prediction. Google Satellite Embedding/AlphaEarth provides annual 64-band, 10 m representations from 2017 onward, making our 2018 baseline defensible and novel.
- High-impact framing should connect model performance to geomorphic interpretation and infrastructure decision-making along CPEC/KKH.

## Strongest Additional Outputs To Create

### 1. Uncertainty And Reliability Map

Create raster maps showing:

- Mean stacked probability.
- Base-model disagreement/standard deviation.
- Entropy or confidence index.
- Reliable high-susceptibility zones where probability is high and uncertainty is low.
- Uncertain high-susceptibility zones where probability is high but model disagreement is high.

Why it matters:

- Recent LSM literature treats uncertainty as essential for responsible map use.
- It gives reviewers more than a single deterministic map.
- It helps planners distinguish confident hazard zones from areas needing field verification.

### 2. CPEC/KKH Infrastructure Exposure Analysis

Overlay susceptibility with:

- KKH and major roads.
- Rail/project corridors if available.
- Hydropower/reservoir infrastructure if available.
- Buffer zones such as 0-250 m, 250-500 m, 500-1000 m from roads.

Outputs:

- Kilometers of road in high-probability zones.
- Hotspot ranking of road segments.
- Administrative summaries by Pakistan province and Kashgar.
- Priority field-verification segments.

Why it matters:

- This turns the map into corridor risk intelligence.
- It directly matches CPEC relevance and teacher/reviewer expectations.

### 3. AlphaEarth Added-Value Diagnostics

Create explicit evidence for what AlphaEarth contributes beyond conventional factors:

- Delta probability map: Fused minus Conventional.
- Improved-hit zones: landslide points better ranked by fused model.
- AlphaEarth-only vs Conventional-only disagreement map.
- Areas where AlphaEarth captures missing environmental information.
- Embedding SHAP cluster interpretation.

Why it matters:

- The main innovation is not just using AlphaEarth; it is proving where and why it helps.
- This supports a high-impact novelty claim.

### 4. Spatial Generalization And Robustness Figures

Create:

- Fold-wise ROC-AUC/PR-AUC boxplots.
- Spatial fold maps showing train/test regions.
- Per-fold performance table.
- Stability ranking of factors across folds.

Why it matters:

- Spatial-CV is more defensible than random CV for geohazard maps.
- Reviewers often question whether the model learned geography instead of landslide controls.

### 5. Model Calibration And Decision Threshold Products

Create:

- Calibration curve already prepared.
- Reliability table by probability bin.
- Confusion matrix at multiple thresholds.
- Threshold selected for high recall, balanced accuracy, and high precision.
- Probability-to-priority interpretation table.

Why it matters:

- ROC-AUC alone does not show whether probabilities are meaningful.
- A calibrated susceptibility probability is easier to justify for planning.

### 6. Landslide Mechanism Profiles

Use SHAP dependence and interaction-style plots for key factors:

- Distance to roads.
- Elevation.
- Valley depth.
- Terrain ruggedness.
- Distance to rivers/streams.
- Distance to active faults.
- AlphaEarth A16/A55/A44.

Outputs:

- SHAP dependence plots.
- Partial dependence/ALE plots.
- Factor-threshold interpretation table.

Why it matters:

- It converts black-box ML into geomorphic explanation.
- It answers reviewer questions about physical plausibility.

### 7. Inventory Sensitivity And Sampling Robustness

Create:

- Multiple non-landslide sampling realizations.
- Performance distribution across realizations.
- Map uncertainty from sampling.
- Comparison of random non-landslide, distance-constrained, and terrain-stratified controls.

Why it matters:

- Recent literature explicitly highlights non-landslide sampling as a major source of uncertainty.
- It protects the study from “your absence points are arbitrary” criticism.

### 8. Embedding Compression And Interpretability

Create:

- PCA/UMAP of AlphaEarth embeddings.
- RGB composite of first 3 principal components.
- PCA vs full 64-band performance.
- Cluster map of embedding regimes.

Why it matters:

- AlphaEarth bands are abstract. PCA/cluster outputs make them interpretable.
- It gives a visually strong “foundation model learned terrain/environment regimes” story.

## Best Publishable Figure Set

Recommended high-impact figure package:

1. Study area, landslide inventory, CPEC/KKH infrastructure context.
2. Workflow diagram: conventional factors + AlphaEarth + spatial-CV stacking + uncertainty.
3. Model comparison curves: ROC, PR, calibration.
4. Susceptibility probability maps for Conventional, AlphaEarth, Fused.
5. Delta map: Fused minus Conventional.
6. Uncertainty/reliability map.
7. Factor-level SHAP beeswarm and SHAP dependence plots.
8. Infrastructure exposure hotspot map and road-length table.
9. Spatial fold robustness/performance stability plot.

## Recommended Next Step

The highest value next product is:

**Fused Stacked Ensemble probability + uncertainty + CPEC/KKH infrastructure hotspot analysis.**

This gives the paper a practical engineering contribution, not only a modelling contribution.

## Sources Consulted

- Frontiers in Environmental Science, 2024: uncertainty and interpretability in ML-based landslide susceptibility mapping using feature selection, explainable AI, Bayesian optimization, bootstrapping, and Monte Carlo.
- Scientific Reports, 2024: non-landslide sampling strategies affect machine-learning landslide susceptibility mapping.
- Advances in Space Research, 2024: local/global explainable AI for ensemble learning in landslide susceptibility mapping.
- Remote Sensing, 2024: optimized sampling plus heterogeneous ensemble machine learning for LSM.
- Scientific Reports, 2024: ensembled transfer learning for error reduction in data-scarce LSM.
- Applied Sciences, 2025: SHAP-based and optimized non-landslide sampling strategies.
- Google Earth Engine Data Catalog: Google Satellite Embedding V1 annual 64-band, 10 m AlphaEarth/Google DeepMind embeddings from 2017 onward.
- Google Earth Engine community tutorial: Satellite Embedding dataset introduction.
- AlphaEarth Foundations preprint, 2025: annual analysis-ready embedding layers for sparse-label global mapping.
- AlphaEarth landslide susceptibility preprint, 2026: conventional landslide conditioning factors versus AlphaEarth embeddings using deep learning.
