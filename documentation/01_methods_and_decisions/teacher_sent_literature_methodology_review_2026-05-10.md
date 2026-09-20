# Teacher-Sent Literature Review And Publishable Methodology Direction

Date: 2026-05-10

Project context reviewed:

- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS`
- 2018 V3 supervisor summary, factor audit, VIF report, model-family report, stacked ensemble outputs, SHAP outputs, uncertainty/added-value/road exposure outputs, and temporal extension plan.

## Short Conclusion

The teacher's new papers are pointing toward a stronger paper than a simple 2018 model comparison. The most publishable direction for our data is:

**Transferability-aware dynamic foundation-embedding ensemble landslide susceptibility mapping for the CPEC/KKH corridor.**

This means we should keep the clean 2018 V3 framework, but extend it with:

1. Annual dynamic-factor susceptibility maps from 2017-2024.
2. AlphaEarth/Satellite Embedding added-value and annual-change analysis.
3. Spatial/domain transferability testing across geomorphic/CPEC subregions.
4. Area-of-applicability / environmental similarity mapping.
5. Sampling-robust uncertainty from repeated non-landslide sampling.
6. SHAP/ALE/counterfactual interpretation and road-distance ablation.
7. KKH/CPEC infrastructure exposure and persistent hotspot analysis.

## Why This Fits The Papers Sent By The Teacher

### 1. Transfer learning for data-scarce LSM

The Scientific Reports 2024 paper, **Ensembled transfer learning approach for error reduction in landslide susceptibility mapping of the data scarce region**, argues that LSM is limited by expensive landslide inventory collection in mountain regions and uses source-to-target transfer learning to improve target-area performance. It also uses VIF, KL divergence for source-target similarity, RF/MLP, and metrics including AUC-ROC, precision, recall, F-score, and accuracy.

How to use this idea in our project:

- Do not simply train one CPEC model and stop.
- Divide CPEC into meaningful target domains, for example:
  - KKH/Kashgar-northern Pakistan high mountains
  - Himalaya/Karakoram/Hindu Kush belt
  - Indus basin / central Pakistan
  - Balochistan / western Pakistan
  - southern coastal/lowland corridor
- Run leave-one-domain-out or source-target transfer tests.
- Calculate source-target feature similarity using KL divergence, Jensen-Shannon divergence, or MMD.
- Produce a transferability map showing where the 2018 model is expected to generalize well or poorly.

This would answer a reviewer question: **does the model really transfer across a huge transboundary corridor, or only memorize local geography?**

### 2. Multi-source domain adaptation for large/complex no-sample regions

The Geoscience Frontiers 2025 paper on **multi-source domain transfer learning** states that single-source transfer can fail when landslide types and triggering mechanisms are diverse, and proposes using multiple source domains to reduce domain feature shift.

How to use this idea in our project:

- Treat CPEC as a multi-domain system rather than one homogeneous region.
- Train source-domain combinations and test on held-out target domains.
- Compare:
  - single-source transfer
  - all-source pooled model
  - spatial-CV stacked ensemble
  - fused Conventional + AlphaEarth Embeddings model
- Use feature/domain shift diagnostics to explain where performance changes.

This is highly relevant because CPEC includes high mountains, tectonic belts, arid regions, plains, and road-disturbed slopes.

### 3. Cross-regional extrapolation and SHAP

The Geoscience Frontiers 2025 paper **Cross-regional extrapolation of landslide susceptibility mapping via transfer learning** compares RF, CNN-BiLSTM, and transfer-learning models, and uses SHAP for global model interpretation.

How to use this idea in our project:

- We already have SHAP for the 2018 V3 models.
- Add cross-domain extrapolation evaluation and use SHAP to compare whether controlling factors change between domains.
- Instead of claiming deep learning just because it is fashionable, use deep learning only if there is enough spatially structured labelled data. For our current table/raster data, a stacked tree ensemble is more defensible.

### 4. Causal representation / counterfactual robustness

The GIScience & Remote Sensing 2025 CICRL-FLM paper is mainly about landslide **mapping/segmentation** from satellite images, not classical susceptibility mapping. Its useful message for us is not that we must build the same deep network; it is that reviewers now care about spurious correlations, causal robustness, and interpretable inference.

How to use this idea in our project:

- Add counterfactual and ablation checks:
  - remove distance to roads and see whether KKH hotspots remain physically plausible
  - rainfall scenario: replace rainfall with high-percentile or return-period rainfall
  - seismic scenario: add PGA or scenario shaking when data are available
  - AlphaEarth ablation: fused model minus Conventional model
- Report these as **causal-robustness inspired diagnostics**, not as formal causal proof.

### 5. Transfer learning + GNN landslide detection

The Geoscience Frontiers 2025 GNN paper is also focused on landslide **detection/segmentation**, especially group-occurring landslides. It supports transfer learning and spatial-context modelling, but it is not directly required for our susceptibility raster.

How to use this idea in our project:

- Do not replace the current V3 susceptibility model with a GNN unless we create a landslide-image segmentation dataset.
- Borrow the spatial-context idea by:
  - adding slope-unit or road-segment aggregation
  - testing spatial neighborhood smoothing only as a post-analysis, not as a hidden way to inflate AUC
  - ranking KKH road segments by mean probability, uncertainty, and persistence

### 6. Meta-learning / feature selection / Extra Trees

The Scientific Reports 2025 meta-learning paper supports a framework with LR, SVM, RF, Extra Trees, GB, XGBoost, and a meta-classifier, reporting strong AUC and accuracy for the meta-classifier and Extra Trees.

How to use this idea in our project:

- Our current V3 already includes Logistic Regression, RF, Extra Trees, XGBoost, LightGBM, CatBoost, and Spatial-CV Stacked Ensemble.
- Add a manuscript paragraph explaining model roles:
  - Logistic Regression: linear baseline and calibration reference
  - RF/Extra Trees: variance-reducing bagging, robust nonlinear interactions
  - XGBoost/LightGBM/CatBoost: boosted tree learners for complex nonlinear factor interactions
  - Spatial-CV Stacked Ensemble: meta-learning layer combining complementary learners using out-of-fold predictions
- Add RFE or permutation-stability only as a **diagnostic**, not necessarily to change the final V3 feature set, because V3 already passed VIF and literature screening.

### 7. Uncertainty, interpretability, and sampling

Recent uncertainty/XAI LSM literature supports bootstrapping, Monte Carlo simulation, feature selection, and SHAP. Recent sampling papers show that non-landslide/control-point selection can strongly affect susceptibility results.

How to use this idea in our project:

- Keep current feature-set disagreement maps.
- Add repeated non-landslide sampling:
  - 20-50 repeats if computationally feasible
  - report mean probability, standard deviation, and confidence masks
- Use calibration curves and Brier score along with ROC-AUC and PR-AUC.
- Use SHAP/ALE dependence plots for physical interpretation.
- Use SCAI/LDI only as an optional class-consistency diagnostic if we publish susceptibility classes.

## Recommended Final Methodology

I recommend structuring the manuscript method as eight modules.

### Module 1. CPEC/KKH inventory and factor database

Use the official CPEC study boundary and the current V3 landslide inventory/samples. Keep the three feature-set design:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

Conventional factors should remain the V3 set:

- topography: elevation, slope, aspect, valley depth, TRI, TWI, profile curvature, plan curvature
- hydro-climate: monsoon rainfall total, maximum 1-day rainfall, distance to rivers/streams
- vegetation/land cover: NDVI median, NDVI amplitude, land-cover class
- geology/seismicity: lithology, soil type, distance to active faults, earthquake density
- infrastructure disturbance: distance to roads

### Module 2. Multicollinearity and feature stability gate

Already completed for V3:

- VIF accepted, all final conventional factors below 4.

Add before manuscript submission:

- permutation importance stability or RFE stability as a supplementary diagnostic
- grouped embedding importance so AlphaEarth is not misrepresented as 64 independent physical variables

### Module 3. Spatial-CV stacked ensemble baseline

Keep the current model family:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Spatial-CV Stacked Ensemble

For the main paper, present the stacked ensemble as the framework model for all three feature sets, while reporting that XGBoost/CatBoost are competitive individual learners.

### Module 4. Transferability and area-of-applicability analysis

This is the strongest new addition from the teacher's papers.

Recommended outputs:

- CPEC subdomain map.
- Leave-one-domain-out AUC/AP/calibration table.
- Source-target similarity table using KL divergence or Jensen-Shannon divergence.
- Area-of-applicability or dissimilarity-index raster.
- Transfer confidence map.

This directly connects our paper to transfer-learning and cross-regional extrapolation literature.

### Module 5. Annual dynamic-factor susceptibility, 2017-2024

Use the already planned Option 1:

- fixed 2018 V3 trained model
- annual rainfall, NDVI, land cover, and AlphaEarth Embeddings
- annual probability maps at 250 m

Outputs:

- annual probability maps, 2017-2024
- year-to-year change maps
- trend map
- mean susceptibility map
- persistent high-susceptibility mask
- annual KKH/CPEC road exposure table

Do not call it fully validated temporal susceptibility unless we enrich dated landslide inventory.

### Module 6. AlphaEarth added-value and representation-change analysis

This is likely our biggest novelty.

Recommended outputs:

- Conventional + AlphaEarth minus Conventional probability map
- AlphaEarth-only vs Conventional disagreement map
- annual AlphaEarth embedding-change map using cosine distance / dot product
- zones where AlphaEarth adds susceptibility information beyond conventional factors
- grouped SHAP importance of AlphaEarth contribution

This is strongly justified by the official Satellite Embedding documentation because the embedding vectors are consistent across years and designed for classification and change detection.

### Module 7. Uncertainty, sampling robustness, and calibration

Recommended outputs:

- feature-set disagreement uncertainty, already created
- repeated non-landslide sampling uncertainty
- calibration curves and Brier score
- reliable high-probability zones
- uncertain high-probability zones
- optional SCAI/LDI class-consistency table if classes are used

This will make the results reviewer-friendly because high AUC alone is no longer enough.

### Module 8. KKH/CPEC infrastructure exposure and scenario branch

Keep the 2018 KKH road exposure outputs and extend annually:

- road length exposed to high probability
- persistent high-susceptibility road segments
- increasing susceptibility road segments
- uncertain high-risk segments for field verification

Scenario branch to add if data are available:

- rainfall scenario using high-percentile or return-period rainfall
- seismic scenario using PGA or earthquake shaking proxy
- road-distance ablation to prove the road hotspot output is not only a circular consequence of road distance

## What We Should Not Do Yet

Do not replace the whole project with a deep CNN/GNN unless we build a pixel-level landslide image dataset. The teacher-sent deep-learning papers are mainly about landslide detection/segmentation, while our current study is susceptibility mapping from tabular/raster predictors.

Do not claim formal causal inference. We can use causal-robustness language carefully:

- counterfactual tests
- ablation tests
- spurious-correlation checks
- physically interpretable SHAP/ALE

Do not call the annual 2017-2024 maps fully validated temporal susceptibility unless dated inventory validation is added.

## Most Publishable Manuscript Framing

Suggested working title:

**Transferability-aware dynamic landslide susceptibility mapping of the CPEC/KKH corridor using AlphaEarth foundation embeddings and spatial-CV stacked ensembles**

Main innovation in simple words:

**We combine physically interpretable landslide factors with annual AlphaEarth foundation embeddings, then test whether the model is transferable across the diverse CPEC corridor, reliable through uncertainty/sampling checks, and useful for identifying persistent KKH infrastructure hotspots.**

## Priority Work Plan

1. Finish annual 2017-2024 predictor exports and probability maps.
2. Create CPEC subdomain/domain map.
3. Run leave-one-domain-out transferability evaluation.
4. Produce area-of-applicability / dissimilarity-index map.
5. Add repeated non-landslide sampling uncertainty.
6. Create annual KKH road exposure and persistent hotspot outputs.
7. Add road-distance ablation and rainfall/seismic scenario branch if feasible.
8. Prepare manuscript framework diagram.

## Key References Read

- Singh et al. (2024), Scientific Reports: Ensembled transfer learning approach for error reduction in landslide susceptibility mapping of the data scarce region. https://www.nature.com/articles/s41598-024-76541-4
- Halder et al. (2025), Scientific Reports: Improving landslide susceptibility prediction through ensemble recursive feature elimination and meta-learning framework. https://www.nature.com/articles/s41598-025-87587-3
- Zhao et al. (2025), GIScience & Remote Sensing: CICRL-FLM counterfactual inference causal representation learning network for fine-grained landslide mapping. https://www.tandfonline.com/doi/full/10.1080/15481603.2025.2598078
- Luo et al. (2025), Geoscience Frontiers: A proposed method for landslide detection based on transfer learning and graph neural network. https://www.sciencedirect.com/science/article/pii/S1674987125001768
- Su et al. (2025), Geoscience Frontiers: Complex cross-regional landslide susceptibility mapping by multi-source domain transfer learning. https://www.sciencedirect.com/science/article/pii/S1674987125000581
- Wang et al. (2025), Geoscience Frontiers: Cross-regional extrapolation of landslide susceptibility mapping via transfer learning. https://www.sciencedirect.com/science/article/pii/S1674987125002178
- Le et al. (2024), Frontiers in Environmental Science: Quantitative evaluation of uncertainty and interpretability in ML-based LSM through feature selection and XAI. https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1424988/full
- Dhakal et al. (2025), Scientific Reports: Enhancing landslide disaster prediction by evaluating non-landslide area sampling in ML models for Spiti Valley India. https://www.nature.com/articles/s41598-025-95087-7
- Lombardo et al. (2020), Earth-Science Reviews: Space-time landslide predictive modelling. https://doi.org/10.1016/j.earscirev.2020.103318
- Meyer and Pebesma (2021), Methods in Ecology and Evolution: Predicting into unknown space? Estimating the area of applicability of spatial prediction models. https://doi.org/10.1111/2041-210X.13650
- Google Earth Engine Data Catalog: Satellite Embedding V1 / AlphaEarth Foundations. https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- NASA NSIDC: High Mountain Asia Landslide Catalog, Version 2. https://nsidc.org/data/hma_ls_cat/versions/2

