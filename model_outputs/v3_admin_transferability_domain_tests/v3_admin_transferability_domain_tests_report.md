# CPEC V3 Administrative Transferability Domain Tests

Date: 2026-05-10

## Purpose

This analysis implements the first major step of the teacher-approved updated methodology: **transferability-aware landslide susceptibility modelling**. It tests whether the 2018 V3 model framework transfers across political/admin CPEC subdomains rather than relying only on random or ordinary spatial folds.

## Domain Definition

Domains were assigned by spatially joining the V3 samples to political/administrative boundary shapefiles, then aggregating small or legacy units into analysis domains. These are **modelling transfer domains**, not claims that grouped units are the same administrative unit.

- Pakistan GADM level-1 units from `gadm41_PAK_1.shp`.
- Kashgar, Xinjiang from `cpecpart.shp`.

The grouping is justified on three grounds:

1. Review method: spatial and grouped validation are recommended for structured spatial data because ordinary random validation underestimates prediction error where samples are spatially dependent.
2. Transferability method: large-region LSM should test whether models transfer across coherent source/target regions rather than only within one mixed sample pool.
3. Sample adequacy: several administrative units had too few landslide/non-landslide samples for stable leave-one-domain-out testing, so they were grouped into transparent corridor analysis domains.

Public domain labels:

- Kashgar (Xinjiang, China)
- Gilgit-Baltistan
- KP-AJK
- Balochistan
- Punjab-Sindh lowland corridor

Specific naming/aggregation decisions:

- Former FATA is not displayed as a separate current public domain because it was merged with Khyber Pakhtunkhwa through Pakistan's 25th Constitutional Amendment in 2018; legacy GADM polygons labelled FATA are assigned to the KP side of the `KP-AJK` analysis domain.
- AJK is grouped with KP only for transfer-testing sample adequacy and northern-western mountainous corridor interpretation; it is not presented as the same administrative unit.
- Islamabad is administratively separate, but it is absorbed into the `Punjab-Sindh lowland corridor` analysis domain because it contributed only one sample and cannot support a separate held-out test.

Key support:

- Roberts et al. (2017), Ecography, recommend block/group validation for data with spatial/hierarchical dependence: https://doi.org/10.1111/ecog.02881
- Meyer and Pebesma (2021) define area-of-applicability as the area where cross-validation error can be expected to apply: https://doi.org/10.1111/2041-210X.13650
- Official KP portal uses KP for Khyber Pakhtunkhwa: https://kp.gov.pk/
- Pakistan Constitution text reflects the Twenty-fifth Amendment effect on FATA/KP: https://pakistancode.gov.pk/
- AJK official portal uses AJK/AJ&K naming: https://ajk.gov.pk/

## Domain Sample Balance

| cpec_admin_transfer_domain | n | positives | negatives | positive_share |
| --- | --- | --- | --- | --- |
| Kashgar (Xinjiang, China) | 846 | 323 | 523 | 0.382 |
| Gilgit-Baltistan | 721 | 427 | 294 | 0.592 |
| KP-AJK | 351 | 167 | 184 | 0.476 |
| Balochistan | 992 | 516 | 476 | 0.520 |
| Punjab-Sindh lowland corridor | 406 | 75 | 331 | 0.185 |

## Method

For each held-out domain and each feature set, models were trained on all other domains and tested only on the held-out domain. Feature sets:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

Models:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Spatial-CV Stacked Ensemble

The stacked ensemble used source-domain spatial-fold out-of-fold predictions to train the meta-learner. The held-out domain was not used to tune the decision threshold or train the base/meta models.

## Top Stacked-Ensemble Transfer Results

| Feature set | Held-out domain | ROC-AUC | PR-AUC/AP | Balanced accuracy | N test |
| --- | --- | ---: | ---: | ---: | ---: |
| Conventional + AlphaEarth Embeddings | Punjab-Sindh lowland corridor | 0.990 | 0.954 | 0.954 | 406 |
| Conventional | Punjab-Sindh lowland corridor | 0.990 | 0.950 | 0.938 | 406 |
| Conventional + AlphaEarth Embeddings | Gilgit-Baltistan | 0.979 | 0.984 | 0.943 | 721 |
| Conventional + AlphaEarth Embeddings | KP-AJK | 0.976 | 0.973 | 0.932 | 351 |
| Conventional | Kashgar (Xinjiang, China) | 0.972 | 0.963 | 0.918 | 846 |
| Conventional | Gilgit-Baltistan | 0.971 | 0.981 | 0.921 | 721 |
| Conventional | KP-AJK | 0.970 | 0.966 | 0.887 | 351 |
| AlphaEarth Embeddings | Gilgit-Baltistan | 0.968 | 0.972 | 0.905 | 721 |
| AlphaEarth Embeddings | KP-AJK | 0.965 | 0.953 | 0.918 | 351 |
| AlphaEarth Embeddings | Punjab-Sindh lowland corridor | 0.965 | 0.889 | 0.912 | 406 |

## Interpretation

- These results directly support the new paper direction: CPEC should be treated as a multi-domain corridor, not a single homogeneous region.
- Administrative-domain transfer metrics are more rigorous than random-split accuracy because they test extrapolation to different political/planning regions.
- The domain-shift table links model performance to source-target feature similarity, following the logic of recent transfer-learning and area-of-applicability literature.
- The fused feature set should remain the main candidate if it performs well across domains, because it combines physical interpretability with annual foundation-model representation.

## Output Files

- Metrics: `v3_admin_transferability_leave_one_domain_metrics.csv`
- Base learner metrics: `v3_admin_transferability_base_learner_metrics.csv`
- Domain shift table: `v3_admin_transferability_domain_shift.csv`
- Predictions: `v3_admin_transferability_held_out_predictions.csv`
- Domain sample table: `cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv`
- Figures:
  - `figure_cpec_v3_admin_transferability_subdomain_sample_map.png`
  - `figure_cpec_v3_admin_transferability_stacked_auc_heatmap.png`
  - `figure_cpec_v3_admin_domain_shift_vs_transfer_auc.png`

## Next Methodology Step

The next step is to convert this sample-level domain transfer analysis into a raster-level **area-of-applicability / transfer confidence map** using the same predictor space.
