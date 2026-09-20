# CPEC 2018 Area-of-Applicability and Transfer Confidence

Date: 2026-05-10

## Purpose

This output converts the sample-level transferability analysis into raster-level reliability information. It identifies where 2018 susceptibility predictions are made inside the environmental/embedding feature space represented by the training samples, and where predictions should be interpreted more cautiously because they require extrapolation.

## Method Summary

The method follows the area-of-applicability logic used in spatial machine learning and environmental prediction: predictions are most reliable where the predictor-space dissimilarity from training samples is low. For each feature set, predictors were standardized using the 2018 sample table, weighted by the fitted Extra Trees feature importance, compressed with PCA when more than 10 predictors were present, and compared to the nearest training sample in feature space. PCA was used to avoid unstable high-dimensional distances while retaining the dominant predictor-space structure. The area-of-applicability threshold is the 95th percentile of spatial-CV validation-to-training nearest-neighbour distances, so the threshold is calibrated to the same spatial generalization problem used in model evaluation.

Three products were created for each feature set:

- Dissimilarity index: nearest-neighbour feature-space distance divided by the 95th percentile training threshold. Lower is more familiar.
- Area-of-applicability mask: 1 means the pixel is inside the 95th percentile training envelope; 0 means extrapolative.
- Transfer confidence: 0-1 map derived from the dissimilarity index. Higher means the pixel is closer to known training conditions.

## Overall Coverage

| feature_set | valid_study_area_pixels | in_area_of_applicability_percent | mean_transfer_confidence | mean_dissimilarity_index |
| --- | --- | --- | --- | --- |
| Conventional | 19996413 | 34.43 | 0.071 | 1.080 |
| AlphaEarth Embeddings | 19996413 | 95.80 | 0.411 | 0.594 |
| Conventional + AlphaEarth Embeddings | 19996413 | 97.66 | 0.388 | 0.615 |

## Subdomain Coverage

| feature_set | domain | study_area_pixels | in_area_of_applicability_pixels | in_area_of_applicability_percent | mean_transfer_confidence |
| --- | --- | --- | --- | --- | --- |
| Conventional | Kashgar (Xinjiang, China) | 3745470 | 2725942 | 72.78 | 0.095 |
| Conventional | Gilgit-Baltistan | 1347245 | 303351 | 22.52 | 0.028 |
| Conventional | KP-AJK | 2232492 | 133214 | 5.97 | 0.008 |
| Conventional | Balochistan | 6298553 | 3621190 | 57.49 | 0.157 |
| Conventional | Punjab-Sindh lowland corridor | 6390937 | 103310 | 1.62 | 0.003 |
| AlphaEarth Embeddings | Kashgar (Xinjiang, China) | 3745470 | 3672728 | 98.06 | 0.400 |
| AlphaEarth Embeddings | Gilgit-Baltistan | 1347245 | 1310578 | 97.28 | 0.379 |
| AlphaEarth Embeddings | KP-AJK | 2232492 | 1788209 | 80.10 | 0.219 |
| AlphaEarth Embeddings | Balochistan | 6298553 | 6188631 | 98.25 | 0.422 |
| AlphaEarth Embeddings | Punjab-Sindh lowland corridor | 6390937 | 6213600 | 97.23 | 0.481 |
| Conventional + AlphaEarth Embeddings | Kashgar (Xinjiang, China) | 3745470 | 3703174 | 98.87 | 0.373 |
| Conventional + AlphaEarth Embeddings | Gilgit-Baltistan | 1347245 | 1316960 | 97.75 | 0.328 |
| Conventional + AlphaEarth Embeddings | KP-AJK | 2232492 | 2024093 | 90.67 | 0.242 |
| Conventional + AlphaEarth Embeddings | Balochistan | 6298553 | 6226131 | 98.85 | 0.398 |
| Conventional + AlphaEarth Embeddings | Punjab-Sindh lowland corridor | 6390937 | 6276306 | 98.21 | 0.452 |

## Method Parameters

```json
[
  {
    "feature_set": "Conventional",
    "n_features_original": 19,
    "n_features_for_distance": 15,
    "pca_used": true,
    "pca_explained_variance": 0.9573853015899658,
    "aoa_threshold": 3.1186551589620146,
    "aoa_threshold_source": "spatial_cv_validation_to_training_95pct",
    "weight_source": "Extra Trees feature importance from the fitted 2018 model family",
    "weight_min": 0.6694156527519226,
    "weight_max": 1.8069427013397217,
    "weight_mean": 0.9552543759346008
  },
  {
    "feature_set": "AlphaEarth Embeddings",
    "n_features_original": 64,
    "n_features_for_distance": 20,
    "pca_used": true,
    "pca_explained_variance": 0.9394072890281677,
    "aoa_threshold": 4.039782395556711,
    "aoa_threshold_source": "spatial_cv_validation_to_training_95pct",
    "weight_source": "Extra Trees feature importance from the fitted 2018 model family",
    "weight_min": 0.651398241519928,
    "weight_max": 2.1828083992004395,
    "weight_mean": 0.9591771364212036
  },
  {
    "feature_set": "Conventional + AlphaEarth Embeddings",
    "n_features_original": 83,
    "n_features_for_distance": 20,
    "pca_used": true,
    "pca_explained_variance": 0.8858920335769653,
    "aoa_threshold": 5.264113241658917,
    "aoa_threshold_source": "spatial_cv_validation_to_training_95pct",
    "weight_source": "Extra Trees feature importance from the fitted 2018 model family",
    "weight_min": 0.5982375741004944,
    "weight_max": 2.5262858867645264,
    "weight_mean": 0.9405156373977661
  }
]
```

## Literature Basis

- Area-of-applicability and dissimilarity-index concepts are based on spatial/environmental ML transferability literature, especially Meyer and Pebesma's work on predicting into unknown space: https://doi.org/10.1111/2041-210X.13650
- The implementation follows the same core logic used in the CAST area-of-applicability family of spatial prediction tools: https://arxiv.org/abs/2404.06978
- Recent landslide susceptibility and transfer-learning papers emphasize that high model accuracy alone is not enough; spatial transferability, domain shift, and extrapolation risk must also be reported.
- This step therefore supports the updated teacher-approved direction: transferability-aware dynamic landslide susceptibility mapping for CPEC/KKH.

## Subdomain Naming Note

The subdomain labels are analysis-domain labels created for transferability and reporting. They are based on political boundaries but grouped where sample counts would otherwise be too small for stable spatial transfer testing.

- Former FATA is assigned to KP because it was merged with Khyber Pakhtunkhwa under Pakistan's 25th Constitutional Amendment in 2018.
- AJK is grouped with KP only as a northern-western mountainous transfer domain; it is not treated as the same administrative unit.
- Islamabad is grouped into the Punjab-Sindh lowland corridor only for modelling stability because it had one sample in the 2018 V3 table.

## Output Folders

- Managed rasters: `D:\DING PROJECT\04_maps\area_of_applicability_transfer_confidence_250m`
- Clean package rasters: `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\03_transfer_confidence_area_of_applicability`
- Figures: `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\03_figures\06_area_of_applicability_transfer_confidence`
