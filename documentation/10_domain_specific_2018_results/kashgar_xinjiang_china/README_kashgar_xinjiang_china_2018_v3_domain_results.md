# 2018 V3 Domain Results: Kashgar (Xinjiang, China)

## Interpretation rule

The probability rasters in `01_probability_maps_250m` are the main stacked
ensemble susceptibility probability maps. The reliability-weighted rasters in
`02_transfer_confidence_reliability_250m` are secondary confidence-adjusted
products. They can be close to zero where the Area of Applicability / transfer
confidence is low, especially for the Conventional feature set.

## Module status for this domain

| Module | Status | Domain-specific output |
|---|---:|---|
| Official study area | Complete | Domain polygon is clipped to the official CPEC boundary. |
| 2018 corrected factor database | Complete | Same validated CPEC 2018 V3 factor database; domain sample balance table included. |
| Multicollinearity gate | Complete | Global V3 VIF/correlation results included for transparent factor selection. |
| Model family | Complete | Leave-one-domain-out stacked and base learner metrics included. |
| Feature-set comparison | Complete | Conventional, AlphaEarth Embeddings, and Conventional + AlphaEarth Embeddings probability maps included. |
| Explainability | Complete globally | SHAP outputs remain global; domain probability/transfer outputs are packaged here. |
| Uncertainty and exposure | Complete | Domain AOA, reliability-aware area, uncertainty masks, and road exposure tables included. |
| Annual dynamic extension | Pending by year | Annual outputs are stored separately under the annual 2017-2024 workspace. |

## Leave-one-domain-out performance

- Conventional: ROC-AUC 0.972, PR-AUC 0.963, F1 0.893, Brier 0.062
- AlphaEarth Embeddings: ROC-AUC 0.933, PR-AUC 0.907, F1 0.826, Brier 0.095
- Conventional + AlphaEarth Embeddings: ROC-AUC 0.965, PR-AUC 0.957, F1 0.880, Brier 0.065

## Probability and reliability statistics

| feature_set                          | product                             | valid_pixels | mean   | median | p95    | near_zero_percent_le_0_001 |
| ------------------------------------ | ----------------------------------- | ------------ | ------ | ------ | ------ | -------------------------- |
| Conventional                         | raw_probability                     | 3745471      | 0.1255 | 0.0464 | 0.6628 | 0.0000                     |
| Conventional                         | transfer_confidence                 | 3745470      | 0.0954 | 0.0819 | 0.2723 | 27.4201                    |
| Conventional                         | in_area_of_applicability            | 3745470      | 0.7278 | 1.0000 | 1.0000 | 27.2203                    |
| Conventional                         | reliability_weighted_probability    | 3745470      | 0.0081 | 0.0035 | 0.0308 | 31.8900                    |
| AlphaEarth Embeddings                | raw_probability                     | 3745471      | 0.1213 | 0.0443 | 0.6575 | 0.0000                     |
| AlphaEarth Embeddings                | transfer_confidence                 | 3745470      | 0.4005 | 0.3962 | 0.7199 | 1.9659                     |
| AlphaEarth Embeddings                | in_area_of_applicability            | 3745470      | 0.9806 | 1.0000 | 1.0000 | 1.9421                     |
| AlphaEarth Embeddings                | reliability_weighted_probability    | 3745470      | 0.0407 | 0.0207 | 0.1904 | 2.2888                     |
| Conventional + AlphaEarth Embeddings | raw_probability                     | 3745471      | 0.0842 | 0.0304 | 0.4020 | 0.0000                     |
| Conventional + AlphaEarth Embeddings | transfer_confidence                 | 3745470      | 0.3728 | 0.3778 | 0.5744 | 1.1379                     |
| Conventional + AlphaEarth Embeddings | in_area_of_applicability            | 3745470      | 0.9887 | 1.0000 | 1.0000 | 1.1293                     |
| Conventional + AlphaEarth Embeddings | reliability_weighted_probability    | 3745470      | 0.0247 | 0.0122 | 0.0889 | 1.2449                     |
| Final fused                          | final_fused_reliability_aware_score | 3745470      | 0.0195 | 0.0116 | 0.0512 | 1.3044                     |
| Final fused                          | final_reliable_high_mask            | 3745470      | 0.0009 | 0.0000 | 0.0000 | 99.9085                    |
| Final fused                          | final_reliable_very_high_mask       | 3745470      | 0.0051 | 0.0000 | 0.0000 | 99.4919                    |
| Final fused                          | final_uncertain_high_mask           | 3745470      | 0.1046 | 0.0000 | 1.0000 | 89.5445                    |
| Final fused                          | field_verification_priority_score   | 3745470      | 0.0740 | 0.0195 | 0.4658 | 0.0000                     |
| Final fused                          | model_disagreement_std              | 3745470      | 0.0481 | 0.0127 | 0.2439 | 0.0122                     |

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
