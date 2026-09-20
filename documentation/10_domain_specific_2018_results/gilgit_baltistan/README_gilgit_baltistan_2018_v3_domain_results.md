# 2018 V3 Domain Results: Gilgit-Baltistan

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

- Conventional: ROC-AUC 0.971, PR-AUC 0.981, F1 0.927, Brier 0.073
- AlphaEarth Embeddings: ROC-AUC 0.968, PR-AUC 0.972, F1 0.926, Brier 0.068
- Conventional + AlphaEarth Embeddings: ROC-AUC 0.979, PR-AUC 0.984, F1 0.956, Brier 0.042

## Probability and reliability statistics

| feature_set                          | product                             | valid_pixels | mean   | median | p95    | near_zero_percent_le_0_001 |
| ------------------------------------ | ----------------------------------- | ------------ | ------ | ------ | ------ | -------------------------- |
| Conventional                         | raw_probability                     | 1347245      | 0.1780 | 0.0339 | 0.9671 | 0.0000                     |
| Conventional                         | transfer_confidence                 | 1347245      | 0.0284 | 0.0000 | 0.1608 | 77.5471                    |
| Conventional                         | in_area_of_applicability            | 1347245      | 0.2252 | 0.0000 | 1.0000 | 77.4836                    |
| Conventional                         | reliability_weighted_probability    | 1347245      | 0.0023 | 0.0000 | 0.0093 | 79.4867                    |
| AlphaEarth Embeddings                | raw_probability                     | 1347245      | 0.2347 | 0.0495 | 0.9668 | 0.0000                     |
| AlphaEarth Embeddings                | transfer_confidence                 | 1347245      | 0.3786 | 0.3923 | 0.6175 | 2.7437                     |
| AlphaEarth Embeddings                | in_area_of_applicability            | 1347245      | 0.9728 | 1.0000 | 1.0000 | 2.7216                     |
| AlphaEarth Embeddings                | reliability_weighted_probability    | 1347245      | 0.0756 | 0.0211 | 0.3944 | 2.8705                     |
| Conventional + AlphaEarth Embeddings | raw_probability                     | 1347245      | 0.1843 | 0.0310 | 0.9780 | 0.0000                     |
| Conventional + AlphaEarth Embeddings | transfer_confidence                 | 1347245      | 0.3282 | 0.3435 | 0.5196 | 2.2723                     |
| Conventional + AlphaEarth Embeddings | in_area_of_applicability            | 1347245      | 0.9775 | 1.0000 | 1.0000 | 2.2479                     |
| Conventional + AlphaEarth Embeddings | reliability_weighted_probability    | 1347245      | 0.0447 | 0.0112 | 0.2586 | 2.4019                     |
| Final fused                          | final_fused_reliability_aware_score | 1347245      | 0.0369 | 0.0108 | 0.2155 | 2.4418                     |
| Final fused                          | final_reliable_high_mask            | 1347245      | 0.0063 | 0.0000 | 0.0000 | 99.3677                    |
| Final fused                          | final_reliable_very_high_mask       | 1347245      | 0.0150 | 0.0000 | 0.0000 | 98.4957                    |
| Final fused                          | final_uncertain_high_mask           | 1347245      | 0.2154 | 0.0000 | 1.0000 | 78.4627                    |
| Final fused                          | field_verification_priority_score   | 1347245      | 0.1619 | 0.0204 | 0.8006 | 0.0000                     |
| Final fused                          | model_disagreement_std              | 1347245      | 0.0452 | 0.0098 | 0.2350 | 0.0212                     |

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
