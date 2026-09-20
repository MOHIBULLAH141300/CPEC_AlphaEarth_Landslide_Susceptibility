# 2018 V3 Domain Results: Balochistan

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

- Conventional: ROC-AUC 0.897, PR-AUC 0.910, F1 0.813, Brier 0.136
- AlphaEarth Embeddings: ROC-AUC 0.834, PR-AUC 0.822, F1 0.774, Brier 0.176
- Conventional + AlphaEarth Embeddings: ROC-AUC 0.913, PR-AUC 0.919, F1 0.835, Brier 0.121

## Probability and reliability statistics

| feature_set                          | product                             | valid_pixels | mean   | median | p95    | near_zero_percent_le_0_001 |
| ------------------------------------ | ----------------------------------- | ------------ | ------ | ------ | ------ | -------------------------- |
| Conventional                         | raw_probability                     | 6298553      | 0.2408 | 0.0961 | 0.8873 | 0.0000                     |
| Conventional                         | transfer_confidence                 | 6298553      | 0.1568 | 0.1359 | 0.4427 | 42.5349                    |
| Conventional                         | in_area_of_applicability            | 6298553      | 0.5749 | 1.0000 | 1.0000 | 42.5076                    |
| Conventional                         | reliability_weighted_probability    | 6298553      | 0.0324 | 0.0060 | 0.1907 | 42.7100                    |
| AlphaEarth Embeddings                | raw_probability                     | 6298553      | 0.2686 | 0.0996 | 0.9171 | 0.0000                     |
| AlphaEarth Embeddings                | transfer_confidence                 | 6298553      | 0.4219 | 0.4285 | 0.6837 | 1.7591                     |
| AlphaEarth Embeddings                | in_area_of_applicability            | 6298553      | 0.9825 | 1.0000 | 1.0000 | 1.7452                     |
| AlphaEarth Embeddings                | reliability_weighted_probability    | 6298553      | 0.1061 | 0.0384 | 0.3892 | 1.9347                     |
| Conventional + AlphaEarth Embeddings | raw_probability                     | 6298553      | 0.2053 | 0.0581 | 0.9372 | 0.0000                     |
| Conventional + AlphaEarth Embeddings | transfer_confidence                 | 6298553      | 0.3981 | 0.4071 | 0.5865 | 1.1584                     |
| Conventional + AlphaEarth Embeddings | in_area_of_applicability            | 6298553      | 0.9885 | 1.0000 | 1.0000 | 1.1498                     |
| Conventional + AlphaEarth Embeddings | reliability_weighted_probability    | 6298553      | 0.0760 | 0.0218 | 0.3421 | 1.3125                     |
| Final fused                          | final_fused_reliability_aware_score | 6298553      | 0.0575 | 0.0189 | 0.2734 | 1.3379                     |
| Final fused                          | final_reliable_high_mask            | 6298553      | 0.0033 | 0.0000 | 0.0000 | 99.6739                    |
| Final fused                          | final_reliable_very_high_mask       | 6298553      | 0.0632 | 0.0000 | 1.0000 | 93.6847                    |
| Final fused                          | final_uncertain_high_mask           | 6298553      | 0.2839 | 0.0000 | 1.0000 | 71.6052                    |
| Final fused                          | field_verification_priority_score   | 6298553      | 0.1646 | 0.0357 | 0.6683 | 0.0000                     |
| Final fused                          | model_disagreement_std              | 6298553      | 0.0738 | 0.0369 | 0.2538 | 0.0111                     |

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
