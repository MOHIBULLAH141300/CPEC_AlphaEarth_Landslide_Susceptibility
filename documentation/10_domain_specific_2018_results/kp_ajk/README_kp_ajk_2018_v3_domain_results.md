# 2018 V3 Domain Results: KP-AJK

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

- Conventional: ROC-AUC 0.970, PR-AUC 0.966, F1 0.887, Brier 0.073
- AlphaEarth Embeddings: ROC-AUC 0.965, PR-AUC 0.953, F1 0.914, Brier 0.066
- Conventional + AlphaEarth Embeddings: ROC-AUC 0.976, PR-AUC 0.973, F1 0.929, Brier 0.057

## Probability and reliability statistics

| feature_set                          | product                             | valid_pixels | mean   | median | p95    | near_zero_percent_le_0_001 |
| ------------------------------------ | ----------------------------------- | ------------ | ------ | ------ | ------ | -------------------------- |
| Conventional                         | raw_probability                     | 2232492      | 0.2250 | 0.0873 | 0.9173 | 0.0000                     |
| Conventional                         | transfer_confidence                 | 2232492      | 0.0079 | 0.0000 | 0.0359 | 94.0626                    |
| Conventional                         | in_area_of_applicability            | 2232492      | 0.0597 | 0.0000 | 1.0000 | 94.0329                    |
| Conventional                         | reliability_weighted_probability    | 2232492      | 0.0009 | 0.0000 | 0.0015 | 94.6184                    |
| AlphaEarth Embeddings                | raw_probability                     | 2232492      | 0.2239 | 0.0733 | 0.9157 | 0.0000                     |
| AlphaEarth Embeddings                | transfer_confidence                 | 2232492      | 0.2193 | 0.2076 | 0.5280 | 20.0080                    |
| AlphaEarth Embeddings                | in_area_of_applicability            | 2232492      | 0.8010 | 1.0000 | 1.0000 | 19.9008                    |
| AlphaEarth Embeddings                | reliability_weighted_probability    | 2232492      | 0.0413 | 0.0152 | 0.2124 | 20.9630                    |
| Conventional + AlphaEarth Embeddings | raw_probability                     | 2232492      | 0.1943 | 0.0591 | 0.9201 | 0.0000                     |
| Conventional + AlphaEarth Embeddings | transfer_confidence                 | 2232492      | 0.2418 | 0.2454 | 0.4890 | 9.4185                     |
| Conventional + AlphaEarth Embeddings | in_area_of_applicability            | 2232492      | 0.9067 | 1.0000 | 1.0000 | 9.3348                     |
| Conventional + AlphaEarth Embeddings | reliability_weighted_probability    | 2232492      | 0.0339 | 0.0118 | 0.1627 | 10.3110                    |
| Final fused                          | final_fused_reliability_aware_score | 2232492      | 0.0258 | 0.0110 | 0.1111 | 10.5007                    |
| Final fused                          | final_reliable_high_mask            | 2232492      | 0.0010 | 0.0000 | 0.0000 | 99.9029                    |
| Final fused                          | final_reliable_very_high_mask       | 2232492      | 0.0043 | 0.0000 | 0.0000 | 99.5701                    |
| Final fused                          | final_uncertain_high_mask           | 2232492      | 0.3081 | 0.0000 | 1.0000 | 69.1881                    |
| Final fused                          | field_verification_priority_score   | 2232492      | 0.1934 | 0.0473 | 0.8533 | 0.0000                     |
| Final fused                          | model_disagreement_std              | 2232492      | 0.0673 | 0.0259 | 0.2634 | 0.0512                     |

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
