# 2018 V3 Domain Results: Punjab-Sindh lowland corridor

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

- Conventional: ROC-AUC 0.990, PR-AUC 0.950, F1 0.785, Brier 0.055
- AlphaEarth Embeddings: ROC-AUC 0.965, PR-AUC 0.889, F1 0.812, Brier 0.055
- Conventional + AlphaEarth Embeddings: ROC-AUC 0.990, PR-AUC 0.954, F1 0.878, Brier 0.033

## Probability and reliability statistics

| feature_set                          | product                             | valid_pixels | mean   | median | p95    | near_zero_percent_le_0_001 |
| ------------------------------------ | ----------------------------------- | ------------ | ------ | ------ | ------ | -------------------------- |
| Conventional                         | raw_probability                     | 6390937      | 0.0693 | 0.0253 | 0.3064 | 0.0000                     |
| Conventional                         | transfer_confidence                 | 6390937      | 0.0033 | 0.0000 | 0.0000 | 98.3983                    |
| Conventional                         | in_area_of_applicability            | 6390937      | 0.0162 | 0.0000 | 0.0000 | 98.3835                    |
| Conventional                         | reliability_weighted_probability    | 6390937      | 0.0010 | 0.0000 | 0.0000 | 98.5991                    |
| AlphaEarth Embeddings                | raw_probability                     | 6390937      | 0.0800 | 0.0376 | 0.3248 | 0.0000                     |
| AlphaEarth Embeddings                | transfer_confidence                 | 6390937      | 0.4809 | 0.5195 | 0.7334 | 2.7954                     |
| AlphaEarth Embeddings                | in_area_of_applicability            | 6390937      | 0.9723 | 1.0000 | 1.0000 | 2.7748                     |
| AlphaEarth Embeddings                | reliability_weighted_probability    | 6390937      | 0.0304 | 0.0213 | 0.0776 | 3.0433                     |
| Conventional + AlphaEarth Embeddings | raw_probability                     | 6390937      | 0.0556 | 0.0230 | 0.1728 | 0.0000                     |
| Conventional + AlphaEarth Embeddings | transfer_confidence                 | 6390937      | 0.4520 | 0.4809 | 0.6267 | 1.8008                     |
| Conventional + AlphaEarth Embeddings | in_area_of_applicability            | 6390937      | 0.9821 | 1.0000 | 1.0000 | 1.7936                     |
| Conventional + AlphaEarth Embeddings | reliability_weighted_probability    | 6390937      | 0.0205 | 0.0117 | 0.0453 | 1.9325                     |
| Final fused                          | final_fused_reliability_aware_score | 6390937      | 0.0173 | 0.0114 | 0.0343 | 1.9568                     |
| Final fused                          | final_reliable_high_mask            | 6390937      | 0.0005 | 0.0000 | 0.0000 | 99.9516                    |
| Final fused                          | final_reliable_very_high_mask       | 6390937      | 0.0091 | 0.0000 | 0.0000 | 99.0913                    |
| Final fused                          | final_uncertain_high_mask           | 6390937      | 0.0527 | 0.0000 | 1.0000 | 94.7293                    |
| Final fused                          | field_verification_priority_score   | 6390937      | 0.0414 | 0.0121 | 0.2098 | 0.0000                     |
| Final fused                          | model_disagreement_std              | 6390937      | 0.0252 | 0.0082 | 0.1246 | 0.0100                     |

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
