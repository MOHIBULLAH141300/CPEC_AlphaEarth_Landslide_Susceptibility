# Manuscript v3 nested spatial-CV analysis

## Design freeze

- Samples: 3316 (1508 positives; 1808 controls).
- Validation: five outer one-degree spatial-block folds.
- Stacking: base learners generated within outer-training data using inner spatial folds.
- Separation buffer: 20 km between training and validation samples.
- Aspect: sine and cosine components.
- Land cover, lithology and soil: categorical one-hot encoding.
- Metric uncertainty: 95% confidence intervals from spatial-block bootstrap.
- Output interpretation: case-control susceptibility score, not population occurrence probability.

## Conventional predictor source fields

- `elevation_m`
- `slope_deg`
- `aspect_deg`
- `rain_monsoon_total`
- `rain_max_1day`
- `ndvi_median`
- `ndvi_amplitude`
- `modis_lc_type1`
- `log1p_dist_road_m`
- `log1p_dist_river_m`
- `log1p_dist_fault_m`
- `lithology_code`
- `profile_curvature`
- `plan_curvature`
- `tri`
- `twi`
- `valley_depth`
- `soil_type`
- `eq_density_ms5`

## Primary stacked-ensemble performance

| Feature set | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Conventional | 0.957 (0.942-0.971) | 0.948 (0.914-0.968) | 0.078 | 0.895 | 0.903 |
| AlphaEarth Embeddings | 0.949 (0.930-0.964) | 0.935 (0.907-0.955) | 0.088 | 0.872 | 0.881 |
| Conventional + AlphaEarth Embeddings | 0.966 (0.956-0.976) | 0.959 (0.942-0.971) | 0.066 | 0.903 | 0.911 |

## Reproducibility

- Python: 3.13.12
- Runtime: 4.7 minutes
- Random seed: 141300
- Feature sets: Conventional, AlphaEarth Embeddings, Conventional + AlphaEarth Embeddings
