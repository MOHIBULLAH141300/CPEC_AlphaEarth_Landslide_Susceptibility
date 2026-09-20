# 250 m Stacked Ensemble Uncertainty, Added-Value, And Road Exposure Outputs

Date: 2026-05-03

## Purpose

This package converts the three 2018 **Spatial-CV Stacked Ensemble** probability rasters into publishable diagnostic products at the true 250 m modelling grid.

The core probability maps remain:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

The fused map, **Conventional + AlphaEarth Embeddings**, is used as the primary planning raster.

## Method Summary

1. Rasterized the official CPEC study-area boundary to the 250 m probability grid.
2. Computed mean probability, three-model disagreement, entropy, and a continuous reliability score.
3. Computed AlphaEarth added-value diagnostics using signed probability differences.
4. Created optional fused-probability quantile priority zones for exposure summaries.
5. Clipped the official 2018 CPEC road network to the study area.
6. Split roads into 10 km segments and sampled raster values every 1 km along each segment.
7. Ranked road hotspots by high fused probability, low model disagreement, and positive AlphaEarth added value.

The uncertainty map here is a **feature-set/model-disagreement uncertainty proxy** from the three stacked outputs. It is not yet a full bootstrap or repeated non-landslide sampling uncertainty product.

## Key Thresholds

- Fused probability P80: `0.126336`
- Fused probability P90: `0.441796`
- Uncertainty std P50: `0.013573`
- Uncertainty std P75: `0.059060`

Reliable high-probability mask: fused probability >= P80 and disagreement std <= P50.

Uncertain high-probability mask: fused probability >= P80 and disagreement std >= P75.

## Raster Diagnostics

| name                                                     | count    | min     | p50     | p80    | p90    | max    | mean    | std    |
| -------------------------------------------------------- | -------- | ------- | ------- | ------ | ------ | ------ | ------- | ------ |
| Conventional stacked probability                         | 19996413 | 0.0159  | 0.0459  | 0.2167 | 0.5523 | 0.9987 | 0.1584  | 0.2417 |
| AlphaEarth Embeddings stacked probability                | 19996413 | 0.0319  | 0.0482  | 0.2325 | 0.6410 | 0.9820 | 0.1735  | 0.2555 |
| Conventional + AlphaEarth Embeddings stacked probability | 19996413 | 0.0204  | 0.0313  | 0.1263 | 0.4418 | 0.9951 | 0.1322  | 0.2369 |
| Mean probability                                         | 19996413 | 0.0239  | 0.0464  | 0.2184 | 0.5045 | 0.9914 | 0.1547  | 0.2282 |
| Model disagreement std                                   | 19996413 | 0.0000  | 0.0136  | 0.0829 | 0.1615 | 0.4408 | 0.0508  | 0.0746 |
| Model disagreement range                                 | 19996413 | 0.0000  | 0.0318  | 0.1886 | 0.3683 | 0.9603 | 0.1162  | 0.1707 |
| Fused reliability score                                  | 19996413 | 0.0027  | 0.0296  | 0.0904 | 0.2566 | 0.9829 | 0.1011  | 0.1868 |
| Fused minus Conventional                                 | 19996413 | -0.9254 | -0.0038 | 0.0050 | 0.0339 | 0.9247 | -0.0263 | 0.1332 |
| AlphaEarth minus Conventional                            | 19996413 | -0.9603 | 0.0142  | 0.0455 | 0.1858 | 0.9476 | 0.0151  | 0.1951 |

## Road Exposure Summary

| road_group                   | segment_count | total_length_km | weighted_length_probability_p80_km | weighted_length_probability_p90_km | length_majority_p80_km | length_majority_p90_km | mean_probability_length_weighted | mean_uncertainty_length_weighted | mean_reliability_length_weighted |
| ---------------------------- | ------------- | --------------- | ---------------------------------- | ---------------------------------- | ---------------------- | ---------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| All clipped roads            | 1803          | 17066.007       | 10620.411                          | 5521.259                           | 10472.280              | 5597.829               | 0.365                            | 0.183                            | 0.202                            |
| KKH proxy routes (N35 / 314) | 148           | 1448.223        | 1396.132                           | 1077.545                           | 1398.223               | 1072.820               | 0.748                            | 0.178                            | 0.553                            |

## Top 15 Road Hotspot Segments

| hotspot_rank | route_code | is_kkh_proxy | length_km | mean_probability | mean_uncertainty_std | mean_reliability_score | mean_added_value_fused_minus_conventional | high80_share | high90_share | priority_score |
| ------------ | ---------- | ------------ | --------- | ---------------- | -------------------- | ---------------------- | ----------------------------------------- | ------------ | ------------ | -------------- |
| 1            | N35        | True         | 10.0000   | 0.9937           | 0.0079               | 0.9770                 | -0.0033                                   | 1.0000       | 1.0000       | 1.0770         |
| 2            | N35        | True         | 10.0000   | 0.9945           | 0.0087               | 0.9761                 | -0.0040                                   | 1.0000       | 1.0000       | 1.0761         |
| 3            | N35        | True         | 10.0000   | 0.9942           | 0.0086               | 0.9760                 | -0.0042                                   | 1.0000       | 1.0000       | 1.0760         |
| 4            | N35        | True         | 10.0000   | 0.9944           | 0.0088               | 0.9758                 | -0.0042                                   | 1.0000       | 1.0000       | 1.0758         |
| 5            | N35        | True         | 10.0000   | 0.9941           | 0.0089               | 0.9754                 | -0.0044                                   | 1.0000       | 1.0000       | 1.0754         |
| 6            | N35        | True         | 10.0000   | 0.9940           | 0.0089               | 0.9754                 | -0.0042                                   | 1.0000       | 1.0000       | 1.0754         |
| 7            | N35        | True         | 10.0000   | 0.9936           | 0.0087               | 0.9753                 | -0.0042                                   | 1.0000       | 1.0000       | 1.0753         |
| 8            | N35        | True         | 10.0000   | 0.9941           | 0.0090               | 0.9752                 | -0.0041                                   | 1.0000       | 1.0000       | 1.0752         |
| 9            | N15        | False        | 10.0000   | 0.9940           | 0.0091               | 0.9748                 | -0.0034                                   | 1.0000       | 1.0000       | 1.0748         |
| 10           | N35        | True         | 10.0000   | 0.9937           | 0.0090               | 0.9747                 | -0.0039                                   | 1.0000       | 1.0000       | 1.0747         |
| 11           | N35        | True         | 7.9283    | 0.9935           | 0.0092               | 0.9741                 | -0.0041                                   | 1.0000       | 1.0000       | 1.0741         |
| 12           | N35        | True         | 10.0000   | 0.9936           | 0.0093               | 0.9741                 | -0.0046                                   | 1.0000       | 1.0000       | 1.0741         |
| 13           | N35        | True         | 10.0000   | 0.9935           | 0.0094               | 0.9736                 | -0.0045                                   | 1.0000       | 1.0000       | 1.0736         |
| 14           | N35        | True         | 4.3177    | 0.9929           | 0.0094               | 0.9730                 | -0.0044                                   | 1.0000       | 1.0000       | 1.0730         |
| 15           | N35        | True         | 10.0000   | 0.9939           | 0.0099               | 0.9730                 | -0.0046                                   | 1.0000       | 1.0000       | 1.0730         |

## Raster Outputs

- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_stacked_mean_probability_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_stacked_probability_disagreement_std_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_stacked_probability_disagreement_range_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_fused_probability_entropy_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_fused_reliability_score_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_fused_minus_conventional_added_value_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_fused_minus_alphaearth_probability_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_alphaearth_minus_conventional_probability_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_alphaearth_conventional_disagreement_abs_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_reliable_high_probability_mask_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_uncertain_high_probability_mask_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\cpec_2018_fused_probability_quantile_priority_zones_250m.tif`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\stacked_250m_thresholds_and_diagnostics.json`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\stacked_250m_raster_diagnostics.csv`

## Road Outputs

- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_segment_exposure_all_250m.csv`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_hotspot_segments_top50_250m.csv`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_exposure_summary_250m.csv`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_exposure_sample_points_250m.csv`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_hotspot_segments_top50_250m.geojson`
- `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs\road_segment_exposure_all_250m.gpkg`
- `D:\DING PROJECT\05_reports\figures\fig_cpec_2018_road_exposure_hotspots_250m.png`

## Figure Outputs

- `D:\DING PROJECT\05_reports\figures\fig_cpec_2018_stacked_uncertainty_reliability_250m.png`
- `D:\DING PROJECT\05_reports\figures\fig_cpec_2018_alphaearth_added_value_250m.png`
- `D:\DING PROJECT\05_reports\figures\fig_cpec_2018_fused_quantile_priority_zones_250m.png`

## Interpretation For Paper

- The fused stacked model is treated as the best planning product because it combines physically interpretable conditioning factors with AlphaEarth embedding information.
- The disagreement map shows where the three feature sets give inconsistent susceptibility probabilities; these zones should be described as lower-confidence prediction areas.
- The reliable high-probability mask highlights locations where the fused model is high and the feature-set disagreement is low.
- The uncertain high-probability mask is useful for field verification because the model suggests possible hazard but the feature sets disagree.
- The fused-minus-Conventional map directly supports the AlphaEarth added-value argument. Positive values show where the fused model raises susceptibility relative to conventional predictors.
- The road hotspot output makes the work CPEC/KKH-applied by translating susceptibility into ranked infrastructure segments.

## Notes

- The optional five quantile zones are included only for map communication and road exposure summaries. They are not used as model classes and should not replace continuous probability maps.
- KKH proxy routes are identified from route codes `N35`, `314`, or `G314` where present in the official 2018 road layer.
- Outside-boundary raster cells are written as nodata in derived products. The no-nodata requirement is preserved inside the official study-area mask.
- Because distance to roads is also an input conditioning factor, the road exposure analysis should be described as a planning-priority overlay rather than a fully independent validation layer. A later road-distance ablation can quantify how sensitive the road hotspots are to that predictor.

## Literature Support

- Uncertainty and confidence maps are supported by recent uncertainty-focused LSM work, including Monte Carlo/confidence-map approaches for sampling randomness: [Quantifying uncertainty in landslide susceptibility mapping due to sampling randomness](https://www.sciencedirect.com/science/article/abs/pii/S2212420924007283).
- Sampling and non-landslide/control point choices are a recognized source of LSM uncertainty: [Scientific Reports 2024 sampling-resolution study](https://www.nature.com/articles/s41598-024-52145-w) and [ScienceDirect 2024 non-landslide sampling comparison](https://www.sciencedirect.com/science/article/pii/S1574954124001250).
- Stacking/heterogeneous ensembles and optimized sampling are consistent with recent LSM model-design literature: [Remote Sensing 2024 optimized sampling and heterogeneous ensemble ML](https://www.mdpi.com/2072-4292/16/19/3663).
- The AlphaEarth/Satellite Embedding novelty is supported by the official Earth Engine dataset and tutorial: [Satellite Embedding V1 Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) and [Earth Engine Satellite Embedding tutorial](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction).
- The foundation-model framing is supported by the AlphaEarth Foundations preprint: [AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291).

## Reproducibility

- Script: `D:\DING PROJECT\06_scripts\python\create_250m_stacked_uncertainty_added_value_and_road_hotspots.py`
- Source rasters: `D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble`
- Output folder: `D:\DING PROJECT\04_maps\stacked_ensemble_250m_high_impact_outputs`
