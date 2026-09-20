# CPEC 2018 Reliability-Aware Susceptibility and Exposure Outputs

Date: 2026-05-10

## Purpose

This step converts the 2018 stacked-ensemble probability maps and transfer-confidence maps into decision-ready products. The goal is to separate high susceptibility that is reliable from high susceptibility that should be treated as uncertain and prioritized for field verification.

## Corrected Public Domain Labels

- Khyber Pakhtunkhwa and Azad Jammu and Kashmir are displayed as `KP-AJK` for the project figures and tables.
- Former FATA is not shown as a separate current public domain because it was merged with Khyber Pakhtunkhwa through Pakistan's 25th Constitutional Amendment in 2018; legacy GADM polygons labelled FATA are assigned to the KP side of the `KP-AJK` analysis domain.
- AJK is grouped with KP only as an analysis-domain decision for the northern-western mountainous corridor and to avoid a statistically small held-out test region; it is not presented as the same administrative unit.
- Punjab, Islamabad, and Sindh are displayed as `Punjab-Sindh lowland corridor`. Islamabad is administratively a federal territory, but it is absorbed into this lowland corridor only for statistical grouping because there was only one Islamabad sample.
- Official sources commonly use `KP` for Khyber Pakhtunkhwa; this project therefore uses `KP-AJK` rather than `KP-AJK`.

## Why These Subdomains Are Defensible

These subdomains are used as **analysis domains for model transferability**, not as a replacement for official political boundaries. The grouping was necessary because leave-one-domain-out testing requires both landslide and non-landslide samples in every held-out region. Small units such as Islamabad and legacy FATA cannot support stable independent testing alone.

The design follows spatial-validation and transferability literature: spatially structured data should be evaluated with block/group validation rather than ordinary random splits, and the area where model error is expected to transfer should be explicitly assessed.

Key support:

- Roberts et al. (2017), Ecography, cross-validation strategies for spatial/hierarchical data: https://doi.org/10.1111/ecog.02881
- Meyer and Pebesma (2021), area of applicability for spatial prediction models: https://doi.org/10.1111/2041-210X.13650
- Official KP government portal: https://kp.gov.pk/
- Pakistan Code / Constitution source for the 25th Amendment context: https://pakistancode.gov.pk/
- AJK official portal: https://ajk.gov.pk/

## Final Rules

```json
{
  "probability_p80_threshold": 0.12633591890335083,
  "probability_p90_threshold": 0.4417957067489624,
  "fused_transfer_confidence_p25": 0.3000139594078064,
  "fused_transfer_confidence_p50": 0.4041162133216858,
  "uncertainty_std_p50": 0.01357282418757677,
  "uncertainty_std_p75": 0.05905983969569206,
  "reliable_high_rule": "probability>=P80, confidence>=P50, inside AOA, disagreement<=P50",
  "reliable_very_high_rule": "probability>=P90, confidence>=P50, inside AOA",
  "uncertain_high_rule": "probability>=P80 and (outside AOA or confidence<P25 or disagreement>=P75)"
}
```

The final score is:

`reliability-aware susceptibility = fused probability x fused transfer confidence x (1 - normalized model disagreement)`

## Feature-Set Summary

| feature_set | study_area_valid_pixels | mean_probability | mean_transfer_confidence | mean_reliability_weighted_probability | p80_reliability_weighted_probability | p90_reliability_weighted_probability | inside_aoa_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Conventional | 19996413 | 0.158 | 0.071 | 0.012 | 0.008 | 0.023 | 34.432 |
| AlphaEarth Embeddings | 19996413 | 0.173 | 0.411 | 0.060 | 0.063 | 0.190 | 95.800 |
| Conventional + AlphaEarth Embeddings | 19996413 | 0.132 | 0.388 | 0.042 | 0.036 | 0.114 | 97.662 |

## Subdomain Area Summary

| domain | area_km2 | mean_fused_probability | mean_fused_transfer_confidence | mean_reliability_aware_score | reliable_high_area_km2 | reliable_high_area_percent | reliable_very_high_area_km2 | uncertain_high_area_km2 | uncertain_high_area_percent | outside_fused_aoa_area_km2 | outside_fused_aoa_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kashgar (Xinjiang, China) | 182260.746 | 0.084 | 0.373 | 0.020 | 168.665 | 0.093 | 936.610 | 19123.850 | 10.493 | 2056.177 | 1.128 |
| Gilgit-Baltistan | 68129.421 | 0.184 | 0.328 | 0.037 | 430.279 | 0.632 | 1022.174 | 14670.597 | 21.533 | 1535.077 | 2.253 |
| KP-AJK | 115200.634 | 0.194 | 0.242 | 0.026 | 110.264 | 0.096 | 492.694 | 35476.156 | 30.795 | 10763.635 | 9.343 |
| Balochistan | 345152.246 | 0.205 | 0.398 | 0.057 | 1130.128 | 0.327 | 21986.973 | 98490.711 | 28.535 | 3935.930 | 1.140 |
| Punjab-Sindh lowland corridor | 347990.657 | 0.056 | 0.452 | 0.017 | 171.964 | 0.049 | 3213.619 | 18249.222 | 5.244 | 6202.991 | 1.783 |

## Road Exposure Summary

| road_group | domain | segment_count | total_length_km | reliable_high_weighted_length_km | reliable_very_high_weighted_length_km | uncertain_high_weighted_length_km | outside_aoa_weighted_length_km | mean_probability_length_weighted | mean_transfer_confidence_length_weighted | mean_reliability_aware_score_length_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All clipped roads | All domains | 1803 | 17066.007 | 4.636 | 41.455 | 10595.047 | 14695.393 | 0.365 | 0.020 | 0.006 |
| All clipped roads | Kashgar (Xinjiang, China) | 163 | 1606.044 | 2.818 | 6.818 | 1437.589 | 1360.769 | 0.589 | 0.020 | 0.014 |
| All clipped roads | Gilgit-Baltistan | 44 | 418.502 | 0.000 | 0.000 | 418.502 | 306.245 | 0.985 | 0.027 | 0.026 |
| All clipped roads | KP-AJK | 309 | 2869.135 | 0.909 | 1.909 | 2382.883 | 2723.813 | 0.570 | 0.005 | 0.004 |
| All clipped roads | Balochistan | 441 | 4287.787 | 0.909 | 32.727 | 4050.236 | 2546.621 | 0.500 | 0.065 | 0.013 |
| All clipped roads | Punjab-Sindh lowland corridor | 846 | 7884.539 | 0.000 | 0.000 | 2305.837 | 7757.944 | 0.138 | 0.001 | 0.000 |
| KKH proxy routes (N35 / 314) | All domains | 148 | 1448.223 | 3.727 | 7.727 | 1380.132 | 1112.784 | 0.748 | 0.030 | 0.024 |
| KKH proxy routes (N35 / 314) | Kashgar (Xinjiang, China) | 72 | 715.403 | 2.818 | 6.818 | 654.676 | 539.676 | 0.603 | 0.037 | 0.026 |
| KKH proxy routes (N35 / 314) | Gilgit-Baltistan | 41 | 388.502 | 0.000 | 0.000 | 388.502 | 279.063 | 0.985 | 0.029 | 0.028 |
| KKH proxy routes (N35 / 314) | KP-AJK | 33 | 324.318 | 0.909 | 0.909 | 317.863 | 274.045 | 0.810 | 0.017 | 0.016 |
| KKH proxy routes (N35 / 314) | Punjab-Sindh lowland corridor | 2 | 20.000 | 0.000 | 0.000 | 19.091 | 20.000 | 0.346 | 0.000 | 0.000 |

## Top 15 Reliability-Aware Road Hotspots

| hotspot_rank | route_code | domain | is_kkh_proxy | length_km | mean_fused_probability | mean_fused_transfer_confidence | mean_final_reliability_aware_score | reliable_high_share | uncertain_high_share | reliability_aware_road_priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.989 | 0.246 | 0.232 | 0.091 | 0.636 | 0.277 |
| 2 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.966 | 0.314 | 0.213 | 0.000 | 0.800 | 0.273 |
| 3 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.985 | 0.165 | 0.160 | 0.091 | 0.727 | 0.206 |
| 4 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.881 | 0.142 | 0.128 | 0.100 | 0.900 | 0.178 |
| 5 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.974 | 0.207 | 0.151 | 0.000 | 0.818 | 0.151 |
| 6 | N35 | Gilgit-Baltistan | True | 10.000 | 0.994 | 0.146 | 0.142 | 0.000 | 1.000 | 0.142 |
| 7 | N35 | KP-AJK | True | 10.000 | 0.993 | 0.087 | 0.085 | 0.091 | 0.909 | 0.130 |
| 8 | N50 | KP-AJK | False | 10.000 | 0.907 | 0.113 | 0.107 | 0.000 | 0.800 | 0.122 |
| 9 | M8 | Balochistan | False | 10.000 | 0.868 | 0.081 | 0.074 | 0.091 | 0.909 | 0.119 |
| 10 | N35 | Gilgit-Baltistan | True | 10.000 | 0.993 | 0.120 | 0.117 | 0.000 | 1.000 | 0.117 |
| 11 | 314 | Kashgar (Xinjiang, China) | True | 10.000 | 0.982 | 0.129 | 0.116 | 0.000 | 1.000 | 0.116 |
| 12 | N85 | Balochistan | False | 10.000 | 0.529 | 0.330 | 0.054 | 0.000 | 1.000 | 0.114 |
| 13 | N65 | Balochistan | False | 10.000 | 0.984 | 0.129 | 0.113 | 0.000 | 0.818 | 0.113 |
| 14 | N85 | Balochistan | False | 10.000 | 0.844 | 0.233 | 0.106 | 0.000 | 1.000 | 0.106 |
| 15 | N85 | Balochistan | False | 10.000 | 0.978 | 0.135 | 0.102 | 0.000 | 1.000 | 0.102 |

## Main Raster Outputs

- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_alphaearth_embeddings_reliability_weighted_probability_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_conventional_alphaearth_embeddings_reliability_weighted_probability_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_conventional_reliability_weighted_probability_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_final_field_verification_priority_score_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_final_fused_reliability_aware_susceptibility_score_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_final_reliable_high_susceptibility_mask_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_final_reliable_very_high_susceptibility_mask_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_final_uncertain_high_susceptibility_mask_250m.tif`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\04_probability_maps_250m\04_reliability_aware_final_outputs\cpec_2018_reliability_aware_thresholds.json`

## Road Outputs

- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\field_verification_priority_road_segments_top50_250m.csv`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_exposure_summary_by_domain_250m.csv`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_hotspot_segments_top50_250m.csv`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_hotspot_segments_top50_250m.geojson`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_sample_points_250m.csv`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_segment_exposure_all_250m.csv`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\05_road_exposure_outputs\reliability_aware_2018\reliability_aware_road_segment_exposure_all_250m.gpkg`

## Figures

- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\03_figures\07_reliability_aware_final_outputs\figure_cpec_2018_reliability_aware_area_by_subdomain.png`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\03_figures\07_reliability_aware_final_outputs\figure_cpec_2018_reliability_aware_road_hotspots.png`
- `D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\03_figures\07_reliability_aware_final_outputs\figure_cpec_2018_reliability_aware_susceptibility_outputs.png`

## Interpretation

- The fused reliability-aware score is the preferred final planning surface because it keeps probability continuous but penalizes low transfer confidence and high model disagreement.
- Reliable high-susceptibility zones are the strongest areas to report as robust hotspots.
- Uncertain high-susceptibility zones are not discarded; they become priority areas for field checking, inventory improvement, and local validation.
- Road exposure is a planning overlay, not an independent validation, because distance to roads is one of the conditioning factors.

