# Conventional Reliability-Weighted Raster Diagnosis

## File checked

`D:\DING PROJECT\04_maps\reliability_aware_susceptibility_2018_250m\cpec_2018_conventional_reliability_weighted_probability_250m.tif`

## Main conclusion

The Conventional reliability-weighted raster is not a raw probability raster. It is a confidence-adjusted susceptibility product:

`reliability-weighted probability = raw Conventional probability x Conventional transfer confidence`

Because the Conventional Area of Applicability / transfer-confidence surface is low over large parts of the official CPEC boundary, the reliability-weighted Conventional raster has many near-zero values. This is expected for the reliability product, but it should not be used as the main susceptibility probability map.

## Diagnostic statistics

| Product | Mean | Median | P95 | Near-zero pixels <= 0.001 |
|---|---:|---:|---:|---:|
| Raw Conventional probability | 0.2292 | 0.2317 | 0.4838 | 0.00% |
| Conventional transfer confidence | 0.0711 | 0.0000 | 0.3486 | 65.63% |
| Conventional reliability-weighted probability | 0.0123 | 0.0000 | 0.0711 | 66.77% |

## How to present this in the paper

Use the raw stacked probability rasters as the main susceptibility maps:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

Use the reliability-weighted rasters as secondary confidence-adjusted outputs. They are useful for identifying locations where the model prediction is both high and located inside an applicable predictor space, but they should not replace the raw probability maps.

## Domain-specific handling

The new domain-specific results package provides separate clipped rasters for each transfer domain:

`D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\10_domain_specific_2018_results`

Each domain folder contains:

- raw probability maps for all three feature sets,
- transfer-confidence and Area-of-Applicability maps,
- reliability-weighted maps,
- leave-one-domain-out metrics,
- uncertainty and road exposure outputs.
