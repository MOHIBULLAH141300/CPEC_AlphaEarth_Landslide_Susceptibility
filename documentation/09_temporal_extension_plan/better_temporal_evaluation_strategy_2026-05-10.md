# Better Temporal Evaluation Strategy for CPEC/KKH LSM

## Decision

Do not make the paper a year-by-year atlas of 2017-2024 susceptibility maps.

Instead, use 2017-2024 annual predictors as a compact temporal stress-test and
dynamic-change layer around the validated 2018 V3 model. The main paper should
remain centered on the 2018 V3 baseline, transferability, AlphaEarth added
value, reliability, and infrastructure exposure. Temporal analysis should be
presented as an additional contribution, not as eight separate modelling
studies.

## Why this is better

Full annual susceptibility mapping from 2017 to 2024 is literature-backed, but
only when the study has annual or event-based inventories, temporal validation,
or clear dynamic triggers. Our project currently has a strong 2018 factor table
and fixed 2018 trained models, while annual landslide labels are not complete
for every year. Therefore, a fixed-model temporal transfer strategy is more
defensible than retraining or claiming independent annual accuracy for each
year.

## Literature-backed basis

Recent high-impact work supports dynamic susceptibility, but it also stresses
proper space-time validation and dynamic covariates:

- Wang et al. developed dynamic landslide susceptibility from annual units
  spanning 2013-2021 and validated it with space-time cross-validation before
  moving toward risk forecasting:
  https://www.sciencedirect.com/science/article/pii/S1674987123002323
- Recent npj Natural Hazards work shows that temporal precipitation should be
  treated as compound long-term plus short-term forcing and evaluated with
  random, spatial, and temporal CV, not only random CV:
  https://www.nature.com/articles/s44304-026-00181-z
- Catena 2024 work on space-time landslide modelling emphasizes dynamic
  covariates, annual variation, and full space/time validation routines:
  https://www.sciencedirect.com/science/article/pii/S0341816224006490
- Environmental Modelling & Software 2024 introduced a Markov-switching
  spatio-temporal susceptibility model and used spatiotemporal cross-validation:
  https://www.sciencedirect.com/science/article/pii/S1364815223002785
- The teacher-sent 2024-2025 transfer-learning papers support the stronger
  cross-domain / transferability framing, especially for data-limited mountain
  regions and cross-regional extrapolation:
  https://www.nature.com/articles/s41598-024-76541-4
  https://www.sciencedirect.com/science/article/pii/S1674987125000581
  https://www.sciencedirect.com/science/article/pii/S1674987125002178
- AlphaEarth/Satellite Embedding is annual from 2017 onward and provides
  64-dimensional annual surface-condition embeddings, which justifies the
  2017-2024 temporal window:
  https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

## Recommended temporal framing

Use this phrase:

**Temporal transfer and dynamic susceptibility stress-testing, 2017-2024**

Avoid saying:

**Eight independently validated annual susceptibility models**

## What we should generate

### Main-text temporal products

1. **Annual fused probability maps, 2017-2024**
   - Use only the final `Conventional + AlphaEarth Embeddings` stacked model
     in the main paper.
   - Keep Conventional and AlphaEarth-only annual maps as supplementary or
     internal sensitivity outputs.

2. **Mean dynamic susceptibility, 2017-2024**
   - Shows stable long-term susceptibility under annual dynamic predictors.

3. **Temporal variability / standard deviation map**
   - Shows where susceptibility changes most across years.

4. **Trend map**
   - Pixel-wise Theil-Sen or linear trend of annual susceptibility.
   - Use significance cautiously; if we do not have enough years for strong
     inference, present it as directional trend.

5. **Persistent high-susceptibility zones**
   - Pixels above a high-probability threshold in at least N of 8 years.
   - More defensible for planning than showing eight separate maps.

6. **Year-to-year change maps**
   - Keep all year-to-year deltas in the project folder.
   - In the paper, show only the largest-change year pair and mention the full
     set is in supplementary material.

7. **KKH/CPEC temporal exposure summary**
   - Road length exposed to persistent high susceptibility.
   - Road segments where susceptibility increased most.
   - This keeps the temporal analysis applied and CPEC-specific.

### Supplementary products

- Annual probability rasters for 2017-2024.
- Annual road exposure tables.
- Annual domain summaries.
- All year-to-year change maps.

## How to select display years

Do not arbitrarily show all years.

Select 3-4 display years using objective criteria:

1. `2017`: first AlphaEarth year and temporal start.
2. `2018`: model baseline year.
3. The year with the largest CPEC-wide dynamic-factor anomaly or largest
   susceptibility-change magnitude.
4. `2024`: latest complete annual end year.

If the largest-change year is 2024, show only three years: 2017, 2018, 2024.

## What changes by year

Static or semi-static factors remain fixed:

- elevation,
- slope,
- aspect,
- curvature,
- terrain ruggedness,
- TWI,
- valley depth,
- lithology,
- soil,
- road distance,
- river distance,
- fault distance,
- earthquake density unless a new annual seismic catalog is added.

Dynamic annual factors change:

- monsoon rainfall total,
- annual / monsoon extreme rainfall metric,
- NDVI median,
- NDVI amplitude,
- land cover,
- AlphaEarth 64 annual embedding bands.

This is reviewer-safe because the temporal signal is limited to factors that
can actually vary annually.

## Validation language

Because we do not have complete independent annual landslide inventories for
each year, use this wording:

**The temporal maps are model-transfer/dynamic-covariate scenario outputs from
the validated 2018 V3 spatial model. They are evaluated through temporal
plausibility, annual predictor anomaly consistency, road-exposure consistency,
and domain transferability, not through separate annual retraining.**

## Final paper structure

1. 2018 V3 model development and spatial/domain validation.
2. AlphaEarth added value and explainability.
3. Area of Applicability and reliability-aware 2018 map.
4. Compact 2017-2024 temporal transfer analysis:
   - mean,
   - variability,
   - trend,
   - persistent hotspots,
   - KKH exposure change.
5. Supplementary annual rasters and tables.

## Practical processing rule

Still process one year at a time in:

`D:\DING PROJECT\04_maps\annual_dynamic_2017_2024`

But publish only the compact temporal products above. This prevents the paper
from becoming overwhelmed while keeping the annual evidence available for
reviewers.
