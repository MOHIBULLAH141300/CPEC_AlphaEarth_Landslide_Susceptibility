# Literature-Based Factor Audit And v3 Revision

Date: 2026-05-03

## Why this audit was needed

The active 2018 sample table did not include:

- distance to roads
- distance to streams/rivers
- distance to active/geological faults

These are classic landslide-conditioning factors and should be included in the corrected model. A literature-based audit also shows that several other factor groups should be added or at least tested before the final CPEC/KKH model is fixed.

## Current active factors already included

Current active sample table:

- `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v2.csv`

Included groups:

- elevation
- slope
- aspect
- rainfall total/extremes
- NDVI/EVI vegetation indicators
- LST thermal indicators
- MODIS land cover
- AlphaEarth annual embeddings

These remain useful and should not be discarded.

## Important missing factors to add in v3

### 1. Geological and structural controls

Add:

- lithology / geological unit
- distance to active faults
- fault density, optional after collinearity testing

Reason:

Lithology controls material strength and weathering, while faults represent fractured/tectonically weakened rock. These are expected predictors in mountain landslide susceptibility studies and are especially important in the Himalaya-Karakoram-Hindu Kush setting.

Local data found:

- active faults: `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\中巴经济走廊1：25万活动断裂带（1964年）\活动断裂带.shp`
- additional faults: `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\FAULTS.shp`

### 2. Hydrological controls

Add:

- distance to streams/rivers
- drainage density, optional after collinearity testing
- soil moisture if reliable source is available

Reason:

Streams/rivers influence slope instability through valley incision, toe erosion, saturation, and runoff concentration. Recent dynamic susceptibility literature also supports adding hydrological state variables such as precipitation and soil moisture where possible.

Local data found:

- river network: `D:\CPEC data\中巴经济走廊1：25万水文数据（2000年）\中巴经济走廊1：25万水文数据（2000年）\河网.shp`
- river-distance raster: `D:\CPEC data\中巴经济走廊30m距离河流的距离（2009年）\中巴经济走廊30m距离河流的距离（2009年）\river_buffer.tif`

### 3. Infrastructure / human disturbance

Add:

- distance to roads
- road density, optional after collinearity testing
- separate KKH/major-road proximity if attributes allow

Reason:

Road cutting, blasting, slope undercutting, drainage modification, and construction disturbance are highly relevant for KKH/CPEC. Since the research is corridor-focused, road proximity is not only an exposure variable but also a susceptibility-conditioning factor.

Local data found:

- road network: `D:\CPEC data\中巴经济走廊1：25万路网数据（2018年）\中巴经济走廊1：25万路网数据（2018年）\路网.shp`

### 4. Additional terrain morphology

Add/test:

- profile curvature
- plan curvature
- terrain ruggedness index
- TWI
- relief / valley depth
- LS factor, only if not redundant

Reason:

The current GEE stack uses elevation, slope, and aspect only. Recent landslide-susceptibility studies commonly include additional terrain-shape and hydrological terrain indices because they represent flow convergence, local relief, drainage concentration, terrain dissection, and slope morphology.

Local data found:

- curvature rasters
- TRI
- TWI
- valley depth
- LS factor

### 5. Seismic trigger / scenario factors

Add:

- PGA or seismic shaking factor
- historical earthquake density as optional background factor

Reason:

The teacher specifically requested strong-earthquake scenario testing. In the CPEC/KKH tectonic setting, a seismic factor is scientifically necessary. PGA should be used for scenario modelling if available; earthquake density can be used only as a static background proxy.

### 6. Cryosphere / high-mountain factors

Add for KKH/high-mountain sub-analysis:

- glacier proximity
- snow/ice frequency
- frozen ground/permafrost class

Reason:

KKH landslides, rockfalls, debris flows, and slope failures can be influenced by glacier retreat, snow/ice cover, freeze-thaw, permafrost/frozen ground, and high mountain thermal processes. These are most important for the KKH sub-analysis, not necessarily for all Pakistan-wide modelling.

### 7. Advanced SAR/InSAR factor, optional

Add only if feasible:

- Sentinel-1 coherence
- deformation velocity
- time-series instability proxy

Reason:

Recent high-impact remote-sensing work supports SAR coherence/deformation as an active state indicator, but this is computationally heavier and should be treated as an advanced optional branch rather than a required baseline factor.

## v3 factor decision

The corrected v3 model should have three tiers.

### Tier 1: Mandatory core factors

- elevation
- slope
- aspect
- curvature
- TRI / relief
- TWI
- lithology
- distance to faults
- distance to rivers/streams
- distance to roads
- monsoon rainfall
- short antecedent/extreme rainfall
- NDVI/EVI indicators
- LST
- land cover
- AlphaEarth embeddings from 2017 onward

### Tier 2: Add if source quality and coverage are acceptable

- soil type
- soil moisture
- fault density
- drainage density
- road density
- rainfall anomaly
- PGA/seismic shaking

### Tier 3: KKH / scenario / advanced branch

- glacier proximity
- snow/ice frequency
- frozen ground/permafrost class
- InSAR/Sentinel-1 coherence or deformation
- proximity to critical infrastructure for exposure/scenario analysis

## Required workflow correction

The corrected workflow should be:

1. Build a new v3 CPEC-wide factor stack.
2. Add road, stream/river, fault, lithology, terrain morphology, and seismic/hydrological factors.
3. Create a new sample table:
   - `cpec_2018_lsm_samples_v3`
4. Re-run multicollinearity with all conventional v3 factors.
5. Re-run:
   - Logistic Regression
   - Random Forest
   - Extra Trees
   - XGBoost
   - LightGBM
   - CatBoost
   - Spatial-CV stacked ensemble
6. Re-run SHAP/TreeSHAP.
7. Export corrected probability rasters.

## Important correction to current export task

The running 73-band stacked-raster predictor export was based on the incomplete factor stack, so it was cancelled after the omission was identified.

Cancelled task:

- `BWCG7QUFRKMJQSN3FJLQ5HM7`

## Saved v3 factor list

- `D:\DING PROJECT\02_methods\factor_set_v3_literature_based_revision.csv`

## Literature basis

Recent landslide-susceptibility literature supports:

- terrain, geology, hydrology, land cover, rainfall, and human disturbance factors as core conditioning variables
- dynamic rainfall and antecedent rainfall windows for temporal susceptibility
- seismic forcing/PGA for earthquake-triggered scenarios
- spatial cross-validation to reduce over-optimistic accuracy
- SHAP/XAI for explaining tree ensemble outputs
- foundation/geospatial embeddings as a new remote-sensing representation, especially from 2017 onward with AlphaEarth

Key examples to cite include recent work in *Remote Sensing*, *Geoscience Frontiers*, *Acta Geotechnica*, *Scientific Reports*, *IEEE JSTARS*, and the AlphaEarth Foundations paper.

