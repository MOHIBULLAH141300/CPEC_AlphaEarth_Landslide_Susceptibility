# Proximity Factor Omission Correction

Date: 2026-05-03

## Issue

The user correctly identified that three important conditioning factors were not included in the active 2018 modelling table:

- proximity to roads
- proximity to streams/rivers
- proximity to geological faults

## Current active 2018 model table

Current table:

- `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v2.csv`

Current GEE factor asset:

- `projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m`

Checked result:

- the sample table does not contain road, river/stream, or fault proximity fields
- the GEE factor asset also does not contain those bands

Therefore, the current 2018 model is valid only for the factor set actually used, but it is incomplete relative to the preferred landslide-conditioning factor design.

## Why these factors should be included

### Road proximity

Road cuts, slope undercutting, drainage modification, blasting, traffic vibration, and construction disturbance can increase landslide occurrence near roads. This is especially important for KKH/CPEC because infrastructure corridors are a core research target.

### Stream/river proximity

Rivers and streams affect landslide susceptibility through toe erosion, valley incision, saturation, and drainage concentration. This is a classic hydrological/geomorphic landslide predictor.

### Fault proximity

Active faults indicate fractured rock, tectonic weakness, seismic disturbance, and structurally controlled slope instability. This is important in the Himalaya-Karakoram-Hindu Kush tectonic setting.

## Data found locally

Official/local CPEC vector sources found:

- road network: `D:\CPEC data\中巴经济走廊1：25万路网数据（2018年）\中巴经济走廊1：25万路网数据（2018年）\路网.shp`
- hydrology/river network: `D:\CPEC data\中巴经济走廊1：25万水文数据（2000年）\中巴经济走廊1：25万水文数据（2000年）\河网.shp`
- active faults: `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\中巴经济走廊1：25万活动断裂带（1964年）\活动断裂带.shp`
- additional faults layer: `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\FAULTS.shp`

Older local KKH/sub-area distance rasters also exist:

- `C:\Users\Administrator\Desktop\cpec landslides\factors\proximity to roads\proximity_roads1.tif`
- `C:\Users\Administrator\Desktop\cpec landslides\factors\proximity to waterways\proximity_waterways2.tif`
- `C:\Users\Administrator\Desktop\cpec landslides\factors\Proximity to FAULT\proximity_faults1.tif`

However, these older local rasters cover only a small KKH-area extent, not the full official CPEC study area. They should not be used directly for the CPEC-wide 2018 model.

## Required correction

The next corrected 2018 dataset should add:

- `dist_road_m`
- `dist_stream_m` or `dist_river_m`
- `dist_fault_m`

Optional additional derivatives:

- `road_density`
- `drainage_density`
- `fault_density`

## Recommended implementation

1. Use the official CPEC vector layers, not the older small local KKH rasters.
2. Rasterize or compute Euclidean distance to roads, streams/rivers, and faults on the same 250 m CPEC-wide grid.
3. Add these predictors to the 2018 GEE factor asset or compute/sample them locally from the vector layers.
4. Create a new sample table:
   - `cpec_2018_lsm_samples_v3`
5. Re-run:
   - multicollinearity assessment
   - all base models
   - CatBoost
   - stacked/meta-learning ensemble
   - SHAP/TreeSHAP
6. Regenerate probability rasters using the corrected final predictor set.

## Current result interpretation

The current 2018 results remain useful as a baseline/foundation experiment, but the final corrected study should include these proximity factors because they are scientifically important and expected by landslide-susceptibility reviewers.

