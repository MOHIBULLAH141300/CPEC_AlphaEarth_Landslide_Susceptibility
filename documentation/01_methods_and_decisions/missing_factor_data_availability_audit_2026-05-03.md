# Missing Factor Data Availability Audit

Date: 2026-05-03

## Short answer

Most of the missing factors are already available on the computer, but they are not yet included in the active 2018 GEE factor stack or the active sample table.

The existing uploaded GEE proximity assets are not enough for the final CPEC-wide model because they cover only a small KKH-area subset around approximately 74.18-75.49E and 35.39-36.96N.

Therefore, the correct next step is not simply to reuse the current uploaded proximity rasters. We need to rebuild a corrected CPEC-wide v3 factor stack and sample table.

## Current active table problem

Current sample table:

- `D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v2.csv`

Missing from this table:

- road proximity
- stream/river proximity
- fault proximity
- lithology
- curvature/TWI/TRI/relief
- PGA/seismic factor
- soil/soil moisture
- cryosphere factors

Current GEE stack:

- `projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m`

Missing from this stack:

- road proximity
- stream/river proximity
- fault proximity
- lithology
- curvature/TWI/TRI/relief
- PGA/seismic factor
- soil/soil moisture
- cryosphere factors

## Data availability matrix

| Factor | Availability | Where found | Action needed |
|---|---|---|---|
| Distance to roads | Available as full CPEC vector; old small KKH raster also exists | `D:\CPEC data\中巴经济走廊1：25万路网数据（2018年）\...\路网.shp`; old raster in desktop factors | Rebuild CPEC-wide distance raster/vector distance at 250 m |
| Road density | Derivable | same road vector | Derive if not too collinear with distance to roads |
| Distance to streams/rivers | Available as full CPEC vector and river-distance raster | `D:\CPEC data\中巴经济走廊1：25万水文数据（2000年）\...\河网.shp`; `river_buffer.tif` | Prefer rebuild/harmonize to official study grid |
| Drainage density | Derivable | same river network | Derive and test by multicollinearity/permutation/SHAP |
| Distance to active faults | Available as full CPEC vector; old small KKH raster also exists | `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\...\活动断裂带.shp`; `D:\CPEC data\中巴经济走廊1：25万活动断裂带（1964年）\FAULTS.shp` | Rebuild CPEC-wide distance raster/vector distance at 250 m |
| Fault density | Derivable | full fault vectors | Optional; derive and test with fault distance |
| Lithology/geology | Available locally; small KKH GEE asset exists | `D:\CPEC data\Geological data of CPEC (1)` and local desktop lithology | Need choose official full-coverage source and rasterize/harmonize |
| Profile curvature | Available | `D:\CPEC data\中巴经济走廊30m剖面曲率（2009年）\...\剖面曲率.tif` | Resample/aggregate to 250 m |
| Plan curvature | Available | `D:\CPEC data\中巴经济走廊30m平面曲率（2009年）\...\平面曲率.tif` | Resample/aggregate to 250 m |
| TRI | Available | `D:\CPEC data\中巴经济走廊30m地形粗糙度（2009年）\...\TRI.tif` | Resample/aggregate to 250 m |
| TWI | Available | `D:\CPEC data\中巴经济走廊30m地形湿度指数（2009年）\...\Topographic Wetness Index1.tif` | Resample/aggregate to 250 m |
| Valley depth / relief | Available | `D:\CPEC data\中巴经济走廊30m谷深（2009年）\...\Valley Depth.tif`; relief folder exists | Resample/aggregate to 250 m |
| LS factor | Available | `D:\CPEC data\中巴经济走廊30mLS因子（2009）\...\LS-Factor1.tif` | Optional; test collinearity with slope/relief |
| Soil type | Available but coarse/older | `D:\CPEC data\中巴经济走廊30m土壤类型数据（1971-1981年）\...\土壤类型.tif` | Add if coverage/classes are usable |
| Historical earthquake density | Available | `D:\CPEC data\中巴经济走廊30m大于5级地震密度（1970-2015年）\...\大于5Ms地震密度.tif` | Optional static seismic background |
| PGA / seismic shaking | Not directly found yet | Need source | Need obtain from USGS/GEM/GSHAP or compute scenario PGA |
| Soil moisture | Not directly found locally | GEE/online source possible | Need derive from ERA5-Land, GLDAS, SMAP, or similar |
| Rainfall anomaly | Derivable in GEE | CHIRPS daily/monthly | Add for annual dynamic modelling |
| Glacier proximity | Available | `D:\CPEC data\中巴经济走廊1：25万冰川数据（2016年）\...\冰川.shp` | Add for KKH/high-mountain branch |
| Frozen ground/permafrost | Available | `D:\CPEC data\中巴经济走廊1：25万冻土数据（2016年）\...\冻土数据.shp` | Add for KKH/high-mountain branch |
| Snow/ice frequency | Derivable | MODIS/Sentinel/GEE | Add if feasible for annual KKH branch |
| InSAR/Sentinel-1 coherence/deformation | Not prepared | GEE/Sentinel-1 possible | Optional advanced branch; not required before v3 baseline |

## GEE asset check

Top-level GEE assets include:

- `proximity_roads11`
- `proximity_waterways2`
- `proximity_faults1`
- `Lithology`
- `Curvature`
- `TRI`
- `TWI`

But these have approximately the following spatial footprint:

- longitude: 74.18 to 75.49E
- latitude: 35.39 to 36.96N

This is a KKH/sub-area footprint, not the full official CPEC study area. They are useful for reference or KKH sub-analysis but should not be used for the full CPEC-wide v3 model.

## Literature-based standard from now onward

Before proceeding with each major/minor step, the workflow should pass a literature and methods gate:

1. Does the factor/model/validation step appear in recent high-impact landslide susceptibility or dynamic susceptibility literature?
2. Is it appropriate for CPEC/KKH physical geography?
3. Is there data coverage for the full study area?
4. Is the resolution/period consistent with the study objective?
5. Can it be justified to reviewers?
6. Will multicollinearity/SHAP/permutation testing decide whether to keep or drop it?

## Next implementation recommendation

Build `cpec_2018_lsm_samples_v3` rather than continuing from v2.

Recommended v3 steps:

1. Build CPEC-wide road, river, and fault distance factors from full CPEC vector data.
2. Add lithology/geology.
3. Add terrain morphology factors: profile curvature, plan curvature, TRI, TWI, valley depth/relief.
4. Add optional seismic/soil/cryosphere factors where coverage is adequate.
5. Sample all predictors at landslide and non-landslide points.
6. Re-run multicollinearity before modelling.
7. Re-run all base models and spatial-CV stacked ensemble.

