# Temporal Inventory Audit For Annual CPEC Susceptibility

Date: 2026-05-04

## Main Finding

The current V3 2018 sample table is excellent for a 2018 baseline, but it is **not yet sufficient for true year-by-year temporal validation**.

Reason:

- Total V3 samples: `3316`
- Positive landslide/rockfall samples: `1508`
- Generated non-landslide samples: `1808`
- Positive samples with explicit event year: `1` / `1508` (`0.07%`)

Therefore, annual maps from 2017-2024 should initially be described as **annual dynamic-factor susceptibility maps** or **annual susceptibility scenarios**, not fully validated temporal landslide prediction maps.

## V3 Positive Inventory By Source

| source                                        | positive_count |
| --------------------------------------------- | -------------- |
| CPEC_1970_2020_landslide_points_geometry_only | 815            |
| CPEC_1970_2020_rockfall_points                | 693            |

## V3 Positive Inventory By Hazard Type

| hazard_type | positive_count |
| ----------- | -------------- |
| landslide   | 815            |
| rockfall    | 693            |

## Dated Events In Current V3 Sample Table

| event_year | positive_count |
| ---------- | -------------- |
| 2018       | 1              |

## Dated HMA Inventory Check

Local NASA HMA point catalog found:

- `C:\Users\Administrator\Desktop\cpec landslides\HMA_LS_Cat_2-20260421_073021\HMA_LS_Cat_point_v02.0.shp`
- Total HMA points: `2801`
- HMA points inside official study area: `185`
- Dated HMA points inside official study area: `185`

HMA year distribution inside the study area:

| event_year | hma_count_inside_study_area |
| ---------- | --------------------------- |
| 2007       | 7                           |
| 2008       | 5                           |
| 2010       | 36                          |
| 2011       | 15                          |
| 2012       | 10                          |
| 2013       | 9                           |
| 2014       | 11                          |
| 2015       | 30                          |
| 2016       | 25                          |
| 2017       | 26                          |
| 2018       | 11                          |

HMA trigger distribution inside the study area:

| trigger           | count |
| ----------------- | ----- |
| downpour          | 68    |
| rain              | 46    |
| unknown           | 17    |
| earthquake        | 11    |
| snowfall_snowmelt | 11    |
| monsoon           | 9     |
| continuous_rain   | 8     |
| mining            | 8     |
| construction      | 5     |
| other             | 2     |

## Local Vector Sources With Possible Temporal Fields

| path                                                                                                              | feature_count | geometry_type | crs        | relevant_to_lsm | has_temporal_fields | temporal_fields                       |
| ----------------------------------------------------------------------------------------------------------------- | ------------- | ------------- | ---------- | --------------- | ------------------- | ------------------------------------- |
| D:\CPEC data\中巴经济走廊1：25万大于5级地震分布（1970-2015年）\中巴经济走廊1：25万大于5级地震分布（1970-2015年）\大于5级地震.shp                           | 266           | Point         | EPSG:32642 | True            | True                | 日期; 时间                                |
| C:\Users\Administrator\Desktop\cpec landslides\HMA_LS_Cat_2-20260421_073021\HMA_LS_Cat_point_v02.0.shp            | 2801          | Point         | EPSG:4326  | True            | True                | ev_date; ev_time; sub_date; edit_date |
| C:\Users\Administrator\Desktop\cpec landslides\HMA_LS_Cat_2-20260421_073021\HMA_LS_Cat_poly_v02.0.shp             | 1             | Polygon       | EPSG:4326  | True            | True                | ev_date; ev_time; sub_date; edit_date |
| D:\CPEC data\中巴经济走廊北部山区1：25万地质灾害点（1990-2019年）\中巴经济走廊北部山区1：25万地质灾害点（1990-2019年）\中巴经济走廊北部山区地质灾害点.shp                | 2044          | Point Z       | EPSG:4326  | True            | False               |                                       |
| D:\CPEC data\中巴经济走廊1：25万地质灾害点（1990-2019年）\中巴经济走廊1：25万地质灾害点（1990-2019年）\中巴经济走廊地质灾害点.shp                            | 2369          | Point Z       | EPSG:4326  | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\buffer study area\studyarea.shp                                    | 1             | Polygon       | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh\kkh.shp                                                        | 1             | LineString    |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\CPEC rockfall\rock_fall_area.shp                          | 962           | Polygon Z     | EPSG:4490  | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\CPEC rockfall\rock_fall_area1.shp                         | 962           | Polygon Z     |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\hotosm_pak_roads_lines_shp\hotosm_pak_roads_lines_shp.shp | 1216700       | LineString    |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\landslides points\landslides_points1.shp                  | 1082          | Point Z       |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\landslides points\landslide_points_CPEC.shp               | 1082          | Point Z       | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\landslides_area\landslides_area1.shp                      | 1082          | Polygon Z     |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\pakistan-roads-shape_2\roads.shp                          | 17092         | LineString    | EPSG:4326  | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\Road\Road.shp                                             | 442           | LineString    |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\study area\studyarea1.shp                                 | 1             | Polygon       |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\studyarea1\studyarea1.shp                                 | 1             | Polygon Z     |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\CPEC rockfall\rockfall points\rock_fall_points.shp        | 962           | Point Z       |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh11\buffer studyarea\studyareakkh.shp                            | 1             | Polygon       | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh11\landslide inventory\landslide_points.shp                     | 195           | Point Z       | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh11\non landslide points\nonlandslidepoints.shp                  | 195           | Point         | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh11\studyarea\kkhstudyarea.shp                                   | 1             | LineString    |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\kkh\kkh\kkh.shp                                                    | 2             | LineString    |            | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\factors\proximity to roads\shp\proximity_roads.shp                 | 1             | LineString    | EPSG:32642 | True            | False               |                                       |
| C:\Users\Administrator\Desktop\cpec landslides\factors\Proximity to FAULT\shp\FAULTKKH.shp                        | 2             | LineString    | EPSG:32642 | True            | False               |                                       |

## Decision

Use the current CPEC 1970-2020 landslide/rockfall inventory as the spatial susceptibility inventory, but do not claim it supports annual validation without date enrichment.

Recommended temporal inventory enrichment:

1. Match the CPEC inventory to dated HMA events where locations overlap.
2. Keep unmatched CPEC inventory points as geometry-only historical susceptibility evidence.
3. If possible, manually add event year/date from original source descriptions, reports, Google Earth interpretation, or published records.
4. Build separate temporal validation only for dated events; keep undated points for spatial baseline training.
