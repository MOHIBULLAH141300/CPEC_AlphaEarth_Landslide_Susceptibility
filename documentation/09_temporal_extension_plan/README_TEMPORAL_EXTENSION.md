# Temporal Extension Plan

This folder contains the next-step plan for extending the clean 2018 V3 baseline into annual dynamic-factor susceptibility maps for 2017-2024.

Read these two files first:

- `temporal_inventory_audit_2017_2024_2026-05-04.md`
- `annual_dynamic_susceptibility_extension_plan_2017_2024_2026-05-04.md`
- `annual_dynamic_pilot_export_status_2017_2019_2026-05-04.md`

Main decision:

- Current 2018 V3 results can be called a **2018 dynamic-factor landslide susceptibility baseline**.
- Multi-year 2017-2024 maps can be called **annual dynamic-factor susceptibility maps**.
- Do not claim full temporal validation until more positive events have reliable event dates.

Annual workspace:

- `D:\DING PROJECT\04_maps\annual_dynamic_2017_2024`

Rules:

- Process one year at a time.
- Keep raw and processed materials inside each year folder.
- Store multi-year trend, mean susceptibility, persistent high-susceptibility zones, and annual road-exposure summaries inside `00_multi_year_summary_outputs`.
- Keep all temporal modelling outputs at 250 m.
- Do not create 30 m display-resampled maps for this temporal workflow.

Current pilot status:

- 2017 and 2018 annual dynamic predictor exports have been started in Earth Engine.
- 2019 annual dynamic predictor exports are queued in Earth Engine.
- No annual GeoTIFF exports had synced into local Google Drive at the latest check.
- The next action is to wait for Earth Engine completion, copy synced Drive files into the managed year folders, and then apply the fixed 2018 stacked models year by year.
