"""Audit temporal readiness and create annual factor manifest for CPEC LSM.

This script prepares the next-step documentation after the clean 2018 V3
baseline. It does not run new models. It answers:

1. Does the current inventory support true year-by-year validation?
2. Which factors should change annually from 2017-2024?
3. Which factors remain static or semi-dynamic?
4. What should be generated next before annual modelling?
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj.datadir

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")

PROJECT_ROOT = Path(r"D:\DING PROJECT")
PACKAGE_ROOT = PROJECT_ROOT / "00_READ_ME_FIRST_2018_V3_RESULTS"
PACKAGE_TEMPORAL = PACKAGE_ROOT / "09_temporal_extension_plan"
SAMPLE_V3 = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
VECTOR_METADATA = PROJECT_ROOT / "00_inventory" / "vector_metadata.csv"
REFERENCES = PROJECT_ROOT / "07_references" / "references_by_step.csv"
BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)

OUT_METHOD_DIR = PROJECT_ROOT / "02_methods" / "temporal_extension_2017_2024"
OUT_REPORT_DIR = PROJECT_ROOT / "05_reports"
OUT_MANIFEST = OUT_METHOD_DIR / "annual_dynamic_factor_manifest_2017_2024.csv"
OUT_YEAR_MATRIX = OUT_METHOD_DIR / "annual_factor_year_coverage_matrix_2017_2024.csv"
OUT_INVENTORY_AUDIT_CSV = OUT_METHOD_DIR / "temporal_inventory_source_audit.csv"
OUT_DECISION_JSON = OUT_METHOD_DIR / "temporal_extension_decision_2017_2024.json"
OUT_INVENTORY_REPORT = (
    OUT_REPORT_DIR / "temporal_inventory_audit_2017_2024_2026-05-04.md"
)
OUT_EXTENSION_REPORT = (
    OUT_REPORT_DIR / "annual_dynamic_susceptibility_extension_plan_2017_2024_2026-05-04.md"
)


def ensure_dirs() -> None:
    OUT_METHOD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_TEMPORAL.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    view = df.fillna("").copy()
    for c in view.select_dtypes(include=[np.floating]).columns:
        view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[i]), max([len(row[i]) for row in rows], default=0))
        for i in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep, *[fmt(row) for row in rows]])


def audit_v3_sample() -> dict:
    df = pd.read_csv(SAMPLE_V3)
    positives = df[df["label"] == 1].copy()
    negatives = df[df["label"] == 0].copy()
    year_counts = (
        positives["event_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("event_year")
        .reset_index(name="positive_count")
    )
    source_counts = (
        positives["source"]
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="positive_count")
    )
    hazard_counts = (
        positives["hazard_type"]
        .value_counts(dropna=False)
        .rename_axis("hazard_type")
        .reset_index(name="positive_count")
    )
    return {
        "rows": int(len(df)),
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "positive_dated_count": int(positives["event_year"].notna().sum()),
        "positive_dated_percent": float(100 * positives["event_year"].notna().mean()),
        "negative_dated_count": int(negatives["event_year"].notna().sum()),
        "year_counts": year_counts,
        "source_counts": source_counts,
        "hazard_counts": hazard_counts,
    }


def audit_vector_temporal_sources() -> pd.DataFrame:
    if not VECTOR_METADATA.exists():
        return pd.DataFrame()
    meta = pd.read_csv(VECTOR_METADATA)
    temporal_keywords = [
        "date",
        "year",
        "time",
        "ev_date",
        "event",
        "日期",
        "时间",
        "年份",
        "anlys_time",
        "src_date",
        "sub_date",
        "edit_date",
    ]
    relevant_keywords = [
        "landslide",
        "rock",
        "fall",
        "hma",
        "geohazard",
        "地质灾害",
        "滑坡",
        "崩塌",
        "earthquake",
        "地震",
    ]
    rows = []
    for _, row in meta.iterrows():
        path = str(row.get("path", row.get("full_path", "")))
        columns = str(row.get("columns", ""))
        lower_path = path.lower()
        lower_cols = columns.lower()
        temporal_fields = [
            col
            for col in columns.split("|")
            if any(k.lower() in col.lower() for k in temporal_keywords)
        ]
        relevant = any(k.lower() in lower_path for k in relevant_keywords)
        has_temporal = bool(temporal_fields)
        if relevant or has_temporal:
            rows.append(
                {
                    "path": path,
                    "feature_count": row.get("feature_count", row.get("features", "")),
                    "geometry_type": row.get("geometry_type", ""),
                    "crs": row.get("crs", ""),
                    "relevant_to_lsm": relevant,
                    "has_temporal_fields": has_temporal,
                    "temporal_fields": "; ".join(temporal_fields),
                    "all_columns": columns,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["relevant_to_lsm", "has_temporal_fields"], ascending=False)
    return out


def audit_hma_inside_boundary() -> dict:
    candidates = sorted(Path(r"C:\Users\Administrator\Desktop\cpec landslides").rglob("HMA_LS_Cat_point_v02.0.shp"))
    if not candidates or not BOUNDARY.exists():
        return {"available": False, "reason": "HMA point shapefile or boundary not found"}
    hma_path = candidates[0]
    hma = gpd.read_file(hma_path)
    hma = hma[hma.geometry.notna() & ~hma.geometry.is_empty].copy()
    if hma.crs is None:
        hma = hma.set_crs(4326, allow_override=True)
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None:
        boundary = boundary.set_crs(32642, allow_override=True)
    boundary = boundary.to_crs(hma.crs)
    inside = gpd.sjoin(hma, boundary[["geometry"]], how="inner", predicate="within")
    if "ev_date" in inside.columns:
        dates = pd.to_datetime(inside["ev_date"], errors="coerce")
        year_counts = (
            dates.dt.year.dropna().astype(int).value_counts().sort_index().rename_axis("event_year").reset_index(name="hma_count_inside_study_area")
        )
    else:
        year_counts = pd.DataFrame(columns=["event_year", "hma_count_inside_study_area"])
    trigger_counts = (
        inside.get("ls_trig", pd.Series(dtype=object))
        .value_counts(dropna=False)
        .rename_axis("trigger")
        .reset_index(name="count")
    )
    return {
        "available": True,
        "path": str(hma_path),
        "total_hma_points": int(len(hma)),
        "inside_study_area": int(len(inside)),
        "dated_inside": int(pd.to_datetime(inside.get("ev_date", pd.Series(dtype=object)), errors="coerce").notna().sum()),
        "year_counts": year_counts,
        "trigger_counts": trigger_counts,
    }


def annual_factor_manifest() -> pd.DataFrame:
    rows = [
        # Static/model-compatible V3 factors.
        ["elevation_m", "Elevation", "Static", "Topography", "Reuse fixed CPEC DEM-derived layer", "Existing V3/local stack", "static", "2017-2024", "Keep in annual maps", "Terrain height controls slope process zones; fixed over study period."],
        ["slope_deg", "Slope", "Static", "Topography", "Reuse fixed DEM-derived slope", "Existing V3/local stack", "static", "2017-2024", "Keep in annual maps", "Slope is a primary physical control on failure initiation."],
        ["aspect_deg", "Aspect", "Static", "Topography", "Reuse fixed DEM-derived aspect", "Existing V3/local stack", "static", "2017-2024", "Keep in annual maps", "Aspect controls radiation, snow/vegetation patterns, and weathering."],
        ["profile_curvature", "Profile curvature", "Static", "Topography", "Reuse official/local 30 m curvature", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Controls flow acceleration and slope morphology."],
        ["plan_curvature", "Plan curvature", "Static", "Topography", "Reuse official/local 30 m curvature", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Controls flow convergence/divergence."],
        ["tri", "Terrain ruggedness index", "Static", "Topography", "Reuse official/local TRI", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Represents local terrain roughness and instability-prone relief."],
        ["twi", "Topographic wetness index", "Static", "Hydrology/topography", "Reuse DEM-derived TWI", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Represents topographic moisture accumulation tendency."],
        ["valley_depth", "Valley depth", "Static", "Topography", "Reuse official/local valley depth", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Captures incision and relief conditions around slopes."],
        ["lithology_code", "Lithology", "Static", "Geology", "Reuse official CPEC geology/lithology coding", "Local CPEC vector/raster", "static", "2017-2024", "Keep in annual maps", "Geologic material controls slope strength and weathering."],
        ["soil_type", "Soil type", "Static", "Soil/geology", "Reuse local CPEC soil type", "Local CPEC raster", "static", "2017-2024", "Keep in annual maps", "Soil texture/type affects infiltration and shear strength."],
        ["log1p_dist_fault_m", "Distance to active faults", "Static", "Proximity/tectonics", "Reuse full-CPEC fault distance", "Local official fault vector", "static", "2017-2024", "Keep in annual maps", "Fault proximity captures tectonic fracturing and structural weakness."],
        ["log1p_dist_river_m", "Distance to rivers/streams", "Semi-static", "Hydrology/proximity", "Reuse official hydrography unless annual water network is added", "Local official hydrography vector", "fixed baseline", "2017-2024", "Keep fixed initially", "River incision and toe erosion matter; current source is not annual."],
        ["log1p_dist_road_m", "Distance to roads", "Semi-static", "Infrastructure/proximity", "Reuse 2018 CPEC road network unless annual road expansion data is obtained", "Local official 2018 road vector", "fixed baseline", "2017-2024", "Keep fixed initially; run road-distance ablation later", "Road cuts and construction disturbance matter, but fixed 2018 roads avoid unsupported annual road change assumptions."],
        ["modis_lc_type1", "Land-cover class", "Annual", "Land cover", "Use annual MODIS MCD12Q1 LC_Type1; verify 2024 image in GEE", "MODIS/061/MCD12Q1", "annual", "2017-2024, with 2024 verification", "Keep in annual maps", "Annual land-cover change is a standard dynamic susceptibility factor."],
        ["rain_monsoon_total", "Monsoon rainfall total", "Annual", "Rainfall", "Sum CHIRPS daily rainfall over monsoon months", "UCSB-CHG/CHIRPS/DAILY", "annual", "2017-2024", "Keep in annual maps", "Seasonal rainfall controls saturation and triggering susceptibility."],
        ["rain_max_1day", "Maximum 1-day rainfall", "Annual", "Rainfall extreme", "Maximum daily CHIRPS precipitation within calendar or monsoon year", "UCSB-CHG/CHIRPS/DAILY", "annual", "2017-2024", "Keep in annual maps", "Extreme rainfall is a direct landslide-triggering proxy."],
        ["rain_annual_total", "Annual rainfall total", "Annual", "Rainfall", "Sum CHIRPS daily rainfall over year", "UCSB-CHG/CHIRPS/DAILY", "annual candidate", "2017-2024", "Candidate; VIF/ablation before final annual model", "Useful hydrologic background, but may be collinear with monsoon rainfall."],
        ["rain_max_3day", "Maximum 3-day rainfall", "Annual", "Rainfall extreme", "Rolling 3-day maximum CHIRPS precipitation", "UCSB-CHG/CHIRPS/DAILY", "annual candidate", "2017-2024", "Candidate; VIF/ablation before final annual model", "Antecedent rainfall can better represent slope saturation."],
        ["rain_max_7day", "Maximum 7-day rainfall", "Annual", "Rainfall extreme", "Rolling 7-day maximum CHIRPS precipitation", "UCSB-CHG/CHIRPS/DAILY", "annual candidate", "2017-2024", "Candidate; VIF/ablation before final annual model", "Longer rainfall accumulation supports dynamic susceptibility interpretation."],
        ["ndvi_median", "Median NDVI", "Annual", "Vegetation", "Median annual MODIS MOD13Q1 NDVI with QA screening", "MODIS/061/MOD13Q1", "annual", "2017-2024", "Keep in annual maps", "Vegetation condition changes slope hydrology/root reinforcement proxies."],
        ["ndvi_amplitude", "NDVI amplitude", "Annual", "Vegetation seasonality", "Annual max-min MODIS MOD13Q1 NDVI", "MODIS/061/MOD13Q1", "annual", "2017-2024", "Keep in annual maps", "Seasonality captures vegetation stress and phenological change."],
        ["ndvi_max", "Maximum NDVI", "Annual", "Vegetation", "Annual maximum MODIS MOD13Q1 NDVI", "MODIS/061/MOD13Q1", "annual candidate", "2017-2024", "Candidate; VIF/ablation before final annual model", "May help interpret vegetation peak condition but can duplicate NDVI median/amplitude."],
        ["evi_median", "Median EVI", "Annual", "Vegetation", "Median annual MODIS MOD13Q1 EVI", "MODIS/061/MOD13Q1", "annual candidate", "2017-2024", "Candidate; VIF/ablation before final annual model", "EVI can improve vegetation representation in dense canopy areas."],
        ["eq_density_ms5", "Earthquake density", "Semi-dynamic", "Seismicity", "Current V3 uses fixed 1970-2015 density; annual extension should test cumulative-to-year seismic density if data are extended", "Local 1970-2015 earthquake data; USGS/GEM optional", "fixed in V3; dynamic candidate", "Fixed for 2017-2024 unless updated", "Keep fixed initially; update later if post-2015 seismic catalog is added", "Seismic shaking/fracturing is relevant but the current source is not annual after 2015."],
        # AlphaEarth branch.
        ["A00-A63", "AlphaEarth Embeddings", "Annual", "Foundation-model embeddings", "Use all 64 annual embedding bands for each year", "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", "annual", "2017-2024", "Keep in AlphaEarth and fused branches", "Official annual 10 m, 64-dimensional embeddings represent surface conditions and cross-sensor context."],
        # Optional v4 additions.
        ["dynamic_world_probabilities", "Dynamic World land-cover probabilities", "Annual/near-real-time", "Land-cover probabilities", "Aggregate annual class probabilities such as built, bare, snow/ice, water", "GOOGLE/DYNAMICWORLD/V1", "annual optional", "2017-2024", "Optional V4/ablation, not in current V3 model", "10 m class probabilities can add interpretable land-cover dynamics for recent years."],
        ["lst_day_mean_c", "Mean daytime LST", "Annual", "Thermal", "Annual mean MODIS MOD11A2 day LST after scaling and C conversion", "MODIS/061/MOD11A2", "annual candidate", "2017-2024", "Do not add blindly; V3 dropped LST due high VIF", "Thermal regime may proxy snow/permafrost stress but was collinear in V3."],
        ["lst_day_max_c", "Maximum daytime LST", "Annual", "Thermal extreme", "Annual maximum MODIS MOD11A2 day LST", "MODIS/061/MOD11A2", "annual candidate", "2017-2024", "Do not add blindly; V3 dropped LST due high VIF", "Thermal extremes may help but require collinearity check."],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "field_name",
            "readable_name",
            "temporal_role",
            "factor_group",
            "annual_processing_rule",
            "recommended_source",
            "annual_status",
            "usable_years",
            "decision_for_next_model",
            "justification",
        ],
    )


def year_coverage_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    years = list(range(2017, 2025))
    rows = []
    for _, row in manifest.iterrows():
        out = {
            "field_name": row["field_name"],
            "readable_name": row["readable_name"],
            "temporal_role": row["temporal_role"],
            "source": row["recommended_source"],
        }
        for year in years:
            if row["field_name"] == "modis_lc_type1" and year == 2024:
                out[str(year)] = "verify in GEE"
            elif row["field_name"] in {"A00-A63", "dynamic_world_probabilities"}:
                out[str(year)] = "available"
            elif row["temporal_role"] in {"Annual", "Annual/near-real-time"}:
                out[str(year)] = "available"
            elif row["temporal_role"] == "Semi-dynamic" and row["field_name"] == "eq_density_ms5":
                out[str(year)] = "fixed until seismic catalog updated"
            else:
                out[str(year)] = "fixed"
        rows.append(out)
    return pd.DataFrame(rows)


def write_reports(sample_audit: dict, vector_audit: pd.DataFrame, hma_audit: dict, manifest: pd.DataFrame, matrix: pd.DataFrame) -> None:
    year_counts = sample_audit["year_counts"]
    source_counts = sample_audit["source_counts"]
    hazard_counts = sample_audit["hazard_counts"]
    hma_years = hma_audit.get("year_counts", pd.DataFrame())
    hma_triggers = hma_audit.get("trigger_counts", pd.DataFrame())

    inventory_text = f"""# Temporal Inventory Audit For Annual CPEC Susceptibility

Date: 2026-05-04

## Main Finding

The current V3 2018 sample table is excellent for a 2018 baseline, but it is **not yet sufficient for true year-by-year temporal validation**.

Reason:

- Total V3 samples: `{sample_audit["rows"]}`
- Positive landslide/rockfall samples: `{sample_audit["positive_count"]}`
- Generated non-landslide samples: `{sample_audit["negative_count"]}`
- Positive samples with explicit event year: `{sample_audit["positive_dated_count"]}` / `{sample_audit["positive_count"]}` (`{sample_audit["positive_dated_percent"]:.2f}%`)

Therefore, annual maps from 2017-2024 should initially be described as **annual dynamic-factor susceptibility maps** or **annual susceptibility scenarios**, not fully validated temporal landslide prediction maps.

## V3 Positive Inventory By Source

{markdown_table(source_counts)}

## V3 Positive Inventory By Hazard Type

{markdown_table(hazard_counts)}

## Dated Events In Current V3 Sample Table

{markdown_table(year_counts if not year_counts.empty else pd.DataFrame([{"event_year": "none", "positive_count": 0}]))}

## Dated HMA Inventory Check

"""
    if hma_audit.get("available"):
        inventory_text += f"""Local NASA HMA point catalog found:

- `{hma_audit["path"]}`
- Total HMA points: `{hma_audit["total_hma_points"]}`
- HMA points inside official study area: `{hma_audit["inside_study_area"]}`
- Dated HMA points inside official study area: `{hma_audit["dated_inside"]}`

HMA year distribution inside the study area:

{markdown_table(hma_years if not hma_years.empty else pd.DataFrame([{"event_year": "none", "hma_count_inside_study_area": 0}]))}

HMA trigger distribution inside the study area:

{markdown_table(hma_triggers if not hma_triggers.empty else pd.DataFrame([{"trigger": "none", "count": 0}]), max_rows=12)}
"""
    else:
        inventory_text += f"HMA audit was not completed: `{hma_audit.get('reason', 'unknown')}`\n"

    inventory_text += f"""
## Local Vector Sources With Possible Temporal Fields

{markdown_table(vector_audit[["path", "feature_count", "geometry_type", "crs", "relevant_to_lsm", "has_temporal_fields", "temporal_fields"]], max_rows=25) if not vector_audit.empty else "No vector metadata table found."}

## Decision

Use the current CPEC 1970-2020 landslide/rockfall inventory as the spatial susceptibility inventory, but do not claim it supports annual validation without date enrichment.

Recommended temporal inventory enrichment:

1. Match the CPEC inventory to dated HMA events where locations overlap.
2. Keep unmatched CPEC inventory points as geometry-only historical susceptibility evidence.
3. If possible, manually add event year/date from original source descriptions, reports, Google Earth interpretation, or published records.
4. Build separate temporal validation only for dated events; keep undated points for spatial baseline training.
"""
    OUT_INVENTORY_REPORT.write_text(inventory_text, encoding="utf-8")

    dynamic_count = int((manifest["temporal_role"].str.contains("Annual", regex=False)).sum())
    static_count = int((manifest["temporal_role"] == "Static").sum())
    semi_count = int((manifest["temporal_role"].str.startswith("Semi")).sum())

    extension_text = f"""# Annual Dynamic Susceptibility Extension Plan, 2017-2024

Date: 2026-05-04

## Recommended Label

For the current 2018 V3 result, use:

**2018 dynamic-factor landslide susceptibility baseline**

For the next multi-year outputs, use:

**annual dynamic-factor landslide susceptibility maps, 2017-2024**

Only call the study fully **temporal landslide susceptibility modelling** after dated event validation or space-time validation is added.

## Why Annual Maps Will Differ

Static terrain/geology factors stay fixed, while annual rainfall, vegetation, land cover, and AlphaEarth embeddings change by year.

Manifest summary:

- Static factors: `{static_count}`
- Semi-static or semi-dynamic fixed-baseline factors: `{semi_count}`
- Annual or annual-candidate factors: `{dynamic_count}`

## Model-Compatible Annual V3 Core

The first annual extension should keep the 2018 V3 model structure stable:

- Conventional branch: keep the final 19 V3 factors.
- AlphaEarth branch: use A00-A63 for each year from 2017-2024.
- Fused branch: Conventional + AlphaEarth Embeddings.
- Grid: 250 m modelling grid.
- Boundary: official CPEC study area.

## Factors To Update Every Year

{markdown_table(manifest[manifest["temporal_role"].str.contains("Annual", regex=False)][["field_name", "readable_name", "recommended_source", "usable_years", "decision_for_next_model"]])}

## Static Or Fixed-Baseline Factors

{markdown_table(manifest[~manifest["temporal_role"].str.contains("Annual", regex=False)][["field_name", "readable_name", "temporal_role", "decision_for_next_model"]])}

## Year Coverage Matrix

{markdown_table(matrix, max_rows=40)}

## Immediate Next Steps

1. Create annual predictor-stack export scripts for 2017-2024 using this manifest.
2. First export only 2017, 2018, and 2019 as a pilot.
3. Confirm that 2018 annual outputs reproduce the current V3 2018 factor values closely.
4. Apply the final 2018 stacked models to annual stacks.
5. Produce annual probability maps and change maps:
   - yearly probability
   - year minus 2018 baseline
   - persistent high susceptibility
   - increasing susceptibility
   - KKH/CPEC road exposure change
6. Add dated-event temporal validation only after inventory date enrichment.

Important validation note:

- The local HMA catalog contributes dated events inside the study area through 2018.
- For 2019-2024, true temporal validation will require additional dated inventory sources or manual event-date enrichment.

## Literature/Data Support

- AlphaEarth/Satellite Embedding V1 is annual, 64-band, 10 m, and available from 2017-2024 in Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- CHIRPS Daily supports annual and extreme rainfall metrics from 1981 onward: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
- MODIS MOD13Q1 supports NDVI/EVI annual summaries at 250 m: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
- MODIS MOD11A2 supports annual LST candidates at 1 km: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
- MODIS MCD12Q1 supports annual land-cover class products: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
- Dynamic World offers optional 10 m land-cover probability features for Sentinel-2 era years: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- NASA HMA landslide catalog provides dated landslide records useful for temporal enrichment: https://nsidc.org/data/hma_ls_cat/versions/2

## Outputs Created

- `{OUT_MANIFEST}`
- `{OUT_YEAR_MATRIX}`
- `{OUT_INVENTORY_AUDIT_CSV}`
- `{OUT_DECISION_JSON}`
- `{OUT_INVENTORY_REPORT}`
- `{OUT_EXTENSION_REPORT}`
"""
    OUT_EXTENSION_REPORT.write_text(extension_text, encoding="utf-8")


def copy_to_package() -> None:
    for path in [
        OUT_MANIFEST,
        OUT_YEAR_MATRIX,
        OUT_INVENTORY_AUDIT_CSV,
        OUT_DECISION_JSON,
        OUT_INVENTORY_REPORT,
        OUT_EXTENSION_REPORT,
    ]:
        target = PACKAGE_TEMPORAL / path.name
        target.write_bytes(path.read_bytes())

    package_readme = PACKAGE_TEMPORAL / "README_TEMPORAL_EXTENSION.md"
    package_readme.write_text(
        """# Temporal Extension Plan

This folder contains the next-step plan for extending the clean 2018 V3 baseline into annual dynamic-factor susceptibility maps for 2017-2024.

Read these two files first:

- `temporal_inventory_audit_2017_2024_2026-05-04.md`
- `annual_dynamic_susceptibility_extension_plan_2017_2024_2026-05-04.md`

Main decision:

- Current 2018 V3 results can be called a **2018 dynamic-factor landslide susceptibility baseline**.
- Multi-year 2017-2024 maps can be called **annual dynamic-factor susceptibility maps**.
- Do not claim full temporal validation until more positive events have reliable event dates.
""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    sample_audit = audit_v3_sample()
    vector_audit = audit_vector_temporal_sources()
    hma_audit = audit_hma_inside_boundary()
    manifest = annual_factor_manifest()
    matrix = year_coverage_matrix(manifest)

    vector_audit.to_csv(OUT_INVENTORY_AUDIT_CSV, index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    matrix.to_csv(OUT_YEAR_MATRIX, index=False, encoding="utf-8-sig")

    decision = {
        "current_2018_label": "2018 dynamic-factor landslide susceptibility baseline",
        "next_multi_year_label": "annual dynamic-factor landslide susceptibility maps, 2017-2024",
        "avoid_claim_until_enriched": "fully validated temporal landslide susceptibility modelling",
        "positive_samples": sample_audit["positive_count"],
        "positive_samples_with_event_year": sample_audit["positive_dated_count"],
        "positive_dated_percent": sample_audit["positive_dated_percent"],
        "hma_inside_study_area": hma_audit.get("inside_study_area"),
        "recommendation": [
            "Create annual predictor stacks first.",
            "Use 2017-2019 pilot before full 2017-2024 export.",
            "Add HMA/date enrichment before claiming temporal validation.",
        ],
    }
    OUT_DECISION_JSON.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    write_reports(sample_audit, vector_audit, hma_audit, manifest, matrix)
    copy_to_package()

    print(f"Wrote: {OUT_MANIFEST}")
    print(f"Wrote: {OUT_INVENTORY_REPORT}")
    print(f"Wrote: {OUT_EXTENSION_REPORT}")
    print(f"Copied temporal plan to: {PACKAGE_TEMPORAL}")


if __name__ == "__main__":
    main()
