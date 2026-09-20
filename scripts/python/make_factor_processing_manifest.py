"""Create a managed processing manifest for the clean CPEC LSM project.

The manifest is generated from the curated factor registry and records where
each factor should be processed, where managed outputs should live, and whether
the factor is used for susceptibility modelling, exposure/scenario analysis, or
quality-control/ablation only.
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(r"D:\DING PROJECT")
REGISTRY = PROJECT_ROOT / "02_methods" / "static_factor_registry.csv"
OUT_CSV = PROJECT_ROOT / "02_methods" / "factor_processing_manifest.csv"
OUT_MD = PROJECT_ROOT / "02_methods" / "factor_processing_manifest.md"


def processing_backend(row: dict[str, str]) -> str:
    source_type = row["source_type"]
    role = row["role"]
    if source_type == "GEE_public":
        return "GEE public-data stack"
    if source_type == "GEE_asset":
        return "Existing GEE asset; inspect only, do not inherit old modelling"
    if source_type == "local_raster":
        return "Local raster QA, then optional clipped GEE asset upload"
    if source_type == "local_vector":
        if "exposure" in role or "scenario" in role:
            return "Local vector QA, then exposure/scenario overlay"
        return "Local vector QA, rasterize to study grid, optional GEE upload"
    return "Manual review"


def clean_output_target(row: dict[str, str]) -> str:
    factor_id = row["factor_id"]
    role = row["role"]
    if row["source_type"] == "GEE_public":
        return f"projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/{factor_id}"
    if "exposure" in role or "scenario" in role:
        return str(PROJECT_ROOT / "01_clean_data" / "exposure_scenario" / f"{factor_id}.tif")
    return str(PROJECT_ROOT / "01_clean_data" / "predictors_static" / f"{factor_id}.tif")


def stage(row: dict[str, str]) -> str:
    role = row["role"]
    if "exposure" in role or "scenario" in role:
        return "scenario/exposure"
    if "susceptibility" in role:
        return "model predictor"
    return "supporting"


def priority(row: dict[str, str]) -> str:
    factor_id = row["factor_id"]
    core = {
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "curvature_profile",
        "curvature_plan",
        "tri",
        "twi",
        "lithology_class",
        "soil_type",
        "dist_fault_m",
        "fault_density",
        "dist_river_m",
        "drainage_density",
    }
    if factor_id in core:
        return "core"
    if "exposure" in row["role"] or "scenario" in row["role"]:
        return "scenario"
    return "candidate"


def main() -> None:
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "factor_id": row["factor_id"],
                "factor_name": row["factor_name"],
                "stage": stage(row),
                "priority": priority(row),
                "source_type": row["source_type"],
                "processing_backend": processing_backend(row),
                "clean_output_target": clean_output_target(row),
                "quality_checks": "clip_to_boundary; projection_recorded; nodata_recorded; leakage_checked; collinearity_checked",
                "model_use": row["model_use"],
                "justification": row["reviewer_justification"],
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    counts: dict[tuple[str, str], int] = {}
    for row in out_rows:
        key = (row["stage"], row["priority"])
        counts[key] = counts.get(key, 0) + 1

    lines = [
        "# CPEC LSM Factor Processing Manifest",
        "",
        "This manifest is generated from `static_factor_registry.csv` and keeps all clean outputs under `D:\\DING PROJECT` or a clearly named new Earth Engine asset namespace.",
        "",
        "## Summary",
        "",
    ]
    for (stage_name, priority_name), count in sorted(counts.items()):
        lines.append(f"- {stage_name} / {priority_name}: {count} factors")
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- The official boundary is `projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area`.",
            "- Old GEE assets may be inspected for source comparison only; they are not accepted as model outputs or methodological foundation.",
            "- Infrastructure, population, night lights, dams, ports/airports, and pipelines are exposure/scenario layers unless an ablation test justifies predictor use.",
            "- Road-distance and road-density predictors require a no-road ablation because landslide inventories are biased toward mapped corridors.",
            "- All exported rasters must record year/period, scale, CRS, nodata handling, and source citation before modelling.",
            "",
            f"CSV manifest: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
