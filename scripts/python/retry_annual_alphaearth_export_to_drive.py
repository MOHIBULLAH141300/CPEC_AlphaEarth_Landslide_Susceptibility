r"""Retry one annual AlphaEarth/Satellite Embedding export to Google Drive.

This is used when a previous annual AlphaEarth export failed after writing
partial tiles. The retry writes to a fresh Drive folder and updates the current
annual export plan so downstream copy/prediction scripts use the retry output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
SCRIPTS = Path(r"D:\DING PROJECT\06_scripts\python")
WORKSPACE = Path(r"D:\DING PROJECT\04_maps\annual_dynamic_2017_2024")
PACKAGE_TEMPORAL = Path(
    r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS\09_temporal_extension_plan"
)
EXPORT_SCRIPT = SCRIPTS / "export_annual_dynamic_predictor_stacks_250m_to_drive.py"
PLAN = WORKSPACE / "gee_annual_export_plan_latest.json"


def load_export_module():
    spec = importlib.util.spec_from_file_location("annual_export", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load export module: {EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_plan(year: int, retry_record: dict[str, object]) -> None:
    if not PLAN.exists():
        raise FileNotFoundError(PLAN)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = WORKSPACE / f"gee_annual_export_plan_before_alphaearth_retry_{year}_{stamp}.json"
    backup.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    replaced = False
    for row in plan:
        if int(row.get("year")) == year and row.get("product") == "alphaearth_embeddings":
            row["previous_task_id"] = row.get("task_id")
            row["previous_status"] = row.get("status")
            row["retry_timestamp"] = stamp
            row.update(retry_record)
            replaced = True
            break
    if not replaced:
        plan.append(retry_record)

    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    PACKAGE_TEMPORAL.mkdir(parents=True, exist_ok=True)
    (PACKAGE_TEMPORAL / PLAN.name).write_bytes(PLAN.read_bytes())


def write_retry_log(year: int, retry_record: dict[str, object]) -> None:
    log_dir = WORKSPACE / str(year) / "06_processing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = log_dir / f"alphaearth_retry_export_{year}_{stamp}.json"
    out_md = log_dir / f"alphaearth_retry_export_{year}_{stamp}.md"
    out_json.write_text(json.dumps(retry_record, indent=2), encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# AlphaEarth Export Retry",
                "",
                f"Year: {year}",
                f"Drive folder: `{retry_record['drive_folder']}`",
                f"File prefix: `{retry_record['file_prefix']}`",
                f"Task ID: `{retry_record['task_id']}`",
                f"Status: `{retry_record['status']}`",
                "",
                "Reason: previous export failed after partial tile creation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--drive-folder", default=None)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    module = load_export_module()
    drive_folder = args.drive_folder or f"GEE_CPEC_ANNUAL_DYNAMIC_{args.year}_ALPHA_RETRY_250M"
    record = module.export_image(
        module.annual_alphaearth_stack(args.year),
        args.year,
        "alphaearth_embeddings",
        drive_folder,
        True,
    )
    record["retry_of_product"] = "alphaearth_embeddings"
    record["retry_reason"] = "previous_export_failed_after_partial_tile_creation"
    record["local_year_folder"] = str(WORKSPACE / str(args.year))
    record["expected_local_raw_export_folder"] = str(
        WORKSPACE / str(args.year) / "00_raw_exports_from_gee"
    )
    update_plan(args.year, record)
    write_retry_log(args.year, record)
    print(json.dumps(record, indent=2))
    print(f"Updated aggregate plan: {PLAN}")


if __name__ == "__main__":
    main()
