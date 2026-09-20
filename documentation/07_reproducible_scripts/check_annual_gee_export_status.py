"""Check Earth Engine task status for annual dynamic 250 m exports."""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
WORKSPACE = Path(r"D:\DING PROJECT\04_maps\annual_dynamic_2017_2024")
PLAN = WORKSPACE / "gee_annual_export_plan_latest.json"


def main() -> None:
    ee.Initialize(project=PROJECT)
    if not PLAN.exists():
        raise FileNotFoundError(PLAN)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    wanted = {row["task_id"]: row for row in plan if row.get("task_id")}
    status_by_id = {task.id: task.status() for task in ee.batch.Task.list()}
    rows = []
    for task_id, row in wanted.items():
        status = status_by_id.get(task_id, {"state": "NOT_FOUND"})
        out = {
            "year": row["year"],
            "product": row["product"],
            "task_id": task_id,
            "drive_folder": row["drive_folder"],
            "file_prefix": row["file_prefix"],
            "state": status.get("state"),
            "description": status.get("description", ""),
            "error_message": status.get("error_message", ""),
            "creation_timestamp_ms": status.get("creation_timestamp_ms", ""),
            "update_timestamp_ms": status.get("update_timestamp_ms", ""),
        }
        rows.append(out)
    out_json = WORKSPACE / "gee_annual_export_status_latest.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Saved status to: {out_json}")


if __name__ == "__main__":
    main()
