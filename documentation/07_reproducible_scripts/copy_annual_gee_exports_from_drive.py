r"""Copy completed annual GEE Drive exports into year-specific folders.

This keeps the Google Drive synced folder as a temporary staging area and moves
managed copies into:

D:\DING PROJECT\04_maps\annual_dynamic_2017_2024\<year>\00_raw_exports_from_gee

By default it copies files and leaves Google Drive untouched. Use --remove-drive-copy
only after verifying the local managed copy is complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


WORKSPACE = Path(r"D:\DING PROJECT\04_maps\annual_dynamic_2017_2024")
PLAN = WORKSPACE / "gee_annual_export_plan_latest.json"
DEFAULT_DRIVE_ROOTS = [
    Path(r"G:\My Drive"),
    Path(r"G:\\"),
    Path(r"C:\Users\Administrator\My Drive"),
    Path(r"C:\Users\Administrator\Google Drive"),
]


def find_drive_root(user_root: str | None) -> Path:
    if user_root:
        root = Path(user_root)
        if root.exists():
            return root
        raise FileNotFoundError(root)
    for root in DEFAULT_DRIVE_ROOTS:
        if root.exists():
            return root
    raise FileNotFoundError("Could not find Google Drive root. Pass --drive-root.")


def copy_product(drive_root: Path, row: dict, remove_drive_copy: bool) -> dict:
    year = int(row["year"])
    folder = drive_root / row["drive_folder"]
    pattern = row["file_prefix"] + "*.tif*"
    matches = sorted(folder.glob(pattern)) if folder.exists() else []
    dest_dir = WORKSPACE / str(year) / "00_raw_exports_from_gee"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in matches:
        dst = dest_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)
        copied.append(str(dst))
        if remove_drive_copy:
            src.unlink()
    return {
        "year": year,
        "product": row["product"],
        "drive_folder": str(folder),
        "matched_files": len(matches),
        "copied_files": copied,
        "remove_drive_copy": remove_drive_copy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--remove-drive-copy", action="store_true")
    args = parser.parse_args()

    drive_root = find_drive_root(args.drive_root)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    results = [copy_product(drive_root, row, args.remove_drive_copy) for row in plan]
    out = WORKSPACE / "copy_annual_gee_exports_from_drive_latest.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved copy log to: {out}")


if __name__ == "__main__":
    main()
