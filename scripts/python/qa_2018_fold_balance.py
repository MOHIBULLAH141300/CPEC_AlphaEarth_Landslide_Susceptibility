"""Check class balance within each spatial fold for the 2018 sample asset."""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
OUT = Path(r"D:\DING PROJECT\03_models\sampling_2018_fold_balance_v2.json")


def main() -> None:
    ee.Initialize(project=PROJECT)
    fc = ee.FeatureCollection(SAMPLES)
    folds = [1, 2, 3, 4, 5]
    rows = []
    for fold in folds:
        subset = fc.filter(ee.Filter.eq("spatial_fold_5", fold))
        hist = subset.aggregate_histogram("label").getInfo()
        rows.append(
            {
                "fold": fold,
                "total": subset.size().getInfo(),
                "negative": int(hist.get("0", 0)),
                "positive": int(hist.get("1", 0)),
            }
        )
    qa = {
        "asset": SAMPLES,
        "folds": rows,
        "passes_basic_fold_balance": all(row["negative"] > 0 and row["positive"] > 0 for row in rows),
    }
    OUT.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
