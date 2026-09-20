"""QA the clean 2018 Earth Engine sampling table."""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
OUT = Path(r"D:\DING PROJECT\03_models\sampling_2018_asset_v2_qa.json")


def main() -> None:
    ee.Initialize(project=PROJECT)
    fc = ee.FeatureCollection(SAMPLES)
    first = ee.Feature(fc.first())
    names = first.propertyNames().getInfo()
    qa = {
        "asset": SAMPLES,
        "feature_count": fc.size().getInfo(),
        "label_histogram": fc.aggregate_histogram("label").getInfo(),
        "use_role_histogram": fc.aggregate_histogram("use_role").getInfo(),
        "fold_histogram": fc.aggregate_histogram("spatial_fold_5").getInfo(),
        "hazard_type_histogram": fc.aggregate_histogram("hazard_type").getInfo(),
        "property_count": len(names),
        "properties": names,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
