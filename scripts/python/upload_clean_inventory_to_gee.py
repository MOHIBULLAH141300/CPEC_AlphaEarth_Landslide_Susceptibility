"""Stage the clean model-candidate inventory into Earth Engine.

The upload is implemented as a server-side Export.table.toAsset from a
client-constructed FeatureCollection, avoiding unmanaged raw uploads. Only the
clean candidate fields needed for modelling/validation are sent to GEE.
"""

from __future__ import annotations

from pathlib import Path

import ee
import pandas as pd


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
CSV_PATH = Path(r"D:\DING PROJECT\01_clean_data\inventory\cpec_landslide_inventory_model_candidates.csv")
ASSET_ID = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/inventory/cpec_inventory_model_candidates_v1"


KEEP_COLUMNS = [
    "inventory_id",
    "source",
    "source_fid",
    "hazard_type",
    "event_date",
    "event_year",
    "event_year_quality",
    "trigger",
    "country_region",
    "confidence",
    "use_role",
    "source_role",
    "duplicate_cluster_id",
    "duplicate_priority",
]


def clean_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def main() -> None:
    ee.Initialize(project=PROJECT)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    features = []
    for row in df.to_dict(orient="records"):
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        props = {col: clean_value(row.get(col, "")) for col in KEEP_COLUMNS}
        props["label"] = 1
        features.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))

    fc = ee.FeatureCollection(features).filterBounds(ee.FeatureCollection(BOUNDARY))
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description="cpec_inventory_model_candidates_v1",
        assetId=ASSET_ID,
    )
    task.start()
    print(f"Started upload task: {task.id}")
    print(f"Asset: {ASSET_ID}")
    print(f"Features sent: {len(features)}")


if __name__ == "__main__":
    main()
