"""Create stable lithological classes from the geology ``type`` attribute.

Historical sample and map workflows factorised ``OBJECTID_1`` independently,
which encoded polygon identifiers rather than geological classes and produced
incompatible training/map codes. This script rasterises the 14 geological
``type`` classes with one shared mapping and updates the corrected sample table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import shapefile


PROJECT_ROOT = Path(r"D:\DING PROJECT")
WORK_DIR = PROJECT_ROOT / "01_clean_data" / "manuscript_v3_corrected_terrain"
GEOLOGY = WORK_DIR / "geology_source_ascii" / "lithology.shp"
REFERENCE = WORK_DIR / "cpec_copernicus_glo30_corrected_slope_aspect_250m.tif"
INPUT_TABLE = WORK_DIR / "cpec_baseline_samples_corrected_terrain.csv"
OUTPUT_RASTER = WORK_DIR / "cpec_corrected_lithology_classes_250m.tif"
OUTPUT_TABLE = WORK_DIR / "cpec_baseline_samples_corrected_terrain_lithology.csv"
MAPPING_CSV = WORK_DIR / "lithology_class_mapping.csv"
QA_CSV = WORK_DIR / "corrected_lithology_qa.csv"


def read_geology() -> tuple[list[dict], list[str], str]:
    reader = shapefile.Reader(str(GEOLOGY), encoding="gbk")
    fields = [field[0] for field in reader.fields[1:]]
    if "type" not in fields:
        raise ValueError(f"Geological type field not found. Available fields: {fields}")
    type_index = fields.index("type")
    records = []
    values = []
    for shape_record in reader.iterShapeRecords():
        value = str(shape_record.record[type_index]).strip()
        if not value:
            value = "Unknown"
        records.append({"geometry": shape_record.shape.__geo_interface__, "type": value})
        values.append(value)
    source_wkt = GEOLOGY.with_suffix(".prj").read_text(encoding="utf-8", errors="ignore")
    return records, sorted(set(values)), source_wkt


def create_raster() -> dict[str, int]:
    records, classes, source_wkt = read_geology()
    mapping = {value: idx for idx, value in enumerate(classes)}
    pd.DataFrame(
        [{"lithology_code": code, "geological_type": value} for value, code in mapping.items()]
    ).sort_values("lithology_code").to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")

    with rasterio.open(REFERENCE) as ref:
        shapes = []
        for record in records:
            geometry = transform_geom(source_wkt, ref.crs, record["geometry"], precision=12)
            shapes.append((geometry, mapping[record["type"]]))
        array = rasterize(
            shapes,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=-1,
            all_touched=True,
            dtype="int16",
        )
        profile = ref.profile.copy()
        profile.update(
            count=1,
            dtype="int16",
            nodata=-1,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
            dst.write(array, 1)
            dst.set_band_description(1, "lithology_code")
            dst.update_tags(
                source_field="type",
                class_count=str(len(mapping)),
                coding="alphabetically sorted geological type values; see lithology_class_mapping.csv",
            )
    return mapping


def update_samples(mapping: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(INPUT_TABLE, encoding="utf-8-sig")
    coords = list(zip(df.longitude.astype(float), df.latitude.astype(float)))
    with rasterio.open(OUTPUT_RASTER) as src:
        values = np.asarray([value[0] for value in src.sample(coords)], dtype=float)
    df["lithology_code_objectid_invalid"] = df["lithology_code"]
    df["lithology_code"] = values
    df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")
    qa = pd.DataFrame(
        [
            {
                "field": "historical_OBJECTID_1_code",
                "unique_values": df["lithology_code_objectid_invalid"].nunique(dropna=True),
                "minimum": df["lithology_code_objectid_invalid"].min(),
                "maximum": df["lithology_code_objectid_invalid"].max(),
                "missing_or_unknown": df["lithology_code_objectid_invalid"].isna().sum(),
            },
            {
                "field": "corrected_geological_type_code",
                "unique_values": df["lithology_code"].nunique(dropna=True),
                "minimum": df["lithology_code"].min(),
                "maximum": df["lithology_code"].max(),
                "missing_or_unknown": (df["lithology_code"] < 0).sum(),
            },
        ]
    )
    qa.to_csv(QA_CSV, index=False)
    manifest = {
        "source_shapefile": str(GEOLOGY),
        "source_attribute": "type",
        "class_count": len(mapping),
        "output_raster": str(OUTPUT_RASTER),
        "output_sample_table": str(OUTPUT_TABLE),
        "historical_problem": "OBJECTID_1 polygon identifiers were independently factorised in sample and map workflows",
        "historical_files_overwritten": False,
    }
    (WORK_DIR / "lithology_correction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return qa


def main() -> None:
    mapping = create_raster()
    qa = update_samples(mapping)
    print(pd.read_csv(MAPPING_CSV).to_string(index=False))
    print(qa.to_string(index=False))
    print(OUTPUT_RASTER)
    print(OUTPUT_TABLE)


if __name__ == "__main__":
    main()
