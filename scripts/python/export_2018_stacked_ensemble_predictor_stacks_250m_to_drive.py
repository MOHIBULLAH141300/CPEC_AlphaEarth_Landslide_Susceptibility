"""Export predictor stacks for the three 2018 Spatial-CV Stacked Ensemble maps.

The stacked models are local Python models, so Earth Engine is used only to
prepare complete predictor rasters. The local apply script then creates the
stacked-ensemble probability rasters.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
DRIVE_FOLDER = "GEE_CPEC_LSM_STACKED_ENSEMBLE_INPUTS_250M"
PLAN = Path(r"D:\DING PROJECT\04_maps\stacked_ensemble_predictor_stacks_250m_export_plan.json")
CONVENTIONAL_LIST = (
    Path(r"D:\DING PROJECT")
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)
LOCAL_SAMPLE_TABLE = Path(r"D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v3.csv")

ALPHA = [f"A{i:02d}" for i in range(64)]


def read_conventional() -> list[str]:
    import pandas as pd

    return pd.read_csv(CONVENTIONAL_LIST)["factor"].tolist()


def sample_means(bands: list[str]) -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(LOCAL_SAMPLE_TABLE, encoding="utf-8-sig")
    return {band: float(pd.to_numeric(df[band], errors="coerce").mean()) for band in bands}


def fill_predictor_masks(image: ee.Image, bands: list[str], means: dict[str, float]) -> ee.Image:
    filled = []
    for band in bands:
        fill_value = ee.Number(means[band])
        filled.append(image.select(band).unmask(fill_value, False).rename(band))
    return ee.Image.cat(filled)


def export_stack(name: str, bands: list[str], study_area: ee.FeatureCollection) -> dict[str, object]:
    raw = ee.Image(FACTORS_2018).select(bands)
    means = sample_means(bands)
    stack = (
        fill_predictor_masks(raw, bands, means)
        .toFloat()
        .clip(study_area)
        .set(
            {
                "year": 2018,
                "resolution_m": 250,
                "feature_set": name,
                "output_type": "predictor_stack_for_local_spatial_cv_stacked_ensemble",
                "feature_count": len(bands),
                "sample_table": str(LOCAL_SAMPLE_TABLE),
                "factor_asset": FACTORS_2018,
                "nodata_policy": "predictor masks filled from training sample means using unmask(..., sameFootprint=False)",
            }
        )
    )
    prefix = f"cpec_2018_{name}_predictor_stack_250m_no_nodata"
    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=prefix,
        region=study_area.geometry(),
        scale=250,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return {
        "name": name,
        "drive_folder": DRIVE_FOLDER,
        "file_prefix": prefix,
        "task_id": task.id,
        "resolution_m": 250,
        "band_count": len(bands),
        "bands": bands,
        "fill_values_source": str(LOCAL_SAMPLE_TABLE),
    }


def main() -> None:
    ee.Initialize(project=PROJECT)
    study_area = ee.FeatureCollection(BOUNDARY)
    conventional = read_conventional()
    outputs = {
        "conventional": conventional,
        "alphaearth_embeddings": ALPHA,
        "conventional_alphaearth_embeddings": conventional + ALPHA,
    }
    plan = [export_stack(name, bands, study_area) for name, bands in outputs.items()]
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
