"""Train deployable GEE models and export 2018 susceptibility probability maps.

No susceptibility classes are created. Outputs are continuous probabilities and
continuous model-difference maps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
ASSET_ROOT = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps"
PLAN = Path(r"D:\DING PROJECT\04_maps\gee_2018_probability_map_export_plan.json")


CONVENTIONAL_REDUCED = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
]

ALPHA = [f"A{i:02d}" for i in range(64)]


def make_classifier(seed: int) -> ee.Classifier:
    return ee.Classifier.smileGradientTreeBoost(
        numberOfTrees=500,
        shrinkage=0.03,
        samplingRate=0.85,
        maxNodes=32,
        loss="Logistic",
        seed=seed,
    ).setOutputMode("PROBABILITY")


def train_probability_image(feature_name: str, features: list[str], seed: int) -> ee.Image:
    samples = ee.FeatureCollection(SAMPLES).filter(ee.Filter.inList("label", [0, 1]))
    factors = ee.Image(FACTORS_2018)
    classifier = make_classifier(seed).train(
        features=samples,
        classProperty="label",
        inputProperties=features,
    )
    return (
        factors.select(features)
        .classify(classifier)
        .rename(feature_name)
        .toFloat()
        .clip(ee.FeatureCollection(BOUNDARY))
        .set(
            {
                "year": 2018,
                "model_family": "GEE_smileGradientTreeBoost",
                "feature_set": feature_name,
                "sample_asset": SAMPLES,
                "factor_asset": FACTORS_2018,
                "output_type": "continuous_probability_no_classes",
                "seed": seed,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=250)
    parser.add_argument("--seed", type=int, default=141300)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    study_area = ee.FeatureCollection(BOUNDARY)
    feature_sets = {
        "conventional_reduced": CONVENTIONAL_REDUCED,
        "alphaearth_only": ALPHA,
        "fused_reduced": CONVENTIONAL_REDUCED + ALPHA,
    }

    images = {
        name: train_probability_image(f"prob_{name}", features, args.seed)
        for name, features in feature_sets.items()
    }
    images["diff_fused_minus_conventional_reduced"] = (
        images["fused_reduced"]
        .subtract(images["conventional_reduced"])
        .rename("diff_fused_minus_conventional_reduced")
        .toFloat()
        .set({"output_type": "continuous_probability_difference_no_classes", "year": 2018})
    )
    images["diff_fused_minus_alphaearth_only"] = (
        images["fused_reduced"]
        .subtract(images["alphaearth_only"])
        .rename("diff_fused_minus_alphaearth_only")
        .toFloat()
        .set({"output_type": "continuous_probability_difference_no_classes", "year": 2018})
    )

    plan = []
    for name, image in images.items():
        asset_id = f"{ASSET_ROOT}/cpec_2018_{name}_probability_250m"
        item = {"name": name, "asset_id": asset_id, "scale": args.scale}
        if args.start:
            task = ee.batch.Export.image.toAsset(
                image=image,
                description=f"cpec_2018_{name}_probability_250m",
                assetId=asset_id,
                region=study_area.geometry(),
                scale=args.scale,
                crs="EPSG:4326",
                maxPixels=1e13,
            )
            task.start()
            item["task_id"] = task.id
        plan.append(item)

    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    print(f"Saved plan to {PLAN}")


if __name__ == "__main__":
    main()
