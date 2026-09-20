"""Build and export the clean 2018 CPEC LSM sampling table in Earth Engine.

This creates fresh terrain-stratified non-landslide samples, merges them with
clean positive inventory candidates, assigns spatial folds, extracts the 2018
factor stack, and exports the resulting table to a managed GEE asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee


PROJECT = "ee-mohibullah141300"
BOUNDARY = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"
INVENTORY = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/inventory/cpec_inventory_model_candidates_v1"
FACTORS_2018 = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m"
SAMPLES_ASSET_DEFAULT = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
PLAN_PATH = Path(r"D:\DING PROJECT\03_models\sampling_2018_export_plan.json")


def build_samples(scale: int, seed: int, positive_buffer_m: int, negative_ratio: float, negative_multiplier: float) -> ee.FeatureCollection:
    study_area = ee.FeatureCollection(BOUNDARY)
    positives = ee.FeatureCollection(INVENTORY).filter(ee.Filter.eq("use_role", "train_candidate"))
    factors = ee.Image(FACTORS_2018)

    valid_mask = factors.select("slope_deg").mask()
    buffered_positives = positives.map(lambda f: f.buffer(positive_buffer_m))
    positive_mask = ee.Image(0).byte().paint(buffered_positives, 1).selfMask()
    negative_mask = valid_mask.updateMask(positive_mask.unmask(0).Not()).clip(study_area)

    slope = factors.select("slope_deg")
    elevation = factors.select("elevation_m")
    slope_class = slope.expression(
        "b('slope_deg') < 5 ? 1"
        ": b('slope_deg') < 15 ? 2"
        ": b('slope_deg') < 30 ? 3"
        ": b('slope_deg') < 45 ? 4"
        ": 5"
    ).rename("slope_class")
    elev_class = elevation.expression(
        "b('elevation_m') < 500 ? 1"
        ": b('elevation_m') < 1500 ? 2"
        ": b('elevation_m') < 3000 ? 3"
        ": b('elevation_m') < 4500 ? 4"
        ": 5"
    ).rename("elev_class")
    strata = slope_class.multiply(10).add(elev_class).rename("strata").updateMask(negative_mask)

    positive_count = positives.size()
    negative_total = positive_count.multiply(negative_ratio).round()
    negative_points = strata.stratifiedSample(
        numPoints=positive_count.multiply(negative_ratio).multiply(negative_multiplier).divide(25).ceil(),
        classBand="strata",
        region=study_area.geometry(),
        scale=scale,
        seed=seed,
        geometries=True,
        dropNulls=True,
        tileScale=4,
    ).randomColumn("negative_random", seed).sort("negative_random").limit(negative_total)

    negative_points = negative_points.map(
        lambda f: f.set(
            {
                "label": 0,
                "use_role": "negative_train_candidate",
                "source": "GEE_stratified_negative_sampling",
                "seed": seed,
                "positive_buffer_m": positive_buffer_m,
                "negative_ratio": negative_ratio,
                "inventory_id": "",
                "hazard_type": "non_landslide",
                "event_year": "",
                "confidence": "generated_negative",
            }
        )
    )
    positive_samples = positives.map(lambda f: f.set("label", 1))
    samples = positive_samples.merge(negative_points)

    def add_fold(f):
        xy = f.geometry().coordinates()
        lon_block = ee.Number(xy.get(0)).floor()
        lat_block = ee.Number(xy.get(1)).floor()
        block_id = lon_block.format("%d").cat("_").cat(lat_block.format("%d"))
        fold = lon_block.multiply(31).add(lat_block.multiply(17)).abs().mod(5).add(1)
        return f.set({"spatial_block_1deg": block_id, "spatial_fold_5": fold})

    samples = samples.map(add_fold)
    return factors.sampleRegions(
        collection=samples,
        properties=[
            "label",
            "inventory_id",
            "source",
            "hazard_type",
            "event_year",
            "confidence",
            "use_role",
            "spatial_block_1deg",
            "spatial_fold_5",
            "seed",
            "positive_buffer_m",
            "negative_ratio",
        ],
        scale=scale,
        geometries=True,
        tileScale=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=250)
    parser.add_argument("--seed", type=int, default=141300)
    parser.add_argument("--positive-buffer-m", type=int, default=500)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-multiplier", type=float, default=8.0)
    parser.add_argument("--asset-id", default=SAMPLES_ASSET_DEFAULT)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    positives = ee.FeatureCollection(INVENTORY).filter(ee.Filter.eq("use_role", "train_candidate"))
    positive_count = positives.size().getInfo()
    sampled = build_samples(args.scale, args.seed, args.positive_buffer_m, args.negative_ratio, args.negative_multiplier)
    plan = {
        "asset": args.asset_id,
        "factor_asset": FACTORS_2018,
        "inventory_asset": INVENTORY,
        "scale": args.scale,
        "seed": args.seed,
        "positive_buffer_m": args.positive_buffer_m,
        "negative_ratio": args.negative_ratio,
        "negative_multiplier": args.negative_multiplier,
        "positive_train_candidate_count": positive_count,
        "target_negative_count": int(round(positive_count * args.negative_ratio)),
        "estimated_total_before_mask_drop": int(round(positive_count * (1 + args.negative_ratio))),
    }
    if args.start:
        task = ee.batch.Export.table.toAsset(
            collection=sampled,
            description=Path(args.asset_id).name,
            assetId=args.asset_id,
        )
        task.start()
        plan["task_id"] = task.id

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    print(f"Saved plan to {PLAN_PATH}")


if __name__ == "__main__":
    main()
