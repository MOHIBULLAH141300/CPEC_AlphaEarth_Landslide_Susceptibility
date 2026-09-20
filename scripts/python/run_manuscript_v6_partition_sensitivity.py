"""Re-run fully nested LODO evaluation with an alternative physiographic partition.

The primary manuscript partition is administrative-geographic. This analysis
uses deterministic latitude, longitude, and elevation rules to test whether
the comparative feature-set conclusion depends on those boundaries.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_manuscript_v3_leave_one_domain_out as lodo  # noqa: E402
from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    INNER_BUFFER_KM,
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    load_data,
)


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v6_partition_sensitivity"
DOMAIN_ORDER = [
    "Northern high mountains",
    "Western arid region",
    "Indus lowland",
    "Southern coastal region",
    "Transitional foothills and plateaus",
]


def assign_physiographic_domain(df: pd.DataFrame) -> pd.Series:
    longitude = df.longitude.to_numpy(float)
    latitude = df.latitude.to_numpy(float)
    elevation = df.elevation_m.to_numpy(float)
    domain = np.full(len(df), "Transitional foothills and plateaus", dtype=object)

    northern = (latitude >= 34.0) | (elevation >= 2200.0)
    southern = (~northern) & (latitude < 26.5)
    western = (~northern) & (~southern) & (longitude < 68.5)
    lowland = (~northern) & (~southern) & (~western) & (elevation < 700.0)

    domain[northern] = "Northern high mountains"
    domain[southern] = "Southern coastal region"
    domain[western] = "Western arid region"
    domain[lowland] = "Indus lowland"
    return pd.Series(domain, index=df.index)


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    df = df.copy()
    df["primary_admin_transfer_domain"] = df.cpec_admin_transfer_domain
    df["cpec_admin_transfer_domain"] = assign_physiographic_domain(df)

    lodo.DOMAIN_DISPLAY.update({domain: domain for domain in DOMAIN_ORDER})
    balance = (
        df.groupby("cpec_admin_transfer_domain")
        .label.agg(n="count", positives="sum")
        .reindex(DOMAIN_ORDER)
        .reset_index()
        .rename(columns={"cpec_admin_transfer_domain": "held_out_domain"})
    )
    balance["controls"] = balance.n - balance.positives
    balance["positive_share"] = balance.positives / balance.n
    balance.to_csv(OUT_DIR / "alternative_partition_sample_balance.csv", index=False)

    predictions, metrics, coefficients = [], [], []
    for spec in specs.values():
        for domain in DOMAIN_ORDER:
            print(f"Alternative LODO: {spec.display_name}; held out {domain}", flush=True)
            prediction, row, coefficient = lodo.run_transfer(df, spec, domain)
            predictions.append(prediction)
            metrics.append(row)
            coefficients.append(coefficient)

    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_table = pd.DataFrame(metrics)
    coefficient_table = pd.concat(coefficients, ignore_index=True)
    intervals = lodo.block_bootstrap_ci(prediction_table, repetitions=2000)

    prediction_table.to_csv(
        OUT_DIR / "alternative_partition_lodo_predictions.csv", index=False
    )
    metric_table.to_csv(OUT_DIR / "alternative_partition_lodo_metrics.csv", index=False)
    coefficient_table.to_csv(
        OUT_DIR / "alternative_partition_lodo_meta_coefficients.csv", index=False
    )
    intervals.to_csv(
        OUT_DIR / "alternative_partition_lodo_block_bootstrap_ci.csv", index=False
    )

    macro = (
        metric_table.groupby(["feature_set", "feature_set_name"])
        .agg(
            macro_mean_roc_auc=("roc_auc", "mean"),
            macro_mean_pr_auc=("pr_auc", "mean"),
            macro_mean_brier=("brier", "mean"),
            minimum_domain_roc_auc=("roc_auc", "min"),
            maximum_domain_roc_auc=("roc_auc", "max"),
        )
        .reset_index()
    )
    pivot = metric_table.pivot(
        index="held_out_domain", columns="feature_set", values="roc_auc"
    )
    comparison = pd.DataFrame(
        {
            "held_out_domain": pivot.index,
            "fusion_minus_conventional_roc_auc": pivot[
                "conventional_alphaearth_embeddings"
            ]
            - pivot["conventional"],
            "alphaearth_minus_conventional_roc_auc": pivot["alphaearth_embeddings"]
            - pivot["conventional"],
        }
    ).reset_index(drop=True)
    macro.to_csv(OUT_DIR / "alternative_partition_macro_summary.csv", index=False)
    comparison.to_csv(
        OUT_DIR / "alternative_partition_feature_set_comparison.csv", index=False
    )

    manifest = {
        "analysis": "manuscript_v6_alternative_partition_sensitivity",
        "purpose": "test dependence of LODO conclusions on the primary administrative-geographic boundaries",
        "partition_rules": {
            "Northern high mountains": "latitude >= 34 degrees or elevation >= 2200 m",
            "Southern coastal region": "remaining samples with latitude < 26.5 degrees",
            "Western arid region": "remaining samples with longitude < 68.5 degrees",
            "Indus lowland": "remaining samples with elevation < 700 m",
            "Transitional foothills and plateaus": "all remaining samples",
        },
        "outer_buffer_km": OUTER_BUFFER_KM,
        "inner_buffer_km": INNER_BUFFER_KM,
        "inner_validation": "five existing one-degree spatial-block folds within source domains",
        "feature_sets": [spec.display_name for spec in specs.values()],
        "bootstrap": "2000 repetitions resampling one-degree blocks within held-out domains",
        "random_seed": SEED,
        "elapsed_seconds": time.time() - start,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nAlternative-partition macro summary\n", macro.to_string(index=False), flush=True)
    print("\nFusion contrasts\n", comparison.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
