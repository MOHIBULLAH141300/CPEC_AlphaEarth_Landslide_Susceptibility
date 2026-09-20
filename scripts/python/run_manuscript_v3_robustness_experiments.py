"""Sampling, inventory-typology, and road-distance robustness for manuscript v3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    FeatureSpec,
    INNER_BUFFER_KM,
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    nested_feature_set,
    pooled_metrics,
)
from run_manuscript_v3_nested_spatial_cv import load_data as load_primary_data  # noqa: E402


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_robustness_experiments"
PRIMARY_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"
RASTER_MATCH_DIR = PROJECT_ROOT / "01_clean_data" / "manuscript_v3_raster_matched_controls"

MATCH_CONTINUOUS = [
    "elevation_m",
    "slope_deg",
    "log1p_dist_road_m",
    "log1p_dist_river_m",
    "log1p_dist_fault_m",
    "rain_monsoon_total",
]
MATCH_CATEGORICAL = ["cpec_admin_transfer_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--matched-repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def propensity_scores(df: pd.DataFrame) -> np.ndarray:
    preprocess = ColumnTransformer(
        [
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                MATCH_CONTINUOUS,
            ),
            (
                "domain",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                MATCH_CATEGORICAL,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED),
            ),
        ]
    )
    model.fit(df[MATCH_CONTINUOUS + MATCH_CATEGORICAL], df["label"].to_numpy(int))
    return model.predict_proba(df[MATCH_CONTINUOUS + MATCH_CATEGORICAL])[:, 1]


def draw_accessibility_matched_controls(
    df: pd.DataFrame, n_controls: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["propensity"] = propensity_scores(work)
    positives = work[work["label"] == 1]
    controls = work[work["label"] == 0].copy()
    odds = controls["propensity"].to_numpy(float) / np.clip(
        1.0 - controls["propensity"].to_numpy(float), 1e-6, None
    )
    odds = np.minimum(odds, np.quantile(odds, 0.99))
    probabilities = odds / odds.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(controls.index.to_numpy(), size=n_controls, replace=False, p=probabilities)
    selected = pd.concat([positives, controls.loc[chosen]], ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return selected, standardised_difference_table(work, selected, seed)


def standardised_difference_table(
    original: pd.DataFrame, matched: pd.DataFrame, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in MATCH_CONTINUOUS:
        for design, data in [("original", original), ("matched", matched)]:
            pos = pd.to_numeric(data.loc[data.label == 1, factor], errors="coerce")
            neg = pd.to_numeric(data.loc[data.label == 0, factor], errors="coerce")
            pooled_sd = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
            smd = (pos.mean() - neg.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            rows.append(
                {
                    "seed": seed,
                    "design": design,
                    "factor": factor,
                    "positive_mean": pos.mean(),
                    "control_mean": neg.mean(),
                    "standardised_mean_difference": smd,
                    "absolute_standardised_mean_difference": abs(smd),
                }
            )
    return pd.DataFrame(rows)


def run_one(
    df: pd.DataFrame,
    spec: FeatureSpec,
    experiment: str,
    out_dir: Path,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Running {experiment}: n={len(df)}, positives={int(df.label.sum())}", flush=True)
    theta = np.deg2rad(pd.to_numeric(df["aspect_deg"], errors="coerce"))
    df = df.copy()
    df["aspect_sin"] = np.sin(theta)
    df["aspect_cos"] = np.cos(theta)
    predictions, folds, coefficients, _ = nested_feature_set(
        df.reset_index(drop=True),
        spec,
        outer_buffer_km=OUTER_BUFFER_KM,
        inner_buffer_km=INNER_BUFFER_KM,
        quick=quick,
        fit_final_models=False,
    )
    experiment_dir = out_dir / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "sample_id",
        "inventory_id",
        "label",
        "hazard_type",
        "spatial_block_1deg",
        "spatial_fold_5",
        "longitude",
        "latitude",
        "cpec_admin_transfer_domain",
        "feature_set",
        "feature_set_name",
        "stacked_score",
        "threshold",
        "prediction",
    ]
    predictions[keep].to_csv(experiment_dir / "stacked_predictions.csv", index=False)
    folds.to_csv(experiment_dir / "fold_metrics.csv", index=False)
    coefficients.to_csv(experiment_dir / "meta_coefficients.csv", index=False)
    summary = pooled_metrics(predictions)
    summary.insert(0, "experiment", experiment)
    summary.to_csv(experiment_dir / "pooled_metrics.csv", index=False)
    return summary, folds.assign(experiment=experiment)


def main() -> None:
    args = parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_primary_data()
    fused = specs["conventional_alphaearth_embeddings"]
    summaries: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []
    balance_tables: list[pd.DataFrame] = []

    raster_balance = pd.read_csv(RASTER_MATCH_DIR / "raster_matched_control_balance.csv")
    for repeat in range(1, args.matched_repeats + 1):
        matched = pd.read_csv(
            RASTER_MATCH_DIR / f"cpec_baseline_raster_matched_controls_repeat_{repeat:02d}.csv",
            encoding="utf-8-sig",
        )
        balance = raster_balance[raster_balance["repeat"] == repeat].copy()
        experiment = f"accessibility_matched_controls_repeat_{repeat:02d}"
        summary, folds = run_one(matched, fused, experiment, args.out_dir, args.quick)
        summaries.append(summary)
        fold_tables.append(folds)
        balance_tables.append(balance.assign(experiment=experiment, design="raster_matched"))

    from build_manuscript_v3_raster_matched_controls import (  # noqa: E402
        assemble_table,
        greedy_match,
        standardised_differences,
    )

    candidate_pool = pd.read_parquet(RASTER_MATCH_DIR / "complete_raster_candidate_pool.parquet")
    for hazard_type in ["landslide", "rockfall"]:
        positives = df[(df.label == 1) & (df.hazard_type == hazard_type)].copy().reset_index(drop=True)
        controls = greedy_match(positives, candidate_pool, SEED + 100)
        balance = standardised_differences(positives, controls, repeat=100)
        matched = assemble_table(positives, controls, repeat=100)
        experiment = f"inventory_typology_{hazard_type}_only"
        summary, folds = run_one(matched, fused, experiment, args.out_dir, args.quick)
        summaries.append(summary)
        fold_tables.append(folds)
        balance_tables.append(balance.assign(experiment=experiment, design="raster_matched"))

    no_road = FeatureSpec(
        key="conventional_alphaearth_embeddings_no_road_distance",
        display_name="Conventional + AlphaEarth Embeddings (road-distance ablation)",
        continuous=[c for c in fused.continuous if c != "log1p_dist_road_m"],
        categorical=fused.categorical,
    )
    summary, folds = run_one(df, no_road, "road_distance_ablation", args.out_dir, args.quick)
    summaries.append(summary)
    fold_tables.append(folds)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_folds = pd.concat(fold_tables, ignore_index=True)
    balance = pd.concat(balance_tables, ignore_index=True)
    primary = pd.read_csv(PRIMARY_DIR / "nested_spatial_cv_pooled_metrics.csv")
    primary_fused = primary[primary.feature_set == "conventional_alphaearth_embeddings"].iloc[0]
    all_summary["delta_roc_auc_vs_primary_fused"] = all_summary.roc_auc - primary_fused.roc_auc
    all_summary["delta_pr_auc_vs_primary_fused"] = all_summary.pr_auc - primary_fused.pr_auc
    all_summary["delta_brier_vs_primary_fused"] = all_summary.brier - primary_fused.brier
    all_summary.to_csv(args.out_dir / "robustness_experiment_summary.csv", index=False)
    all_folds.to_csv(args.out_dir / "robustness_experiment_fold_metrics.csv", index=False)
    balance.to_csv(args.out_dir / "control_balance_diagnostics.csv", index=False)

    grouped = (
        all_summary[all_summary.experiment.str.startswith("accessibility_matched")]
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_min=("pr_auc", "min"),
            pr_auc_max=("pr_auc", "max"),
            brier_mean=("brier", "mean"),
        )
        .to_dict()
    )
    manifest = {
        "analysis": "manuscript_v3_robustness_experiments",
        "primary_analysis": str(PRIMARY_DIR),
        "matched_control_method": "raster-derived greedy nearest-neighbour matching without replacement",
        "matching_predictors": MATCH_CONTINUOUS,
        "matched_repeats": args.matched_repeats,
        "typology_tests": ["landslide_only", "rockfall_only"],
        "ablation": "remove log-transformed distance-to-road predictor",
        "outer_buffer_km": OUTER_BUFFER_KM,
        "inner_buffer_km": INNER_BUFFER_KM,
        "quick_ensemble": args.quick,
        "elapsed_seconds": time.time() - start,
        "matched_control_summary": grouped,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(all_summary.to_string(index=False), flush=True)
    print(f"Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
