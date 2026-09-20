"""Sensitivity of the fused nested stack to buffers and spatial block size."""

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

from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    PROJECT_ROOT,
    SEED,
    nested_feature_set,
    pooled_metrics,
    load_data,
)


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v6_spatial_design_sensitivity"
PRIMARY_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"
BUFFER_VALUES = [0.0, 10.0, 40.0]
BLOCK_SIZES = [0.5, 2.0]


def balanced_block_folds(df: pd.DataFrame, block_size_deg: float) -> pd.DataFrame:
    """Create deterministic five-fold groups from square geographic blocks."""
    work = df.copy()
    bx = np.floor(work.longitude.to_numpy(float) / block_size_deg).astype(int)
    by = np.floor(work.latitude.to_numpy(float) / block_size_deg).astype(int)
    work["spatial_block_1deg"] = [f"{block_size_deg:g}:{x}:{y}" for x, y in zip(bx, by)]
    stats = (
        work.groupby("spatial_block_1deg")
        .label.agg(n="count", positives="sum")
        .reset_index()
    )
    stats["controls"] = stats.n - stats.positives
    stats = stats.sort_values(
        ["n", "positives", "controls", "spatial_block_1deg"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    totals = stats[["n", "positives", "controls"]].sum().to_numpy(float) / 5.0
    fold_totals = np.zeros((5, 3), dtype=float)
    assignment: dict[str, int] = {}
    for row_number, row in stats.iterrows():
        values = row[["n", "positives", "controls"]].to_numpy(float)
        if row_number < 5:
            chosen = row_number
        else:
            costs = []
            for fold in range(5):
                trial = fold_totals.copy()
                trial[fold] += values
                imbalance = np.sum(((trial - totals) / np.clip(totals, 1.0, None)) ** 2)
                costs.append((imbalance, fold_totals[fold, 0], fold))
            chosen = min(costs)[2]
        assignment[str(row.spatial_block_1deg)] = int(chosen + 1)
        fold_totals[chosen] += values

    work["spatial_fold_5"] = work.spatial_block_1deg.map(assignment).astype(int)
    balance = work.groupby("spatial_fold_5").label.agg(n="count", positives="sum")
    if len(balance) != 5 or (balance.positives == 0).any() or (balance["n"] - balance.positives == 0).any():
        raise RuntimeError(f"Invalid class balance for {block_size_deg:g}-degree folds")
    return work


def save_run(
    name: str,
    df: pd.DataFrame,
    spec,
    outer_buffer_km: float,
    inner_buffer_km: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    run_dir = OUT_DIR / name
    metrics_file = run_dir / "pooled_metrics.csv"
    if metrics_file.exists():
        row = pd.read_csv(metrics_file).iloc[0].to_dict()
        print(f"Using existing sensitivity run: {name}", flush=True)
        return {
            **metadata,
            "outer_buffer_km": outer_buffer_km,
            "inner_buffer_km": inner_buffer_km,
            **row,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(
        f"Running {name}: outer buffer={outer_buffer_km:g} km, inner buffer={inner_buffer_km:g} km",
        flush=True,
    )
    predictions, fold_metrics, coefficients, _ = nested_feature_set(
        df,
        spec,
        outer_buffer_km=outer_buffer_km,
        inner_buffer_km=inner_buffer_km,
        quick=False,
        fit_final_models=False,
    )
    pooled = pooled_metrics(predictions)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    coefficients.to_csv(run_dir / "meta_coefficients.csv", index=False)
    pooled.to_csv(metrics_file, index=False)
    run_manifest = {
        **metadata,
        "outer_buffer_km": outer_buffer_km,
        "inner_buffer_km": inner_buffer_km,
        "elapsed_seconds": time.time() - start,
        "random_seed": SEED,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    return {
        **metadata,
        "outer_buffer_km": outer_buffer_km,
        "inner_buffer_km": inner_buffer_km,
        **pooled.iloc[0].to_dict(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    spec = specs["conventional_alphaearth_embeddings"]
    rows: list[dict[str, object]] = []

    primary = pd.read_csv(PRIMARY_DIR / "nested_spatial_cv_pooled_metrics.csv")
    primary = primary[
        primary.feature_set == "conventional_alphaearth_embeddings"
    ].iloc[0].to_dict()
    rows.append(
        {
            "sensitivity_type": "primary",
            "setting": "1-degree blocks; 20 km buffers",
            "block_size_deg": 1.0,
            "outer_buffer_km": 20.0,
            "inner_buffer_km": 20.0,
            **primary,
        }
    )

    for buffer_km in BUFFER_VALUES:
        rows.append(
            save_run(
                f"buffer_{buffer_km:g}km",
                df,
                spec,
                outer_buffer_km=buffer_km,
                inner_buffer_km=buffer_km,
                metadata={
                    "sensitivity_type": "buffer",
                    "setting": f"1-degree blocks; {buffer_km:g} km buffers",
                    "block_size_deg": 1.0,
                },
            )
        )

    for block_size in BLOCK_SIZES:
        block_df = balanced_block_folds(df, block_size)
        balance = (
            block_df.groupby("spatial_fold_5")
            .label.agg(n="count", positives="sum")
            .reset_index()
        )
        balance["controls"] = balance.n - balance.positives
        run_name = f"block_{str(block_size).replace('.', 'p')}deg_buffer_20km"
        run_dir = OUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        balance.to_csv(run_dir / "fold_balance.csv", index=False)
        rows.append(
            save_run(
                run_name,
                block_df,
                spec,
                outer_buffer_km=20.0,
                inner_buffer_km=20.0,
                metadata={
                    "sensitivity_type": "block_size",
                    "setting": f"{block_size:g}-degree blocks; 20 km buffers",
                    "block_size_deg": block_size,
                },
            )
        )

    summary = pd.DataFrame(rows)
    keep = [
        "sensitivity_type",
        "setting",
        "block_size_deg",
        "outer_buffer_km",
        "inner_buffer_km",
        "n",
        "n_positive",
        "n_negative",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "brier",
        "ece_10bin",
    ]
    summary = summary[[column for column in keep if column in summary.columns]]
    summary.to_csv(OUT_DIR / "spatial_design_sensitivity_summary.csv", index=False)
    manifest = {
        "analysis": "manuscript_v6_spatial_design_sensitivity",
        "feature_set": spec.display_name,
        "buffer_values_km": [0, 10, 20, 40],
        "block_sizes_degrees": [0.5, 1.0, 2.0],
        "model": "fully nested six-learner spatial-CV stack",
        "alternative_fold_assignment": "deterministic greedy allocation balancing sample size and both classes",
        "random_seed": SEED,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSpatial-design sensitivity\n", summary.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
