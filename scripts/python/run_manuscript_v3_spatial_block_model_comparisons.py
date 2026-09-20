"""Paired spatial-block bootstrap comparisons among the three v3 feature sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(r"D:\DING PROJECT")
INPUT = (
    PROJECT_ROOT
    / "03_models"
    / "manuscript_v3_nested_spatial_cv"
    / "nested_spatial_cv_predictions.csv"
)
OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_spatial_block_comparisons"
SEED = 141300
REPETITIONS = 2000

COMPARISONS = [
    ("conventional_alphaearth_embeddings", "conventional"),
    ("conventional_alphaearth_embeddings", "alphaearth_embeddings"),
    ("conventional", "alphaearth_embeddings"),
]


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "brier": brier_score_loss(y, score),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    metadata = data[
        ["sample_id", "label", "spatial_block_1deg"]
    ].drop_duplicates("sample_id").set_index("sample_id")
    scores = data.pivot(index="sample_id", columns="feature_set", values="stacked_score")
    table = metadata.join(scores, how="inner").reset_index()
    blocks = table.spatial_block_1deg.astype(str).unique()
    block_rows = {
        block: np.flatnonzero(table.spatial_block_1deg.astype(str).to_numpy() == block)
        for block in blocks
    }
    rng = np.random.default_rng(SEED)
    rows = []
    replicate_rows = []
    y_all = table.label.to_numpy(int)

    for model_a, model_b in COMPARISONS:
        score_a = table[model_a].to_numpy(float)
        score_b = table[model_b].to_numpy(float)
        point_a = metrics(y_all, score_a)
        point_b = metrics(y_all, score_b)
        differences = {metric: [] for metric in point_a}
        for repetition in range(REPETITIONS):
            sampled_blocks = rng.choice(blocks, len(blocks), replace=True)
            idx = np.concatenate([block_rows[block] for block in sampled_blocks])
            y = y_all[idx]
            if len(np.unique(y)) < 2:
                continue
            a = metrics(y, score_a[idx])
            b = metrics(y, score_b[idx])
            for metric in differences:
                difference = a[metric] - b[metric]
                differences[metric].append(difference)
                replicate_rows.append(
                    {
                        "comparison_a": model_a,
                        "comparison_b": model_b,
                        "repetition": repetition,
                        "metric": metric,
                        "difference_a_minus_b": difference,
                    }
                )
        for metric, values in differences.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "comparison_a": model_a,
                    "comparison_b": model_b,
                    "metric": metric,
                    "estimate_a": point_a[metric],
                    "estimate_b": point_b[metric],
                    "difference_a_minus_b": point_a[metric] - point_b[metric],
                    "ci95_low": np.quantile(array, 0.025),
                    "ci95_high": np.quantile(array, 0.975),
                    "two_sided_bootstrap_p": min(
                        1.0,
                        2.0 * min(float((array <= 0).mean()), float((array >= 0).mean())),
                    ),
                    "bootstrap_repetitions": len(array),
                    "resampling_unit": "1-degree spatial block",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "paired_spatial_block_model_comparisons.csv", index=False)
    pd.DataFrame(replicate_rows).to_parquet(
        OUT_DIR / "paired_spatial_block_bootstrap_replicates.parquet", index=False
    )
    manifest = {
        "input": str(INPUT),
        "method": "paired nonparametric bootstrap of 1-degree spatial blocks",
        "repetitions": REPETITIONS,
        "paired_unit": "same held-out sample predictions for each feature set",
        "reason": "accounts for spatial clustering more appropriately than observation-level DeLong testing",
        "metrics": ["ROC-AUC", "PR-AUC", "Brier score"],
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(OUT_DIR)


if __name__ == "__main__":
    main()
