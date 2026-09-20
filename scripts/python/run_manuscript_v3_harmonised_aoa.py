"""Harmonised area-of-applicability diagnostic for manuscript v3.

All three feature sets are projected to the same 15-dimensional latent space.
The applicability threshold is the 95th percentile of nearest-neighbour
distances between different 1-degree training blocks. This avoids comparing
raw distances produced by feature spaces of different dimensionality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    buffered_indices,
    load_data,
    make_preprocessor,
)
from run_manuscript_v3_leave_one_domain_out import DOMAIN_DISPLAY  # noqa: E402


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_harmonised_aoa"
TRANSFER_METRICS = (
    PROJECT_ROOT
    / "03_models"
    / "manuscript_v3_leave_one_domain_out"
    / "leave_one_domain_out_metrics.csv"
)
LATENT_DIMENSIONS = 15
THRESHOLD_QUANTILE = 0.95


def different_block_distance(x: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    neighbours = min(200, len(x))
    model = NearestNeighbors(n_neighbors=neighbours, n_jobs=-1).fit(x)
    distances, indices = model.kneighbors(x)
    out = np.full(len(x), np.nan, dtype=float)
    for row in range(len(x)):
        valid = np.flatnonzero(blocks[indices[row]] != blocks[row])
        if len(valid):
            out[row] = distances[row, valid[0]]
    if np.isnan(out).any():
        raise RuntimeError("Could not find a neighbour from a different spatial block.")
    return out


def coverage_interval(data: pd.DataFrame, repetitions: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    blocks = data.spatial_block_1deg.astype(str).unique()
    by_block = {
        block: data.loc[data.spatial_block_1deg.astype(str) == block, "inside_aoa"].to_numpy(float)
        for block in blocks
    }
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(blocks, len(blocks), replace=True)
        values.append(np.concatenate([by_block[block] for block in sampled]).mean())
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    domains = sorted(df.cpec_admin_transfer_domain.astype(str).unique())
    all_idx = np.arange(len(df))
    rows, summary = [], []

    for spec in specs.values():
        for domain in domains:
            test_idx = all_idx[df.cpec_admin_transfer_domain.astype(str).to_numpy() == domain]
            raw_train = all_idx[df.cpec_admin_transfer_domain.astype(str).to_numpy() != domain]
            train_idx = buffered_indices(df, raw_train, test_idx, OUTER_BUFFER_KM)
            preprocess = make_preprocessor(spec)
            x_train = preprocess.fit_transform(df.loc[train_idx, spec.columns])
            x_test = preprocess.transform(df.loc[test_idx, spec.columns])
            dimensions = min(LATENT_DIMENSIONS, x_train.shape[1], len(train_idx) - 1)
            pca = PCA(n_components=dimensions, whiten=True, random_state=SEED)
            z_train = pca.fit_transform(x_train)
            z_test = pca.transform(x_test)
            train_blocks = df.loc[train_idx, "spatial_block_1deg"].astype(str).to_numpy()
            reference_distance = different_block_distance(z_train, train_blocks)
            threshold = float(np.quantile(reference_distance, THRESHOLD_QUANTILE))
            test_distance = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_train).kneighbors(
                z_test, return_distance=True
            )[0].ravel()
            di = test_distance / threshold
            output = df.loc[
                test_idx,
                [
                    "label",
                    "hazard_type",
                    "spatial_block_1deg",
                    "longitude",
                    "latitude",
                ],
            ].copy()
            output["sample_id"] = test_idx
            output["feature_set"] = spec.key
            output["feature_set_name"] = spec.display_name
            output["held_out_domain"] = DOMAIN_DISPLAY[domain]
            output["dissimilarity_index"] = di
            output["inside_aoa"] = di <= 1.0
            rows.append(output)
            low, high = coverage_interval(output)
            summary.append(
                {
                    "feature_set": spec.key,
                    "feature_set_name": spec.display_name,
                    "held_out_domain": DOMAIN_DISPLAY[domain],
                    "n_test": len(test_idx),
                    "n_positive": int(df.loc[test_idx, "label"].sum()),
                    "n_negative": int((df.loc[test_idx, "label"] == 0).sum()),
                    "latent_dimensions": dimensions,
                    "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
                    "reference_distance_threshold_p95": threshold,
                    "aoa_coverage": float((di <= 1.0).mean()),
                    "aoa_coverage_ci95_low": low,
                    "aoa_coverage_ci95_high": high,
                    "median_dissimilarity_index": float(np.median(di)),
                    "p90_dissimilarity_index": float(np.quantile(di, 0.9)),
                }
            )
            print(
                f"{spec.display_name}; {DOMAIN_DISPLAY[domain]}: coverage={(di <= 1).mean():.3f}",
                flush=True,
            )

    sample_table = pd.concat(rows, ignore_index=True)
    summary_table = pd.DataFrame(summary)
    transfer = pd.read_csv(TRANSFER_METRICS)
    joined = summary_table.merge(
        transfer[
            [
                "feature_set",
                "held_out_domain",
                "roc_auc",
                "pr_auc",
                "brier",
                "ece_10bin",
            ]
        ],
        on=["feature_set", "held_out_domain"],
        how="left",
    )
    correlation_rows = []
    for metric in ["roc_auc", "pr_auc", "brier", "ece_10bin"]:
        rho, p_value = spearmanr(joined.aoa_coverage, joined[metric])
        correlation_rows.append(
            {
                "comparison": f"aoa_coverage_vs_{metric}",
                "spearman_rho": rho,
                "p_value": p_value,
                "n_feature_set_domain_pairs": len(joined),
            }
        )
    sample_table.to_csv(OUT_DIR / "harmonised_aoa_sample_diagnostics.csv", index=False)
    joined.to_csv(OUT_DIR / "harmonised_aoa_domain_summary.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(OUT_DIR / "aoa_transfer_metric_correlations.csv", index=False)
    manifest = {
        "analysis": "harmonised_leave_one_domain_out_area_of_applicability",
        "latent_dimensions": LATENT_DIMENSIONS,
        "latent_method": "PCA with whitening, fitted only on source-domain training samples",
        "reference_distance": "nearest sample from a different 1-degree source training block",
        "threshold_quantile": THRESHOLD_QUANTILE,
        "held_out_domain_buffer_km": OUTER_BUFFER_KM,
        "comparison_note": "fixed latent dimensionality and threshold definition are used for all feature sets",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(joined.to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
