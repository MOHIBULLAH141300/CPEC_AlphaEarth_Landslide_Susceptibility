"""Resolve the comparative transfer and AoA issues identified in the V5 audit.

The script uses the existing leakage-resistant LODO predictions rather than
refitting models. It adds paired spatial-block comparisons, a source-only AoA
sensitivity grid, a domain-aware association test, and concise tables for the
V6 manuscript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_leave_one_domain_out import DOMAIN_DISPLAY  # noqa: E402
from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    buffered_indices,
    load_data,
    make_preprocessor,
)


LODO_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_leave_one_domain_out"
AOA_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_harmonised_aoa"
ROBUSTNESS_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_robustness_experiments"
POOLED_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"
OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v6_audit_resolutions"

PREDICTIONS = LODO_DIR / "leave_one_domain_out_predictions.csv"
LODO_METRICS = LODO_DIR / "leave_one_domain_out_metrics.csv"
LODO_INTERVALS = LODO_DIR / "leave_one_domain_out_block_bootstrap_ci.csv"
AOA_SUMMARY = AOA_DIR / "harmonised_aoa_domain_summary.csv"
ROBUSTNESS_SUMMARY = ROBUSTNESS_DIR / "robustness_experiment_summary.csv"
POOLED_METRICS = POOLED_DIR / "nested_spatial_cv_pooled_metrics.csv"

FEATURE_ORDER = [
    "conventional",
    "alphaearth_embeddings",
    "conventional_alphaearth_embeddings",
]
FEATURE_LABEL = {
    "conventional": "Conventional",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_alphaearth_embeddings": "Conventional + AlphaEarth Embeddings",
}
DOMAIN_ORDER = ["Kashgar", "Gilgit-Baltistan", "KPK-AJK", "Balochistan", "Punjab-Sindh"]
DIMENSIONS = [5, 10, 15, 20]
QUANTILES = [0.90, 0.95, 0.975, 0.99]
BOOTSTRAP_REPETITIONS = 2000
PERMUTATION_REPETITIONS = 20000


def metric_value(metric: str, y: np.ndarray, score: np.ndarray) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y, score))
    if metric == "pr_auc":
        return float(average_precision_score(y, score))
    if metric == "brier":
        return float(brier_score_loss(y, score))
    raise ValueError(metric)


def paired_block_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare feature sets on identical held-out samples and spatial blocks."""
    rng = np.random.default_rng(SEED)
    comparisons = [
        ("conventional_alphaearth_embeddings", "conventional", "Fusion - Conventional"),
        ("alphaearth_embeddings", "conventional", "AlphaEarth - Conventional"),
        (
            "conventional_alphaearth_embeddings",
            "alphaearth_embeddings",
            "Fusion - AlphaEarth",
        ),
    ]
    rows: list[dict[str, object]] = []

    def prepare_weighted_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
        return {
            "order": order,
            "starts": starts,
            "y": y[order],
            "squared_error": (score[order] - y[order]) ** 2,
        }

    def weighted_values(prepared: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
        ordered_weights = weights[prepared["order"]]
        y_sorted = prepared["y"]
        positive = np.add.reduceat(ordered_weights * y_sorted, prepared["starts"])
        negative = np.add.reduceat(ordered_weights * (1 - y_sorted), prepared["starts"])
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive <= 0 or total_negative <= 0:
            return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
        negative_before = np.cumsum(negative) - negative
        auc = np.sum(positive * (negative_before + 0.5 * negative)) / (
            total_positive * total_negative
        )
        positive_desc = positive[::-1]
        negative_desc = negative[::-1]
        cumulative_positive = np.cumsum(positive_desc)
        cumulative_total = cumulative_positive + np.cumsum(negative_desc)
        precision = cumulative_positive / np.clip(cumulative_total, 1e-12, None)
        average_precision = np.sum((positive_desc / total_positive) * precision)
        brier = np.average(prepared["squared_error"], weights=ordered_weights)
        return {"roc_auc": float(auc), "pr_auc": float(average_precision), "brier": float(brier)}

    for domain in DOMAIN_ORDER:
        domain_data = predictions[predictions.held_out_domain == domain].copy()
        reference = domain_data[
            domain_data.feature_set == "conventional"
        ].sort_values("sample_id")
        y = reference.label.to_numpy(int)
        block_vector = reference.spatial_block_1deg.astype(str).to_numpy()
        blocks = np.unique(block_vector)
        block_position = {block: pos for pos, block in enumerate(blocks)}
        row_block_position = np.array([block_position[block] for block in block_vector], dtype=int)
        score_by_feature: dict[str, np.ndarray] = {}
        prepared_by_feature: dict[str, dict[str, np.ndarray]] = {}
        point_by_feature: dict[str, dict[str, float]] = {}
        for feature_set in FEATURE_ORDER:
            feature_data = domain_data[
                domain_data.feature_set == feature_set
            ].sort_values("sample_id")
            if not np.array_equal(reference.sample_id.to_numpy(), feature_data.sample_id.to_numpy()):
                raise RuntimeError(f"Unpaired samples for {domain}: {feature_set}")
            if not np.array_equal(y, feature_data.label.to_numpy(int)):
                raise RuntimeError(f"Unpaired labels for {domain}: {feature_set}")
            score = feature_data.stacked_score.to_numpy(float)
            score_by_feature[feature_set] = score
            prepared_by_feature[feature_set] = prepare_weighted_metrics(y, score)
            point_by_feature[feature_set] = {
                metric: metric_value(metric, y, score)
                for metric in ["roc_auc", "pr_auc", "brier"]
            }

        bootstrap_values = {
            feature_set: {metric: [] for metric in ["roc_auc", "pr_auc", "brier"]}
            for feature_set in FEATURE_ORDER
        }
        for _ in range(BOOTSTRAP_REPETITIONS):
            sampled_positions = rng.integers(0, len(blocks), size=len(blocks))
            block_counts = np.bincount(sampled_positions, minlength=len(blocks)).astype(float)
            row_weights = block_counts[row_block_position]
            if np.sum(row_weights * y) <= 0 or np.sum(row_weights * (1 - y)) <= 0:
                continue
            for feature_set in FEATURE_ORDER:
                values = weighted_values(prepared_by_feature[feature_set], row_weights)
                for metric, value in values.items():
                    bootstrap_values[feature_set][metric].append(value)

        for feature_a, feature_b, comparison in comparisons:
            for metric in ["roc_auc", "pr_auc", "brier"]:
                point_a = point_by_feature[feature_a][metric]
                point_b = point_by_feature[feature_b][metric]
                sample_array = np.asarray(bootstrap_values[feature_a][metric]) - np.asarray(
                    bootstrap_values[feature_b][metric]
                )
                lower = float(np.quantile(sample_array, 0.025))
                upper = float(np.quantile(sample_array, 0.975))
                p_value = float(
                    min(
                        1.0,
                        2
                        * min(
                            (np.sum(sample_array <= 0) + 1) / (len(sample_array) + 1),
                            (np.sum(sample_array >= 0) + 1) / (len(sample_array) + 1),
                        ),
                    )
                )
                rows.append(
                    {
                        "held_out_domain": domain,
                        "comparison": comparison,
                        "feature_set_a": feature_a,
                        "feature_set_b": feature_b,
                        "metric": metric,
                        "estimate_a": point_a,
                        "estimate_b": point_b,
                        "difference_a_minus_b": point_a - point_b,
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "bootstrap_p_two_sided": p_value,
                        "bootstrap_repetitions": len(sample_array),
                        "resampling_unit": "1-degree spatial block",
                    }
                )
    return pd.DataFrame(rows)


def comparative_lodo_table(
    metrics: pd.DataFrame, intervals: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    ci = intervals.pivot_table(
        index=["feature_set", "held_out_domain"],
        columns="metric",
        values=["ci95_low", "ci95_high"],
        aggfunc="first",
    )
    rows = []
    for domain in DOMAIN_ORDER:
        for feature_set in FEATURE_ORDER:
            source = metrics[
                (metrics.feature_set == feature_set) & (metrics.held_out_domain == domain)
            ].iloc[0]
            row = {
                "held_out_domain": domain,
                "feature_set": feature_set,
                "feature_set_name": FEATURE_LABEL[feature_set],
                "n_test": int(source.n_test),
                "n_positive": int(source.n_test_positive),
                "n_control": int(source.n_test_negative),
                "roc_auc": source.roc_auc,
                "roc_auc_ci95_low": ci.loc[(feature_set, domain), ("ci95_low", "roc_auc")],
                "roc_auc_ci95_high": ci.loc[(feature_set, domain), ("ci95_high", "roc_auc")],
                "pr_auc": source.pr_auc,
                "pr_auc_ci95_low": ci.loc[(feature_set, domain), ("ci95_low", "pr_auc")],
                "pr_auc_ci95_high": ci.loc[(feature_set, domain), ("ci95_high", "pr_auc")],
                "brier": source.brier,
                "brier_ci95_low": ci.loc[(feature_set, domain), ("ci95_low", "brier")],
                "brier_ci95_high": ci.loc[(feature_set, domain), ("ci95_high", "brier")],
                "ece_10bin": source.ece_10bin,
            }
            if feature_set == "conventional_alphaearth_embeddings":
                for metric in ["roc_auc", "pr_auc", "brier"]:
                    delta = paired[
                        (paired.held_out_domain == domain)
                        & (paired.comparison == "Fusion - Conventional")
                        & (paired.metric == metric)
                    ].iloc[0]
                    row[f"delta_{metric}_vs_conventional"] = delta.difference_a_minus_b
                    row[f"delta_{metric}_ci95_low"] = delta.ci95_low
                    row[f"delta_{metric}_ci95_high"] = delta.ci95_high
                    row[f"delta_{metric}_p"] = delta.bootstrap_p_two_sided
            rows.append(row)
    return pd.DataFrame(rows)


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
        raise RuntimeError("No different-block neighbour found for at least one source sample.")
    return out


def aoa_sensitivity_grid() -> pd.DataFrame:
    """Evaluate LODO support across latent dimensions and threshold quantiles."""
    df, specs, _ = load_data()
    domain_values = df.cpec_admin_transfer_domain.astype(str).to_numpy()
    all_idx = np.arange(len(df))
    rows: list[dict[str, object]] = []
    for feature_set in FEATURE_ORDER:
        spec = specs[feature_set]
        for source_domain in sorted(np.unique(domain_values)):
            domain = DOMAIN_DISPLAY[source_domain]
            test_idx = all_idx[domain_values == source_domain]
            raw_train = all_idx[domain_values != source_domain]
            train_idx = buffered_indices(df, raw_train, test_idx, OUTER_BUFFER_KM)
            preprocess = make_preprocessor(spec)
            x_train = preprocess.fit_transform(df.loc[train_idx, spec.columns])
            x_test = preprocess.transform(df.loc[test_idx, spec.columns])
            max_dimensions = min(max(DIMENSIONS), x_train.shape[1], len(train_idx) - 1)
            pca = PCA(n_components=max_dimensions, whiten=True, random_state=SEED)
            z_train_full = pca.fit_transform(x_train)
            z_test_full = pca.transform(x_test)
            blocks = df.loc[train_idx, "spatial_block_1deg"].astype(str).to_numpy()
            for dimensions in DIMENSIONS:
                if dimensions > max_dimensions:
                    continue
                z_train = z_train_full[:, :dimensions]
                z_test = z_test_full[:, :dimensions]
                reference = different_block_distance(z_train, blocks)
                test_distance = (
                    NearestNeighbors(n_neighbors=1, n_jobs=-1)
                    .fit(z_train)
                    .kneighbors(z_test, return_distance=True)[0]
                    .ravel()
                )
                for quantile in QUANTILES:
                    threshold = float(np.quantile(reference, quantile))
                    di = test_distance / threshold
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "feature_set_name": FEATURE_LABEL[feature_set],
                            "held_out_domain": domain,
                            "n_test": len(test_idx),
                            "latent_dimensions": dimensions,
                            "threshold_quantile": quantile,
                            "explained_variance_ratio": float(
                                pca.explained_variance_ratio_[:dimensions].sum()
                            ),
                            "reference_distance_threshold": threshold,
                            "aoa_coverage": float((di <= 1).mean()),
                            "median_dissimilarity_index": float(np.median(di)),
                            "p90_dissimilarity_index": float(np.quantile(di, 0.90)),
                        }
                    )
            print(f"AoA sensitivity: {FEATURE_LABEL[feature_set]} / {domain}", flush=True)
    return pd.DataFrame(rows)


def summarise_aoa_sensitivity(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        grid.groupby(["feature_set", "feature_set_name", "held_out_domain"])
        .aoa_coverage.agg(["min", "median", "max", "mean", "std"])
        .reset_index()
    )
    summary["range"] = summary["max"] - summary["min"]

    standard = grid[
        (grid.latent_dimensions == 15) & np.isclose(grid.threshold_quantile, 0.95)
    ][["feature_set", "held_out_domain", "aoa_coverage"]].rename(
        columns={"aoa_coverage": "standard_aoa_coverage"}
    )
    summary = summary.merge(standard, on=["feature_set", "held_out_domain"], how="left")

    winners = []
    for (dimensions, quantile, domain), data in grid.groupby(
        ["latent_dimensions", "threshold_quantile", "held_out_domain"]
    ):
        ranked = data.sort_values("aoa_coverage", ascending=False)
        conventional = data[data.feature_set == "conventional"].aoa_coverage.iloc[0]
        fusion = data[
            data.feature_set == "conventional_alphaearth_embeddings"
        ].aoa_coverage.iloc[0]
        winners.append(
            {
                "latent_dimensions": dimensions,
                "threshold_quantile": quantile,
                "held_out_domain": domain,
                "highest_support_feature_set": ranked.feature_set_name.iloc[0],
                "fusion_minus_conventional_coverage": fusion - conventional,
                "fusion_exceeds_conventional": bool(fusion > conventional),
            }
        )
    return summary, pd.DataFrame(winners)


def domain_aware_association(aoa_summary: pd.DataFrame) -> pd.DataFrame:
    """Test feature-set association after removing fixed domain-level differences."""
    rng = np.random.default_rng(SEED)
    data = aoa_summary.copy().sort_values(["held_out_domain", "feature_set"])
    rows = []
    for metric in ["roc_auc", "pr_auc", "brier", "ece_10bin"]:
        full_rho = float(spearmanr(data.aoa_coverage, data[metric]).statistic)
        centered_aoa = data.aoa_coverage - data.groupby("held_out_domain").aoa_coverage.transform(
            "mean"
        )
        centered_metric = data[metric] - data.groupby("held_out_domain")[metric].transform("mean")
        observed = float(spearmanr(centered_aoa, centered_metric).statistic)

        permuted = []
        grouped_indices = [g.index.to_numpy() for _, g in data.groupby("held_out_domain")]
        aoa_values = data.aoa_coverage.copy()
        for _ in range(PERMUTATION_REPETITIONS):
            shuffled = aoa_values.copy()
            for indices in grouped_indices:
                shuffled.loc[indices] = rng.permutation(aoa_values.loc[indices].to_numpy())
            centered = shuffled - shuffled.groupby(data.held_out_domain).transform("mean")
            permuted.append(float(spearmanr(centered, centered_metric).statistic))
        permuted_array = np.asarray(permuted)
        permutation_p = float(
            (np.sum(np.abs(permuted_array) >= abs(observed)) + 1)
            / (len(permuted_array) + 1)
        )

        cluster_boot = []
        domains = data.held_out_domain.unique()
        centered_frame = pd.DataFrame(
            {
                "domain": data.held_out_domain,
                "aoa": centered_aoa,
                "metric": centered_metric,
            }
        )
        by_domain = {
            domain: centered_frame[centered_frame.domain == domain] for domain in domains
        }
        for _ in range(PERMUTATION_REPETITIONS):
            sampled_domains = rng.choice(domains, len(domains), replace=True)
            sample = pd.concat([by_domain[domain] for domain in sampled_domains], ignore_index=True)
            cluster_boot.append(float(spearmanr(sample.aoa, sample.metric).statistic))
        cluster_boot = np.asarray(cluster_boot)
        rows.append(
            {
                "metric": metric,
                "naive_spearman_rho": full_rho,
                "within_domain_spearman_rho": observed,
                "within_domain_permutation_p": permutation_p,
                "cluster_bootstrap_ci95_low": float(np.nanquantile(cluster_boot, 0.025)),
                "cluster_bootstrap_ci95_high": float(np.nanquantile(cluster_boot, 0.975)),
                "n_domains": len(domains),
                "n_feature_set_domain_pairs": len(data),
                "interpretation": "exploratory clustered analysis; absence of detection is not evidence of independence",
            }
        )
    return pd.DataFrame(rows)


def control_estimand_table() -> pd.DataFrame:
    pooled = pd.read_csv(POOLED_METRICS)
    primary = pooled[pooled.feature_set == "conventional_alphaearth_embeddings"].iloc[0]
    robustness = pd.read_csv(ROBUSTNESS_SUMMARY)
    rows = [
        {
            "analysis": "Primary broad-background controls",
            "estimand": "Corridor-wide ranking of mapped slope failures against the valid 250 m study-area background",
            "n_positive": int(primary.n_positive),
            "n_control": int(primary.n_negative),
            "roc_auc": primary.roc_auc,
            "pr_auc": primary.pr_auc,
            "brier": primary.brier,
            "role": "Mapping estimand used to fit the corridor-wide susceptibility surface",
        }
    ]
    for _, row in robustness[
        robustness.experiment.str.startswith("accessibility_matched_controls_repeat")
    ].iterrows():
        repeat = int(row.experiment.rsplit("_", 1)[-1])
        rows.append(
            {
                "analysis": f"Terrain/accessibility-matched controls, repeat {repeat}",
                "estimand": "Conditional ranking among locations similar in terrain, access, drainage, fault proximity, and rainfall",
                "n_positive": int(row.n_positive),
                "n_control": int(row.n_negative),
                "roc_auc": row.roc_auc,
                "pr_auc": row.pr_auc,
                "brier": row.brier,
                "role": "Conservative contrast and inventory/accessibility-bias stress test",
            }
        )
    return pd.DataFrame(rows)


def macro_domain_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_ORDER:
        data = metrics[metrics.feature_set == feature_set]
        rows.append(
            {
                "feature_set": feature_set,
                "feature_set_name": FEATURE_LABEL[feature_set],
                "macro_mean_roc_auc": data.roc_auc.mean(),
                "macro_mean_pr_auc": data.pr_auc.mean(),
                "macro_mean_brier": data.brier.mean(),
                "minimum_domain_roc_auc": data.roc_auc.min(),
                "maximum_domain_roc_auc": data.roc_auc.max(),
            }
        )
    table = pd.DataFrame(rows)
    fusion_conv = paired[paired.comparison == "Fusion - Conventional"]
    for metric in ["roc_auc", "pr_auc", "brier"]:
        sub = fusion_conv[fusion_conv.metric == metric]
        table.loc[
            table.feature_set == "conventional_alphaearth_embeddings",
            f"domains_fusion_higher_{metric}",
        ] = int((sub.difference_a_minus_b > 0).sum())
    return table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS)
    metrics = pd.read_csv(LODO_METRICS)
    intervals = pd.read_csv(LODO_INTERVALS)
    aoa_summary = pd.read_csv(AOA_SUMMARY)

    paired = paired_block_bootstrap(predictions)
    comparative = comparative_lodo_table(metrics, intervals, paired)
    macro = macro_domain_summary(metrics, paired)
    sensitivity_grid = aoa_sensitivity_grid()
    sensitivity_summary, sensitivity_winners = summarise_aoa_sensitivity(sensitivity_grid)
    association = domain_aware_association(aoa_summary)
    controls = control_estimand_table()

    paired.to_csv(OUT_DIR / "paired_lodo_feature_set_differences.csv", index=False)
    comparative.to_csv(OUT_DIR / "table_6_comparative_lodo_all_feature_sets.csv", index=False)
    macro.to_csv(OUT_DIR / "lodo_macro_domain_summary.csv", index=False)
    sensitivity_grid.to_csv(OUT_DIR / "aoa_sensitivity_grid.csv", index=False)
    sensitivity_summary.to_csv(OUT_DIR / "aoa_sensitivity_summary.csv", index=False)
    sensitivity_winners.to_csv(OUT_DIR / "aoa_sensitivity_feature_set_winners.csv", index=False)
    association.to_csv(OUT_DIR / "aoa_performance_domain_aware_association.csv", index=False)
    controls.to_csv(OUT_DIR / "control_sampling_estimands_and_metrics.csv", index=False)

    manifest = {
        "analysis": "manuscript_v6_audit_resolutions",
        "input_lodo_predictions": str(PREDICTIONS),
        "paired_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "paired_bootstrap_unit": "1-degree spatial block within each held-out domain",
        "aoa_sensitivity_dimensions": DIMENSIONS,
        "aoa_sensitivity_threshold_quantiles": QUANTILES,
        "aoa_design": "preprocessing and whitened PCA fitted only on source-domain training samples after 20 km target buffer",
        "association_test": "within-domain centering plus feature-set-label permutation within each of five domains",
        "association_repetitions": PERMUTATION_REPETITIONS,
        "control_estimands": {
            "primary": "corridor-wide broad-background mapping estimand",
            "matched": "conditional discrimination stress test among environmentally similar locations",
        },
        "random_seed": SEED,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nComparative LODO macro summary\n", macro.to_string(index=False), flush=True)
    print("\nDomain-aware AoA association\n", association.to_string(index=False), flush=True)
    print("\nAoA sensitivity summary\n", sensitivity_summary.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
