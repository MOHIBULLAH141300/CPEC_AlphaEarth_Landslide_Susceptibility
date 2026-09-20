"""Prepare traceable descriptive summaries used to expand the v3.1 Results section."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"D:\DING PROJECT")
OUT = ROOT / "FINAL PAPER" / "Manuscript_single_file" / "v3_1_2026-07-17" / "results_expansion_summary.json"
MODEL = ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"
PAIR = ROOT / "03_models" / "manuscript_v3_spatial_block_comparisons"
ROBUST = ROOT / "03_models" / "manuscript_v3_robustness_experiments"
LODO = ROOT / "03_models" / "manuscript_v3_leave_one_domain_out"
AOA = ROOT / "03_models" / "manuscript_v3_harmonised_aoa"
SHAP = ROOT / "03_models" / "manuscript_v3_xgboost_treeshap"
MAPS = ROOT / "04_maps" / "manuscript_v3_baseline_susceptibility_scores_250m"
ROADS = ROOT / "04_maps" / "manuscript_v3_road_exposure"


def frame_records(frame: pd.DataFrame, digits: int = 6) -> list[dict]:
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["number"]).columns:
        rounded[column] = rounded[column].round(digits)
    return rounded.to_dict(orient="records")


def histogram_quantile(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    target = q * hist.sum()
    index = int(np.searchsorted(np.cumsum(hist), target, side="left"))
    index = min(max(index, 0), len(hist) - 1)
    return float((edges[index] + edges[index + 1]) / 2)


def difference_summary() -> dict:
    conventional = MAPS / "cpec_baseline_conventional_stacked_susceptibility_score_250m.tif"
    fused = MAPS / "cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif"
    edges = np.linspace(-1.0, 1.0, 20001)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    count = positive = negative = near_zero = abs_ge_010 = abs_ge_020 = 0
    total = total_sq = 0.0
    minimum, maximum = np.inf, -np.inf

    with rasterio.open(conventional) as src_c, rasterio.open(fused) as src_f:
        for _, window in src_c.block_windows(1):
            c = src_c.read(1, window=window, masked=True)
            f = src_f.read(1, window=window, masked=True)
            valid = ~(np.ma.getmaskarray(c) | np.ma.getmaskarray(f))
            if not valid.any():
                continue
            diff = np.asarray(f)[valid].astype(np.float64) - np.asarray(c)[valid].astype(np.float64)
            count += diff.size
            total += float(diff.sum())
            total_sq += float(np.square(diff).sum())
            positive += int(np.count_nonzero(diff > 1e-6))
            negative += int(np.count_nonzero(diff < -1e-6))
            near_zero += int(np.count_nonzero(np.abs(diff) <= 1e-6))
            abs_ge_010 += int(np.count_nonzero(np.abs(diff) >= 0.10))
            abs_ge_020 += int(np.count_nonzero(np.abs(diff) >= 0.20))
            minimum = min(minimum, float(diff.min()))
            maximum = max(maximum, float(diff.max()))
            hist += np.histogram(diff, bins=edges)[0]

    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return {
        "valid_pixels": count,
        "mean": mean,
        "standard_deviation": variance**0.5,
        "minimum": minimum,
        "p10": histogram_quantile(hist, edges, 0.10),
        "median": histogram_quantile(hist, edges, 0.50),
        "p90": histogram_quantile(hist, edges, 0.90),
        "maximum": maximum,
        "positive_percent": 100 * positive / count,
        "negative_percent": 100 * negative / count,
        "near_zero_percent": 100 * near_zero / count,
        "absolute_difference_ge_0_10_percent": 100 * abs_ge_010 / count,
        "absolute_difference_ge_0_20_percent": 100 * abs_ge_020 / count,
    }


def main() -> None:
    pooled = pd.read_csv(MODEL / "nested_spatial_cv_pooled_metrics.csv")
    folds = pd.read_csv(MODEL / "nested_spatial_cv_fold_metrics.csv")
    fold_ranges = folds.groupby("feature_set_name").agg(
        roc_auc_min=("roc_auc", "min"),
        roc_auc_max=("roc_auc", "max"),
        pr_auc_min=("pr_auc", "min"),
        pr_auc_max=("pr_auc", "max"),
        brier_min=("brier", "min"),
        brier_max=("brier", "max"),
        n_test_min=("n_test", "min"),
        n_test_max=("n_test", "max"),
    ).reset_index()

    meta = pd.read_csv(MODEL / "nested_spatial_cv_meta_coefficients.csv")
    meta_summary = meta.groupby(["feature_set_name", "base_model_name"]).meta_coefficient.agg(["mean", "std", "min", "max"]).reset_index()

    pairwise = pd.read_csv(PAIR / "paired_spatial_block_model_comparisons.csv")
    balance = pd.read_csv(ROBUST / "control_balance_diagnostics.csv")
    balance_summary = balance.groupby("experiment").agg(
        maximum_absolute_smd=("absolute_standardised_mean_difference", "max"),
        mean_absolute_smd=("absolute_standardised_mean_difference", "mean"),
    ).reset_index()
    robust = pd.read_csv(ROBUST / "robustness_experiment_summary.csv")

    lodo = pd.read_csv(LODO / "leave_one_domain_out_metrics.csv")
    fused_lodo = lodo[lodo.feature_set == "conventional_alphaearth_embeddings"].copy()
    fused_lodo["positive_fraction"] = fused_lodo.n_test_positive / fused_lodo.n_test
    aoa = pd.read_csv(AOA / "harmonised_aoa_domain_summary.csv")
    aoa_correlations = pd.read_csv(AOA / "aoa_transfer_metric_correlations.csv")

    importance = pd.read_csv(SHAP / "xgboost_treeshap_global_importance.csv")
    fused_importance = importance[importance.feature_set == "conventional_alphaearth_embeddings"].nlargest(15, "mean_absolute_treeshap")

    map_qa = pd.read_csv(MAPS / "baseline_susceptibility_score_raster_qa.csv")
    road_summary = pd.read_csv(ROADS / "road_network_exposure_summary.csv")
    route_ranking = pd.read_csv(ROADS / "named_route_exposure_ranking.csv")
    for threshold in ("p80", "p90"):
        road_summary[f"actual_{threshold}_share_percent"] = (
            100 * road_summary[f"actual_length_ge_{threshold}_km"] / road_summary.total_length_km
        )
    road_summary["supported_share_of_p90_percent"] = (
        100 * road_summary.supported_high_p90_length_km / road_summary.actual_length_ge_p90_km
    )
    road_summary["verification_share_of_p90_percent"] = (
        100 * road_summary.verification_priority_p90_length_km / road_summary.actual_length_ge_p90_km
    )

    output = {
        "pooled_metrics": frame_records(pooled),
        "outer_fold_metric_ranges": frame_records(fold_ranges),
        "meta_coefficient_summary": frame_records(meta_summary),
        "pairwise_spatial_block_comparisons": frame_records(pairwise),
        "control_balance_summary": frame_records(balance_summary),
        "robustness_experiments": frame_records(robust),
        "fused_lodo_metrics": frame_records(fused_lodo),
        "harmonised_aoa_domain_summary": frame_records(aoa),
        "aoa_transfer_correlations": frame_records(aoa_correlations),
        "fused_xgboost_top15_treeshap": frame_records(fused_importance),
        "baseline_raster_qa": frame_records(map_qa),
        "fused_minus_conventional_difference": difference_summary(),
        "road_network_summary": frame_records(road_summary),
        "named_route_top10": frame_records(route_ranking.head(10)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
