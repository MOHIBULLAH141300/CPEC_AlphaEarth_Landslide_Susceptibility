"""Create reviewer-ready evaluation metric tables for 2018 baseline models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\DING PROJECT")
SUMMARY = PROJECT_ROOT / "03_models" / "baseline_2018_spatial_cv" / "spatial_cv_summary.csv"
FOLDS = PROJECT_ROOT / "03_models" / "baseline_2018_spatial_cv" / "spatial_cv_fold_metrics.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "baseline_2018_metrics"
REPORT = PROJECT_ROOT / "05_reports" / "baseline_2018_evaluation_metrics.md"


KEY_ROWS = [
    ("conventional_reduced", "xgboost"),
    ("alphaearth_only", "xgboost"),
    ("fused_reduced", "xgboost"),
    ("fused_full", "xgboost"),
]


def fmt(mean, std):
    return f"{float(mean):.3f} +/- {float(std):.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    folds = pd.read_csv(FOLDS)

    key = pd.DataFrame(KEY_ROWS, columns=["feature_set", "model"])
    key_summary = key.merge(summary, on=["feature_set", "model"], how="left")
    key_summary.to_csv(OUT_DIR / "key_model_metrics_summary.csv", index=False)

    all_ranked = summary.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
    all_ranked.to_csv(OUT_DIR / "all_model_metrics_ranked.csv", index=False)

    key_fold_metrics = key.merge(folds, on=["feature_set", "model"], how="left")
    key_fold_metrics.to_csv(OUT_DIR / "key_model_fold_metrics.csv", index=False)

    lines = [
        "# 2018 Baseline Evaluation Metrics",
        "",
        "Metrics are computed from 5-fold spatial block cross-validation.",
        "",
        "## Key Model Comparison",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC | Balanced accuracy | F1 | Brier score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "conventional_reduced": "Conventional reduced",
        "alphaearth_only": "AlphaEarth Embeddings",
        "fused_reduced": "Fused reduced",
        "fused_full": "Fused full",
    }
    for _, row in key_summary.iterrows():
        lines.append(
            f"| {labels.get(row['feature_set'], row['feature_set'])} | {row['model']} | "
            f"{fmt(row['roc_auc_mean'], row['roc_auc_std'])} | "
            f"{fmt(row['pr_auc_mean'], row['pr_auc_std'])} | "
            f"{fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} | "
            f"{fmt(row['f1_mean'], row['f1_std'])} | "
            f"{fmt(row['brier_mean'], row['brier_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Metric Meanings",
            "",
            "- ROC-AUC: discrimination across all probability thresholds.",
            "- PR-AUC: precision-recall performance, especially useful when class balance or event rarity matters.",
            "- Balanced accuracy: average of sensitivity and specificity at the selected threshold.",
            "- F1: harmonic mean of precision and recall at the selected threshold.",
            "- Brier score: probability calibration/error; lower is better.",
            "",
            "## Saved Tables",
            "",
            f"- Key summary: `{OUT_DIR / 'key_model_metrics_summary.csv'}`",
            f"- All ranked models: `{OUT_DIR / 'all_model_metrics_ranked.csv'}`",
            f"- Fold-level key metrics: `{OUT_DIR / 'key_model_fold_metrics.csv'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

