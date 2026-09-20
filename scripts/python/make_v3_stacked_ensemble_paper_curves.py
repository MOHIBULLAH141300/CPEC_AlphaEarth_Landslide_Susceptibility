"""Create paper-facing curves for the Spatial-CV Stacked Ensemble only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(r"D:\DING PROJECT")
PRED = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv" / "v3_model_family_spatial_cv_predictions.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "stacked_ensemble_paper_outputs"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "stacked_ensemble_paper_curves_2018.md"

STACKED_SETS = [
    ("conventional_v3_alphaearth_embeddings", "Conventional + AlphaEarth Embeddings"),
    ("conventional_v3", "Conventional"),
    ("alphaearth_embeddings", "AlphaEarth Embeddings"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED)
    rows = []
    roc_rows = []
    pr_rows = []
    cal_rows = []

    plt.figure(figsize=(7.4, 6.2))
    for fs, label in STACKED_SETS:
        d = pred[(pred["feature_set"] == fs) & (pred["model"] == "stacked_l2_logistic")].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, roc_thresholds = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.4, label=f"{label} (AUC={roc_auc:.3f})")
        for x, yy, th in zip(fpr, tpr, roc_thresholds):
            roc_rows.append({"feature_set": label, "fpr": x, "tpr": yy, "threshold": th})
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Spatial-CV Stacked Ensemble ROC Curves")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_roc_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_roc_curves_2018.pdf")
    plt.close()

    plt.figure(figsize=(7.4, 6.2))
    for fs, label in STACKED_SETS:
        d = pred[(pred["feature_set"] == fs) & (pred["model"] == "stacked_l2_logistic")].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        precision, recall, pr_thresholds = precision_recall_curve(y, p)
        pr_auc = average_precision_score(y, p)
        plt.plot(recall, precision, linewidth=2.4, label=f"{label} (AP={pr_auc:.3f})")
        for x, yy, th in zip(recall, precision, list(pr_thresholds) + [float("nan")]):
            pr_rows.append({"feature_set": label, "recall": x, "precision": yy, "threshold": th})
    prevalence = pred[(pred["model"] == "stacked_l2_logistic") & (pred["feature_set"] == "conventional_v3")][
        "label"
    ].astype(int).mean()
    plt.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Spatial-CV Stacked Ensemble Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_pr_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_pr_curves_2018.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 6.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Perfect calibration")
    for fs, label in STACKED_SETS:
        d = pred[(pred["feature_set"] == fs) & (pred["model"] == "stacked_l2_logistic")].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2.2, label=f"{label} (Brier={brier:.3f})")
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
            cal_rows.append(
                {
                    "feature_set": label,
                    "bin": i,
                    "mean_predicted_probability": mp,
                    "fraction_observed_landslide": fp,
                    "brier": brier,
                }
            )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed landslide fraction")
    plt.title("Spatial-CV Stacked Ensemble Calibration")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_calibration_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_stacked_ensemble_calibration_2018.pdf")
    plt.close()

    for fs, label in STACKED_SETS:
        d = pred[(pred["feature_set"] == fs) & (pred["model"] == "stacked_l2_logistic")].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        threshold = d["threshold"].astype(float).mean()
        yhat = (p >= threshold).astype(int)
        rows.append(
            {
                "feature_set": label,
                "model": "Spatial-CV Stacked Ensemble",
                "roc_auc": roc_auc_score(y, p),
                "pr_auc_ap": average_precision_score(y, p),
                "balanced_accuracy": balanced_accuracy_score(y, yhat),
                "f1": f1_score(y, yhat),
                "brier": brier_score_loss(y, p),
                "n": len(d),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("pr_auc_ap", ascending=False)
    metrics.to_csv(OUT_DIR / "stacked_ensemble_metrics_pooled.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(OUT_DIR / "stacked_ensemble_roc_coordinates.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(OUT_DIR / "stacked_ensemble_pr_coordinates.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "stacked_ensemble_calibration_curve.csv", index=False)

    lines = [
        "# Spatial-CV Stacked Ensemble Paper Curves",
        "",
        "These outputs present the same Spatial-CV Stacked Ensemble model type for all three official variable sets.",
        "",
        "| Feature set | ROC-AUC | PR-AUC/AP | Balanced accuracy | F1 | Brier | N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['feature_set']} | {row['roc_auc']:.3f} | {row['pr_auc_ap']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['f1']:.3f} | {row['brier']:.3f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Saved Figures",
            "",
            f"- ROC curves: `{FIG_DIR / 'figure_stacked_ensemble_roc_curves_2018.png'}`",
            f"- PR curves: `{FIG_DIR / 'figure_stacked_ensemble_pr_curves_2018.png'}`",
            f"- Calibration curves: `{FIG_DIR / 'figure_stacked_ensemble_calibration_2018.png'}`",
            "",
            "## Saved Tables",
            "",
            f"- Metrics: `{OUT_DIR / 'stacked_ensemble_metrics_pooled.csv'}`",
            f"- ROC coordinates: `{OUT_DIR / 'stacked_ensemble_roc_coordinates.csv'}`",
            f"- PR coordinates: `{OUT_DIR / 'stacked_ensemble_pr_coordinates.csv'}`",
            f"- Calibration coordinates: `{OUT_DIR / 'stacked_ensemble_calibration_curve.csv'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(REPORT)


if __name__ == "__main__":
    main()
