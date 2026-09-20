"""Plot ROC and precision-recall curves for 2018 spatial-CV predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_ROOT = Path(r"D:\DING PROJECT")
PRED = PROJECT_ROOT / "03_models" / "baseline_2018_spatial_cv" / "spatial_cv_predictions.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "baseline_2018_curves"
REPORT = PROJECT_ROOT / "05_reports" / "baseline_2018_curve_outputs.md"


KEEP = [
    ("conventional_reduced", "xgboost", "Conventional reduced"),
    ("alphaearth_only", "xgboost", "AlphaEarth Embeddings"),
    ("fused_reduced", "xgboost", "Fused reduced"),
    ("fused_full", "xgboost", "Fused full"),
]


def plot_roc(df: pd.DataFrame, selected: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(7.2, 6.2))
    for feature_set, model, label in selected:
        d = df[(df["feature_set"] == feature_set) & (df["model"] == model)].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{label} (AUC={roc_auc:.3f})")
        rows.extend(
            {
                "feature_set": feature_set,
                "model": model,
                "curve": "roc",
                "x": x,
                "y": yy,
                "threshold": th,
                "auc": roc_auc,
            }
            for x, yy, th in zip(fpr, tpr, thresholds)
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("2018 Spatial-CV ROC Curves")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves_2018_spatial_cv.png", dpi=300)
    plt.savefig(OUT_DIR / "roc_curves_2018_spatial_cv.pdf")
    plt.close()
    return pd.DataFrame(rows)


def plot_pr(df: pd.DataFrame, selected: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(7.2, 6.2))
    for feature_set, model, label in selected:
        d = df[(df["feature_set"] == feature_set) & (df["model"] == model)].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(recall, precision, linewidth=2.2, label=f"{label} (AP={ap:.3f})")
        padded_thresholds = list(thresholds) + [float("nan")]
        rows.extend(
            {
                "feature_set": feature_set,
                "model": model,
                "curve": "precision_recall",
                "x": r,
                "y": pr,
                "threshold": th,
                "average_precision": ap,
                "auc_trapezoid": auc(recall, precision),
            }
            for r, pr, th in zip(recall, precision, padded_thresholds)
        )
    prevalence = df[df["model"] == "xgboost"]["label"].astype(int).mean()
    plt.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2018 Spatial-CV Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves_2018_spatial_cv.png", dpi=300)
    plt.savefig(OUT_DIR / "pr_curves_2018_spatial_cv.pdf")
    plt.close()
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED)
    roc_data = plot_roc(df, KEEP)
    pr_data = plot_pr(df, KEEP)
    curve_data = pd.concat([roc_data, pr_data], ignore_index=True)
    curve_data.to_csv(OUT_DIR / "roc_pr_curve_coordinates_2018_spatial_cv.csv", index=False)

    summary_rows = []
    for feature_set, model, label in KEEP:
        d = df[(df["feature_set"] == feature_set) & (df["model"] == model)].copy()
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        summary_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "label": label,
                "roc_auc": roc_auc_score(y, p),
                "average_precision_pr_auc": average_precision_score(y, p),
                "n": len(d),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("average_precision_pr_auc", ascending=False)
    summary.to_csv(OUT_DIR / "curve_auc_summary_2018_spatial_cv.csv", index=False)

    lines = [
        "# 2018 ROC And Precision-Recall Curves",
        "",
        "Generated from spatial-CV out-of-fold predictions, not from training-set predictions.",
        "",
        "## Files",
        "",
        f"- ROC PNG: `{OUT_DIR / 'roc_curves_2018_spatial_cv.png'}`",
        f"- ROC PDF: `{OUT_DIR / 'roc_curves_2018_spatial_cv.pdf'}`",
        f"- PR PNG: `{OUT_DIR / 'pr_curves_2018_spatial_cv.png'}`",
        f"- PR PDF: `{OUT_DIR / 'pr_curves_2018_spatial_cv.pdf'}`",
        f"- Curve coordinates: `{OUT_DIR / 'roc_pr_curve_coordinates_2018_spatial_cv.csv'}`",
        f"- AUC summary: `{OUT_DIR / 'curve_auc_summary_2018_spatial_cv.csv'}`",
        "",
        "## AUC Summary",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC/AP | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['model']} | {row['roc_auc']:.3f} | "
            f"{row['average_precision_pr_auc']:.3f} | {int(row['n'])} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

