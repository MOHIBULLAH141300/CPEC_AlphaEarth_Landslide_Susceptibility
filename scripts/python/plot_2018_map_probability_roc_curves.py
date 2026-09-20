"""Generate ROC curves from full-map probability surfaces sampled at labelled points.

These are apparent/map-fit ROC curves. They are useful for checking probability
surfaces, but the reviewer-facing performance estimate remains spatial-CV
out-of-fold ROC/PR curves.
"""

from __future__ import annotations

from pathlib import Path

import ee
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT = "ee-mohibullah141300"
SAMPLES = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
OUT_DIR = Path(r"D:\DING PROJECT\03_models\map_probability_roc_2018")
REPORT = Path(r"D:\DING PROJECT\05_reports\map_probability_roc_2018.md")

MAP_ASSETS = {
    "Conventional reduced map": {
        "asset": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_conventional_reduced_probability_250m",
        "band": "prob_conventional_reduced",
    },
    "AlphaEarth Embeddings map": {
        "asset": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_alphaearth_only_probability_250m",
        "band": "prob_alphaearth_only",
    },
    "Fused reduced map": {
        "asset": "projects/ee-mohibullah141300/assets/cpec_lsm_clean/maps/cpec_2018_fused_reduced_probability_250m",
        "band": "prob_fused_reduced",
    },
}


def sample_asset(label: str, asset: str, band: str) -> pd.DataFrame:
    samples = ee.FeatureCollection(SAMPLES)
    image = ee.Image(asset).select([band]).rename(["probability"])
    fc = image.sampleRegions(
        collection=samples,
        properties=["label", "spatial_fold_5", "use_role", "hazard_type"],
        scale=250,
        geometries=False,
        tileScale=4,
    )
    size = fc.size().getInfo()
    features = fc.toList(size).getInfo()
    rows = []
    for feature in features:
        props = feature["properties"]
        rows.append(
            {
                "map_label": label,
                "asset": asset,
                "label": int(float(props["label"])),
                "probability": float(props["probability"]),
                "spatial_fold_5": props.get("spatial_fold_5", ""),
                "use_role": props.get("use_role", ""),
                "hazard_type": props.get("hazard_type", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ee.Initialize(project=PROJECT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled = pd.concat(
        [sample_asset(label, info["asset"], info["band"]) for label, info in MAP_ASSETS.items()],
        ignore_index=True,
    )
    sampled.to_csv(OUT_DIR / "map_probability_samples_2018.csv", index=False)

    summary_rows = []
    curve_rows = []
    plt.figure(figsize=(7.2, 6.2))
    for label, d in sampled.groupby("map_label", sort=False):
        y = d["label"].to_numpy()
        p = d["probability"].to_numpy()
        fpr, tpr, thresholds = roc_curve(y, p)
        auc_value = roc_auc_score(y, p)
        ap_value = average_precision_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{label} (AUC={auc_value:.3f})")
        summary_rows.append({"map_label": label, "roc_auc": auc_value, "pr_auc_ap": ap_value, "n": len(d)})
        for x, yy, th in zip(fpr, tpr, thresholds):
            curve_rows.append({"map_label": label, "curve": "roc", "x": x, "y": yy, "threshold": th})
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("2018 Probability Map ROC Curves")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves_2018_probability_maps.png", dpi=300)
    plt.savefig(OUT_DIR / "roc_curves_2018_probability_maps.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 6.2))
    for label, d in sampled.groupby("map_label", sort=False):
        y = d["label"].to_numpy()
        p = d["probability"].to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap_value = average_precision_score(y, p)
        plt.plot(recall, precision, linewidth=2.2, label=f"{label} (AP={ap_value:.3f})")
        padded_thresholds = list(thresholds) + [float("nan")]
        for x, yy, th in zip(recall, precision, padded_thresholds):
            curve_rows.append({"map_label": label, "curve": "precision_recall", "x": x, "y": yy, "threshold": th})
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2018 Probability Map Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves_2018_probability_maps.png", dpi=300)
    plt.savefig(OUT_DIR / "pr_curves_2018_probability_maps.pdf")
    plt.close()

    summary = pd.DataFrame(summary_rows).sort_values("pr_auc_ap", ascending=False)
    summary.to_csv(OUT_DIR / "map_probability_auc_summary_2018.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "map_probability_curve_coordinates_2018.csv", index=False)

    lines = [
        "# 2018 Probability Map ROC Curves",
        "",
        "These curves are generated by sampling the completed probability-map rasters at the labelled 2018 sample points.",
        "",
        "**Important:** these are apparent/map-fit curves, not spatial-CV out-of-fold curves. Use them for checking the probability maps. Use the spatial-CV ROC curves for reviewer-facing model performance.",
        "",
        "## Files",
        "",
        f"- ROC PNG: `{OUT_DIR / 'roc_curves_2018_probability_maps.png'}`",
        f"- ROC PDF: `{OUT_DIR / 'roc_curves_2018_probability_maps.pdf'}`",
        f"- PR PNG: `{OUT_DIR / 'pr_curves_2018_probability_maps.png'}`",
        f"- PR PDF: `{OUT_DIR / 'pr_curves_2018_probability_maps.pdf'}`",
        f"- Sampled probabilities: `{OUT_DIR / 'map_probability_samples_2018.csv'}`",
        f"- AUC summary: `{OUT_DIR / 'map_probability_auc_summary_2018.csv'}`",
        "",
        "## AUC Summary",
        "",
        "| Probability map | ROC-AUC | PR-AUC/AP | N |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['map_label']} | {row['roc_auc']:.3f} | {row['pr_auc_ap']:.3f} | {int(row['n'])} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

