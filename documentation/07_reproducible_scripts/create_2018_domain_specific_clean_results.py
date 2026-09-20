"""Create clean 2018 V3 results separately for each CPEC transfer domain.

This script packages the validated 2018 V3 outputs into reviewer-readable
domain folders. The domain maps are clipped outputs from the same fixed 2018
models, not separately trained regional models. The transferability evidence
comes from leave-one-domain-out testing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio
from rasterio.mask import mask
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
OUT_ROOT = PACKAGE / "10_domain_specific_2018_results"
SCRIPT_COPY_DIR = PACKAGE / "07_reproducible_scripts"

BOUNDARY = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)
PAK_ADMIN1 = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\gadm41_PAK_shp\gadm41_PAK_1.shp"
)
KASHGAR = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\New Folder\cpecpart.shp"
)

PROB_DIR = PROJECT / "04_maps" / "rasters_2018_probability_stacked_ensemble"
AOA_DIR = PROJECT / "04_maps" / "area_of_applicability_transfer_confidence_250m"
REL_DIR = PROJECT / "04_maps" / "reliability_aware_susceptibility_2018_250m"
HI_DIR = PROJECT / "04_maps" / "stacked_ensemble_250m_high_impact_outputs"
MODEL_DIR = PROJECT / "03_models" / "v3_admin_transferability_domain_tests"
VIF_DIR = PACKAGE / "06_samples_factors_and_vif"

PUBLIC_DOMAINS = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]

FEATURE_SETS = {
    "conventional": {
        "label": "Conventional",
        "prob": PROB_DIR / "cpec_2018_conventional_stacked_ensemble_probability_250m_no_nodata.tif",
        "confidence": AOA_DIR / "cpec_2018_conventional_transfer_confidence_250m.tif",
        "aoa": AOA_DIR / "cpec_2018_conventional_in_area_of_applicability_250m.tif",
        "reliability": REL_DIR / "cpec_2018_conventional_reliability_weighted_probability_250m.tif",
    },
    "alphaearth_embeddings": {
        "label": "AlphaEarth Embeddings",
        "prob": PROB_DIR / "cpec_2018_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        "confidence": AOA_DIR / "cpec_2018_alphaearth_embeddings_transfer_confidence_250m.tif",
        "aoa": AOA_DIR / "cpec_2018_alphaearth_embeddings_in_area_of_applicability_250m.tif",
        "reliability": REL_DIR / "cpec_2018_alphaearth_embeddings_reliability_weighted_probability_250m.tif",
    },
    "conventional_alphaearth_embeddings": {
        "label": "Conventional + AlphaEarth Embeddings",
        "prob": PROB_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_stacked_ensemble_probability_250m_no_nodata.tif",
        "confidence": AOA_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_transfer_confidence_250m.tif",
        "aoa": AOA_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_in_area_of_applicability_250m.tif",
        "reliability": REL_DIR
        / "cpec_2018_conventional_alphaearth_embeddings_reliability_weighted_probability_250m.tif",
    },
}

FINAL_RASTERS = {
    "final_fused_reliability_aware_score": REL_DIR
    / "cpec_2018_final_fused_reliability_aware_susceptibility_score_250m.tif",
    "final_reliable_high_mask": REL_DIR / "cpec_2018_final_reliable_high_susceptibility_mask_250m.tif",
    "final_reliable_very_high_mask": REL_DIR
    / "cpec_2018_final_reliable_very_high_susceptibility_mask_250m.tif",
    "final_uncertain_high_mask": REL_DIR / "cpec_2018_final_uncertain_high_susceptibility_mask_250m.tif",
    "field_verification_priority_score": REL_DIR
    / "cpec_2018_final_field_verification_priority_score_250m.tif",
    "model_disagreement_std": HI_DIR / "cpec_2018_stacked_probability_disagreement_std_250m.tif",
}


def slug(text: str) -> str:
    out = text.lower()
    out = out.replace("+", "and")
    out = re.sub(r"[^a-z0-9]+", "_", out)
    return out.strip("_")


def clean_float(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if nodata is not None:
        arr = arr.copy()
        arr[arr == nodata] = np.nan
    return arr


def stats_from_array(arr: np.ndarray) -> dict[str, float | int]:
    data = arr[np.isfinite(arr)]
    if data.size == 0:
        return {"valid_pixels": 0}
    q = np.nanpercentile(data, [1, 5, 25, 50, 75, 95, 99])
    return {
        "valid_pixels": int(data.size),
        "min": float(np.nanmin(data)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "near_zero_percent_le_0_001": float(np.mean(data <= 0.001) * 100),
    }


def read_domains(crs: str | dict) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:32642", allow_override=True)
    boundary = boundary.to_crs(crs)
    study_union = boundary.geometry.unary_union

    pak = gpd.read_file(PAK_ADMIN1).to_crs(crs)
    kas = gpd.read_file(KASHGAR).to_crs(crs)
    mapping = {
        "Kashgar (Xinjiang, China)": "Kashgar (Xinjiang, China)",
        "Gilgit-Baltistan": "Gilgit-Baltistan",
        "Balochistan": "Balochistan",
        "Khyber-Pakhtunkhwa": "KP-AJK",
        "Federally Administered Tribal Ar": "KP-AJK",
        "Azad Kashmir": "KP-AJK",
        "Punjab": "Punjab-Sindh lowland corridor",
        "Islamabad": "Punjab-Sindh lowland corridor",
        "Sindh": "Punjab-Sindh lowland corridor",
    }
    rows = []
    for _, row in pak.iterrows():
        domain = mapping.get(row["NAME_1"])
        if domain:
            geom = row.geometry.intersection(study_union)
            if not geom.is_empty:
                rows.append({"domain": domain, "source_unit": row["NAME_1"], "geometry": geom})
    for geom in kas.geometry:
        geom = geom.intersection(study_union)
        if not geom.is_empty:
            rows.append(
                {
                    "domain": "Kashgar (Xinjiang, China)",
                    "source_unit": "Kashgar",
                    "geometry": geom,
                }
            )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    dissolved = gdf.dissolve(by="domain", as_index=False)
    return dissolved.loc[dissolved["domain"].isin(PUBLIC_DOMAINS)].copy()


def clip_raster(src_path: Path, geom, out_path: Path, out_nodata: float | int | None = None) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dtype = src.dtypes[0]
        if out_nodata is None:
            out_nodata = 255 if dtype.startswith("uint") else -9999.0
        out, transform = mask(
            src,
            [geom],
            crop=True,
            filled=True,
            nodata=out_nodata,
            all_touched=True,
        )
        profile = src.profile.copy()
        predictor = 2 if not dtype.startswith("uint8") else 1
        profile.update(
            {
                "height": out.shape[1],
                "width": out.shape[2],
                "transform": transform,
                "nodata": out_nodata,
                "compress": "deflate",
                "predictor": predictor,
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
        return clean_float(out[0], out_nodata)


def save_domain_tables(domain: str, table_dir: Path) -> dict[str, str]:
    table_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}

    table_specs = [
        (
            MODEL_DIR / "v3_admin_transferability_leave_one_domain_metrics.csv",
            "domain_leave_one_domain_out_model_metrics.csv",
            "held_out_domain",
        ),
        (
            MODEL_DIR / "v3_admin_transferability_base_learner_metrics.csv",
            "domain_base_learner_transfer_metrics.csv",
            "held_out_domain",
        ),
        (
            MODEL_DIR / "v3_admin_transferability_domain_shift.csv",
            "domain_shift_similarity.csv",
            "held_out_domain",
        ),
        (
            MODEL_DIR / "v3_admin_transferability_domain_sample_balance.csv",
            "domain_sample_balance.csv",
            "domain",
        ),
        (
            AOA_DIR / "cpec_2018_area_of_applicability_by_subdomain.csv",
            "domain_area_of_applicability_summary.csv",
            "domain",
        ),
        (
            REL_DIR / "cpec_2018_reliability_aware_subdomain_area_summary.csv",
            "domain_reliability_aware_area_summary.csv",
            "domain",
        ),
        (
            REL_DIR / "reliability_aware_road_exposure_summary_by_domain_250m.csv",
            "domain_road_exposure_summary.csv",
            "domain",
        ),
    ]
    for src, out_name, col in table_specs:
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if col in df.columns:
            df = df.loc[df[col].eq(domain)].copy()
        out = table_dir / out_name
        df.to_csv(out, index=False)
        copied[out_name] = str(out)

    seg_src = REL_DIR / "reliability_aware_road_segment_exposure_all_250m.csv"
    if seg_src.exists():
        seg = pd.read_csv(seg_src)
        seg = seg.loc[seg["domain"].eq(domain)].copy()
        seg.sort_values("hotspot_rank").head(50).to_csv(
            table_dir / "domain_top50_road_hotspot_segments.csv", index=False
        )
        copied["domain_top50_road_hotspot_segments.csv"] = str(
            table_dir / "domain_top50_road_hotspot_segments.csv"
        )

    for src_name in [
        "v3_final_selected_vif.csv",
        "v3_high_correlation_pairs_abs_ge_0_85.csv",
        "readable_conventional_factor_list.csv",
    ]:
        src = VIF_DIR / src_name
        if src.exists():
            shutil.copy2(src, table_dir / f"global_{src_name}")
            copied[f"global_{src_name}"] = str(table_dir / f"global_{src_name}")
    return copied


def make_domain_curves(domain: str, fig_dir: Path) -> None:
    pred_path = MODEL_DIR / "v3_admin_transferability_held_out_predictions.csv"
    if not pred_path.exists():
        return
    pred = pd.read_csv(pred_path)
    pred = pred.loc[pred["held_out_domain"].eq(domain)].copy()
    if pred.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=220)
    colors = {
        "Conventional": "#2b6cb0",
        "AlphaEarth Embeddings": "#38a169",
        "Conventional + AlphaEarth Embeddings": "#d97706",
    }
    for feature_set, group in pred.groupby("feature_set", sort=False):
        y = group["label"].to_numpy()
        p = group["stacked_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        prob_true, prob_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        color = colors.get(feature_set, None)
        axes[0].plot(fpr, tpr, label=f"{feature_set} (AUC={auc(fpr, tpr):.3f})", color=color)
        axes[1].plot(recall, precision, label=feature_set, color=color)
        axes[2].plot(prob_pred, prob_true, marker="o", label=feature_set, color=color)
    axes[0].plot([0, 1], [0, 1], color="#777777", lw=0.8, ls="--")
    axes[2].plot([0, 1], [0, 1], color="#777777", lw=0.8, ls="--")
    titles = ["ROC", "Precision-Recall", "Calibration"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color="#dddddd", lw=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[1].legend(fontsize=6, loc="lower left")
    axes[2].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Leave-one-domain-out stacked ensemble curves: {domain}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "domain_stacked_ensemble_roc_pr_calibration_curves.png", bbox_inches="tight")
    fig.savefig(fig_dir / "domain_stacked_ensemble_roc_pr_calibration_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_clipped_maps(domain: str, clipped: dict[str, Path], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    prob_keys = [
        ("conventional_probability", "Conventional"),
        ("alphaearth_embeddings_probability", "AlphaEarth Embeddings"),
        ("conventional_alphaearth_embeddings_probability", "Conventional + AlphaEarth Embeddings"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220)
    for ax, (key, label) in zip(axes, prob_keys):
        with rasterio.open(clipped[key]) as src:
            arr = clean_float(src.read(1), src.nodata)
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"2018 stacked susceptibility probability: {domain}", fontsize=13)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Probability")
    fig.savefig(fig_dir / "domain_2018_probability_maps_250m.png", bbox_inches="tight")
    fig.savefig(fig_dir / "domain_2018_probability_maps_250m.pdf", bbox_inches="tight")
    plt.close(fig)

    rel_keys = [
        ("conventional_alphaearth_embeddings_probability", "Raw fused probability"),
        ("conventional_alphaearth_embeddings_transfer_confidence", "Fused transfer confidence"),
        ("final_fused_reliability_aware_score", "Reliability-aware score"),
        ("final_uncertain_high_mask", "Uncertain high mask"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=220)
    for ax, (key, label) in zip(axes, rel_keys):
        with rasterio.open(clipped[key]) as src:
            arr = clean_float(src.read(1), src.nodata)
        cmap = "magma" if "mask" not in key else "YlOrRd"
        im = ax.imshow(arr, vmin=0, vmax=1, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"2018 reliability and uncertainty outputs: {domain}", fontsize=13)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    cbar.set_label("Value")
    fig.savefig(fig_dir / "domain_2018_reliability_uncertainty_maps_250m.png", bbox_inches="tight")
    fig.savefig(fig_dir / "domain_2018_reliability_uncertainty_maps_250m.pdf", bbox_inches="tight")
    plt.close(fig)


def write_domain_report(domain: str, domain_dir: Path, stats_df: pd.DataFrame) -> None:
    metrics_path = domain_dir / "03_tables" / "domain_leave_one_domain_out_model_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    best_line = "No domain metrics table was available."
    if not metrics.empty:
        rows = []
        for _, r in metrics.iterrows():
            rows.append(
                f"- {r['feature_set']}: ROC-AUC {r['roc_auc']:.3f}, "
                f"PR-AUC {r['pr_auc_ap']:.3f}, F1 {r['f1']:.3f}, "
                f"Brier {r['brier']:.3f}"
            )
        best_line = "\n".join(rows)

    display_cols = [
        "feature_set",
        "product",
        "valid_pixels",
        "mean",
        "median",
        "p95",
        "near_zero_percent_le_0_001",
    ]
    display = stats_df.loc[:, [c for c in display_cols if c in stats_df.columns]].copy()
    for col in ["mean", "median", "p95", "near_zero_percent_le_0_001"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    stat_table = dataframe_to_markdown(display)

    report = f"""# 2018 V3 Domain Results: {domain}

## Interpretation rule

The probability rasters in `01_probability_maps_250m` are the main stacked
ensemble susceptibility probability maps. The reliability-weighted rasters in
`02_transfer_confidence_reliability_250m` are secondary confidence-adjusted
products. They can be close to zero where the Area of Applicability / transfer
confidence is low, especially for the Conventional feature set.

## Module status for this domain

| Module | Status | Domain-specific output |
|---|---:|---|
| Official study area | Complete | Domain polygon is clipped to the official CPEC boundary. |
| 2018 corrected factor database | Complete | Same validated CPEC 2018 V3 factor database; domain sample balance table included. |
| Multicollinearity gate | Complete | Global V3 VIF/correlation results included for transparent factor selection. |
| Model family | Complete | Leave-one-domain-out stacked and base learner metrics included. |
| Feature-set comparison | Complete | Conventional, AlphaEarth Embeddings, and Conventional + AlphaEarth Embeddings probability maps included. |
| Explainability | Complete globally | SHAP outputs remain global; domain probability/transfer outputs are packaged here. |
| Uncertainty and exposure | Complete | Domain AOA, reliability-aware area, uncertainty masks, and road exposure tables included. |
| Annual dynamic extension | Pending by year | Annual outputs are stored separately under the annual 2017-2024 workspace. |

## Leave-one-domain-out performance

{best_line}

## Probability and reliability statistics

{stat_table}

## Reviewer note

These are domain-specific results from a single, validated CPEC-wide 2018 V3
model workflow. Separate regional retraining is intentionally avoided here
because the goal is to test spatial transferability and keep feature-set
comparisons consistent across political/physiographic subdomains.
"""
    (domain_dir / f"README_{slug(domain)}_2018_v3_domain_results.md").write_text(
        report, encoding="utf-8"
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    widths = [
        max(len(str(header)), *(len(str(value)) for value in text_df[header].tolist()))
        for header in headers
    ]
    header_line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = []
    for _, row in text_df.iterrows():
        rows.append("| " + " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, widths)) + " |")
    return "\n".join([header_line, sep_line, *rows])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(FEATURE_SETS["conventional"]["prob"]) as ref:
        domains = read_domains(ref.crs)
    domains.to_file(OUT_ROOT / "cpec_2018_v3_transfer_domains.gpkg", driver="GPKG")

    master_rows = []
    for domain in PUBLIC_DOMAINS:
        row = domains.loc[domains["domain"].eq(domain)].iloc[0]
        geom = row.geometry
        domain_slug = slug(domain)
        domain_dir = OUT_ROOT / domain_slug
        map_dir = domain_dir / "01_probability_maps_250m"
        rel_dir = domain_dir / "02_transfer_confidence_reliability_250m"
        table_dir = domain_dir / "03_tables"
        fig_dir = domain_dir / "04_figures"
        raw_ref_dir = domain_dir / "05_background_raw_input_references"
        for d in [map_dir, rel_dir, table_dir, fig_dir, raw_ref_dir]:
            d.mkdir(parents=True, exist_ok=True)

        clipped_paths: dict[str, Path] = {}
        stats_rows = []

        for key, cfg in FEATURE_SETS.items():
            label_slug = key
            prob_out = map_dir / f"cpec_2018_{label_slug}_stacked_probability_250m_{domain_slug}.tif"
            conf_out = rel_dir / f"cpec_2018_{label_slug}_transfer_confidence_250m_{domain_slug}.tif"
            aoa_out = rel_dir / f"cpec_2018_{label_slug}_in_area_of_applicability_250m_{domain_slug}.tif"
            rel_out = rel_dir / f"cpec_2018_{label_slug}_reliability_weighted_probability_250m_{domain_slug}.tif"

            prob = clip_raster(cfg["prob"], geom, prob_out)
            conf = clip_raster(cfg["confidence"], geom, conf_out)
            aoa = clip_raster(cfg["aoa"], geom, aoa_out)
            rel = clip_raster(cfg["reliability"], geom, rel_out)

            clipped_paths[f"{key}_probability"] = prob_out
            clipped_paths[f"{key}_transfer_confidence"] = conf_out
            clipped_paths[f"{key}_aoa"] = aoa_out
            clipped_paths[f"{key}_reliability_weighted"] = rel_out

            for product, arr in [
                ("raw_probability", prob),
                ("transfer_confidence", conf),
                ("in_area_of_applicability", aoa),
                ("reliability_weighted_probability", rel),
            ]:
                s = stats_from_array(arr)
                s.update({"domain": domain, "feature_set": cfg["label"], "product": product})
                stats_rows.append(s)
                master_rows.append(s)

        for key, src in FINAL_RASTERS.items():
            if not src.exists():
                continue
            out = rel_dir / f"cpec_2018_{key}_250m_{domain_slug}.tif"
            arr = clip_raster(src, geom, out)
            clipped_paths[key] = out
            s = stats_from_array(arr)
            s.update({"domain": domain, "feature_set": "Final fused", "product": key})
            stats_rows.append(s)
            master_rows.append(s)

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(table_dir / "domain_probability_confidence_reliability_statistics.csv", index=False)
        save_domain_tables(domain, table_dir)
        make_domain_curves(domain, fig_dir)
        plot_clipped_maps(domain, clipped_paths, fig_dir)

        input_manifest = {
            "interpretation": {
                "probability_maps": "Main susceptibility probabilities.",
                "reliability_weighted_maps": "Secondary confidence-adjusted products; low values can reflect low transfer confidence.",
                "domain_training": "No separate regional retraining; domain evidence is leave-one-domain-out transferability.",
            },
            "source_rasters": {
                **{f"{k}_{item}": str(v[item]) for k, v in FEATURE_SETS.items() for item in v},
                **{k: str(v) for k, v in FINAL_RASTERS.items()},
            },
        }
        (raw_ref_dir / "raw_input_reference_manifest.json").write_text(
            json.dumps(input_manifest, indent=2), encoding="utf-8"
        )
        write_domain_report(domain, domain_dir, stats_df)

    master = pd.DataFrame(master_rows)
    master.to_csv(OUT_ROOT / "cpec_2018_v3_domain_probability_confidence_reliability_master_statistics.csv", index=False)

    root_report = f"""# CPEC 2018 V3 Domain-Specific Results

These folders package the finalized 2018 V3 workflow by CPEC transfer domain.
Each domain contains raw stacked probability maps, transfer confidence, AOA,
reliability-aware products, leave-one-domain-out metrics, road exposure tables,
and publication-ready figures.

Important: the domain probability maps are clipped from the same validated
CPEC-wide 2018 model. They are not separately trained regional models. This is
intentional because the paper tests spatial transferability and feature-set
added value across subdomains.

Domains:
{chr(10).join(f'- `{slug(d)}`: {d}' for d in PUBLIC_DOMAINS)}
"""
    (OUT_ROOT / "README_DOMAIN_RESULTS_START_HERE.md").write_text(root_report, encoding="utf-8")
    shutil.copy2(Path(__file__), SCRIPT_COPY_DIR / Path(__file__).name)


if __name__ == "__main__":
    main()
