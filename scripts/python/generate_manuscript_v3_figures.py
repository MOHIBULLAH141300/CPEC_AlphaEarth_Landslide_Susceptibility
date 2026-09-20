"""Generate reproducible, journal-ready figures for the manuscript-v3 analysis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
import shapefile
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import PROJECT_ROOT, SEED  # noqa: E402


PAPER_DIR = PROJECT_ROOT / "FINAL PAPER" / "Manuscript_single_file" / "v3_2026-07-17"
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"
ROBUST_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_robustness_experiments"
LODO_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_leave_one_domain_out"
AOA_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_harmonised_aoa"
SHAP_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_xgboost_treeshap"
MAP_DIR = PROJECT_ROOT / "04_maps" / "manuscript_v3_baseline_susceptibility_scores_250m"
ROAD_DIR = PROJECT_ROOT / "04_maps" / "manuscript_v3_road_exposure"
LAYER_DIR = PROJECT_ROOT / "FINAL PAPER" / "00_Study_Area" / "Figure_1_ArcMap_layers"
DEM = LAYER_DIR / "08_DEM_relief_wgs84_downsampled.tif"
BOUNDARY = LAYER_DIR / "01_CPEC_study_boundary.shp"
DOMAINS = LAYER_DIR / "02_CPEC_transfer_subdomains.shp"
ROADS = LAYER_DIR / "03_CPEC_road_network.shp"
KKH = LAYER_DIR / "04_KKH_route.shp"
RIVERS = LAYER_DIR / "05_major_rivers.shp"
FAULTS = LAYER_DIR / "06_active_faults.shp"
INVENTORY = LAYER_DIR / "07_landslide_inventory_points.shp"
COUNTRIES = (
    PROJECT_ROOT
    / "01_clean_data"
    / "cartographic_reference"
    / "natural_earth_110m_admin0"
    / "ne_110m_admin_0_countries.shp"
)

COLORS = {
    "Conventional": "#0072B2",
    "AlphaEarth Embeddings": "#D55E00",
    "Conventional + AlphaEarth Embeddings": "#009E73",
}
DOMAIN_COLORS = {
    "Balochistan": "#C44E52",
    "Gilgit-Baltistan": "#55A868",
    "KP-AJK": "#E6A23C",
    "Kashgar (Xinjiang, China)": "#4C78A8",
    "Punjab-Sindh lowland corridor": "#8172B2",
}
DOMAIN_DISPLAY = {
    "Balochistan": "Balochistan",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "KP-AJK": "KPK-AJK",
    "Kashgar (Xinjiang, China)": "Kashgar",
    "Punjab-Sindh lowland corridor": "Punjab-Sindh",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 500,
        }
    )


def save(fig: plt.Figure, number: int, stem: str) -> None:
    base = FIG_DIR / f"Figure_{number}_{stem}"
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#333333", alpha=0.92),
        zorder=50,
    )


def shapefile_records(path: Path):
    reader = shapefile.Reader(str(path), encoding="latin1")
    for sr in reader.iterShapeRecords():
        yield sr.shape, sr.record.as_dict()


def shape_parts(shp):
    points = np.asarray(shp.points, dtype=float)
    starts = list(shp.parts) + [len(points)]
    for start, end in zip(starts[:-1], starts[1:]):
        if end > start:
            yield points[start:end]


def polygon_fill(ax, shp, facecolor, edgecolor="white", alpha=0.35, lw=0.8, zorder=3):
    for part in shape_parts(shp):
        if len(part) >= 3:
            ax.fill(part[:, 0], part[:, 1], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=lw, zorder=zorder)


def plot_lines(ax, path: Path, color, lw=0.6, alpha=1.0, zorder=6):
    segments = []
    for shp, _ in shapefile_records(path):
        segments.extend([part for part in shape_parts(shp) if len(part) >= 2])
    collection = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=zorder)
    ax.add_collection(collection)
    return collection


def plot_boundary(ax, lw=1.4, color="#111111", zorder=15):
    plot_lines(ax, BOUNDARY, color=color, lw=lw, zorder=zorder)


def plot_domains(ax, alpha=0.28, labels=False):
    for shp, attrs in shapefile_records(DOMAINS):
        domain = attrs["domain"]
        polygon_fill(ax, shp, DOMAIN_COLORS[domain], edgecolor="white", alpha=alpha, lw=0.8, zorder=4)
        if labels:
            points = np.asarray(shp.points)
            if len(points):
                ax.text(
                    np.median(points[:, 0]),
                    np.median(points[:, 1]),
                    DOMAIN_DISPLAY[domain],
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#202020",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
                    zorder=20,
                )


def read_dem():
    with rasterio.open(DEM) as ds:
        data = ds.read(1, masked=True).filled(np.nan)
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
    return data, extent


def terrain_background(ax, data, extent, vmin=0, vmax=8000):
    valid = np.where(np.isfinite(data), data, np.nanmedian(data))
    hill = LightSource(azdeg=315, altdeg=42).hillshade(valid, vert_exag=1.3, dx=1, dy=1)
    image = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        zorder=0,
    )
    ax.imshow(hill, extent=extent, origin="upper", cmap="gray", alpha=0.22, zorder=1)
    return image


def north_arrow(ax, x=0.95, y=0.92):
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.105),
        xycoords=ax.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        arrowprops=dict(facecolor="black", edgecolor="black", width=2, headwidth=7),
        zorder=40,
    )


def scale_bar(ax, km: float, x=0.62, y=0.06):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    latitude = y0 + y * (y1 - y0)
    longitude = x0 + x * (x1 - x0)
    degrees = km / (111.32 * max(math.cos(math.radians(latitude)), 0.25))
    ax.plot([longitude, longitude + degrees], [latitude, latitude], color="black", lw=2.4, zorder=40)
    ax.text(longitude + degrees / 2, latitude + 0.02 * (y1 - y0), f"{km:g} km", ha="center", va="bottom", fontsize=8, path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=40)


def inventory_points():
    xs, ys = [], []
    for shp, _ in shapefile_records(INVENTORY):
        for point in shp.points:
            xs.append(point[0])
            ys.append(point[1])
    return np.asarray(xs), np.asarray(ys)


def plot_countries(ax, extent, highlight=None, labels=False):
    """Plot Natural Earth country polygons with optional restrained highlighting."""
    highlight = highlight or {}
    xmin, xmax, ymin, ymax = extent
    for shp, attrs in shapefile_records(COUNTRIES):
        points = np.asarray(shp.points, dtype=float)
        if len(points) == 0:
            continue
        if points[:, 0].max() < xmin or points[:, 0].min() > xmax or points[:, 1].max() < ymin or points[:, 1].min() > ymax:
            continue
        name = attrs.get("ADMIN", attrs.get("NAME", ""))
        facecolor = highlight.get(name, "#E8E8E5")
        polygon_fill(ax, shp, facecolor, edgecolor="#777777", alpha=1.0, lw=0.45, zorder=2)

    if labels:
        country_labels = {
            "Pakistan": (69.0, 29.7),
            "China": (83.5, 41.5),
            "India": (78.3, 25.0),
            "Afghanistan": (66.2, 34.5),
            "Iran": (53.8, 31.5),
            "Tajikistan": (70.8, 39.0),
            "Kyrgyzstan": (74.8, 41.8),
            "Nepal": (84.0, 28.0),
        }
        for name, (x, y) in country_labels.items():
            if xmin <= x <= xmax and ymin <= y <= ymax:
                ax.text(
                    x,
                    y,
                    name,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color="#303030",
                    fontweight="bold" if name in {"Pakistan", "China"} else "normal",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=8,
                )


def figure_1_study_area() -> None:
    dem, extent = read_dem()
    xs, ys = inventory_points()
    fig = plt.figure(figsize=(13.2, 7.8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.95, 1.35],
        left=0.045,
        right=0.925,
        bottom=0.17,
        top=0.975,
        hspace=0.20,
        wspace=0.16,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    # (a) Global locator.
    world_extent = (-180, 180, -58, 88)
    ax_a.set_facecolor("#DDECF4")
    plot_countries(ax_a, world_extent, highlight={"Pakistan": "#D95F59", "China": "#F2A65A"})
    ax_a.add_patch(Rectangle((45, 20), 50, 30, fill=False, ec="#B2182B", lw=1.5, zorder=10))
    ax_a.set_xlim(world_extent[0], world_extent[1]); ax_a.set_ylim(world_extent[2], world_extent[3])
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xticks([]); ax_a.set_yticks([])

    # (b) Regional context with country names and the official study-area outline.
    regional_extent = (44, 96, 19, 51)
    ax_b.set_facecolor("#DDECF4")
    plot_countries(
        ax_b,
        regional_extent,
        highlight={"Pakistan": "#F6D7D5", "China": "#F9E2C3"},
        labels=True,
    )
    plot_boundary(ax_b, lw=1.6, color="#B2182B", zorder=12)
    ax_b.add_patch(Rectangle((59.8, 22.9), 20.8, 19.1, fill=False, ec="#222222", lw=0.9, ls="--", zorder=11))
    ax_b.set_xlim(regional_extent[0], regional_extent[1]); ax_b.set_ylim(regional_extent[2], regional_extent[3])
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Longitude (deg E)"); ax_b.set_ylabel("Latitude (deg N)")
    ax_b.grid(color="white", lw=0.45, alpha=0.7)

    # (c) Detailed terrain, subdomain, infrastructure, and inventory map.
    terrain = terrain_background(ax_c, dem, extent)
    plot_domains(ax_c, alpha=0.30, labels=False)
    plot_lines(ax_c, RIVERS, "#0072B2", lw=0.75, alpha=0.95, zorder=7)
    plot_lines(ax_c, FAULTS, "#8C2D04", lw=0.85, alpha=0.90, zorder=8)
    plot_lines(ax_c, ROADS, "#585858", lw=0.45, alpha=0.60, zorder=9)
    plot_lines(ax_c, KKH, "#FFE01B", lw=1.8, alpha=1.0, zorder=12)
    ax_c.scatter(xs, ys, s=7, color="#D73027", edgecolor="white", linewidth=0.25, alpha=0.88, zorder=13)
    plot_boundary(ax_c, lw=1.5)
    ax_c.set_xlim(59.8, 80.6); ax_c.set_ylim(22.9, 42.0)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel("Longitude (deg E)"); ax_c.set_ylabel("Latitude (deg N)")
    ax_c.grid(color="white", lw=0.35, alpha=0.45)

    panel_label(ax_a, "(a)"); panel_label(ax_b, "(b)"); panel_label(ax_c, "(c)")
    north_arrow(ax_c, x=0.955, y=0.93)
    scale_bar(ax_c, 500, x=0.57, y=0.055)
    city_style = dict(fontsize=9.0, fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=30)
    for x, y, name in [(75.99,39.47,"Kashgar"),(74.31,35.92,"Gilgit"),(73.05,33.68,"Islamabad"),(66.99,30.18,"Quetta"),(67.01,24.86,"Karachi")]:
        ax_c.plot(x, y, marker="s", ms=4.2, mfc="white", mec="black", mew=0.5, zorder=29)
        ax_c.text(x+0.12, y+0.09, name, **city_style)
    legend = [Patch(facecolor=DOMAIN_COLORS[d], edgecolor="white", alpha=0.55, label=DOMAIN_DISPLAY[d]) for d in DOMAIN_COLORS]
    legend += [Line2D([0],[0],color="#FFE01B",lw=2.4,label="KKH proxy route (N35/314)"), Line2D([0],[0],color="#0072B2",lw=1.2,label="Major rivers"), Line2D([0],[0],color="#8C2D04",lw=1.2,label="Active faults"), Line2D([0],[0],marker="o",ls="",mfc="#D73027",mec="white",label="Landslide inventory")]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.015))
    cax = fig.add_axes([0.94, 0.30, 0.014, 0.40])
    cbar = fig.colorbar(terrain, cax=cax, orientation="vertical")
    cbar.set_label("Elevation (m)")
    save(fig, 1, "study_area")


def figure_2_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.02, "1", "Inventory and predictors", "1,508 slope failures\n1,808 controls\n19 conventional + 64 AE"),
        (0.215, "2", "Leakage-free modelling", "5 outer spatial folds\n20 km train-test buffers\n6 base learners + L2 stack"),
        (0.41, "3", "Robustness tests", "3 terrain-matched controls\nlandslide/rockfall strata\nroad-distance ablation"),
        (0.605, "4", "Spatial generalisation", "5 leave-one-domain-out tests\n2,000 block bootstraps\n15-D harmonised AoA"),
        (0.80, "5", "Mapped outputs", "Three 250 m score surfaces\nmodel disagreement\nroad priorities and stability"),
    ]
    fills = ["#DCEAF7", "#E4F3EA", "#FCE8D8", "#EEE8F5", "#F7EFD4"]
    for i, ((x, number, title, body), fill) in enumerate(zip(stages, fills)):
        patch = FancyBboxPatch((x,0.20),0.175,0.60,boxstyle="round,pad=0.015,rounding_size=0.015",facecolor=fill,edgecolor="#333333",lw=1.1)
        ax.add_patch(patch)
        ax.text(x+0.022,0.72,number,ha="center",va="center",fontsize=12,fontweight="bold",color="white",bbox=dict(boxstyle="circle,pad=0.28",facecolor="#333333",edgecolor="none"))
        ax.text(x+0.0875,0.61,title,ha="center",va="center",fontsize=10.2,fontweight="bold")
        ax.text(x+0.0875,0.42,body,ha="center",va="center",fontsize=8.8,linespacing=1.45)
        if i < len(stages)-1:
            ax.annotate("",xy=(x+0.198,0.50),xytext=(x+0.178,0.50),arrowprops=dict(arrowstyle="-|>",lw=1.6,color="#444444"))
    ax.text(0.5,0.91,"Transferability-aware analytical workflow",ha="center",va="center",fontsize=13,fontweight="bold")
    ax.text(0.5,0.09,"All preprocessing, encoding, threshold selection and meta-learning were fitted within the relevant training data.",ha="center",va="center",fontsize=9,color="#333333")
    save(fig, 2, "methodological_workflow")


def figure_3_validation() -> None:
    predictions = pd.read_csv(MODEL_DIR / "nested_spatial_cv_predictions.csv")
    calibration = pd.read_csv(MODEL_DIR / "nested_spatial_cv_calibration_bins.csv")
    pooled = pd.read_csv(MODEL_DIR / "nested_spatial_cv_pooled_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    ax = axes[0,0]
    for name, d in predictions.groupby("feature_set_name", sort=False):
        fpr,tpr,_=roc_curve(d.label,d.stacked_score); auc=pooled.loc[pooled.feature_set_name==name,"roc_auc"].iloc[0]
        ax.plot(fpr,tpr,lw=2,color=COLORS[name],label=f"{name} ({auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="#888888",lw=0.9); ax.set(xlabel="False-positive rate",ylabel="True-positive rate",xlim=(0,1),ylim=(0,1)); ax.legend(loc="lower right"); panel_label(ax,"(a)")
    ax = axes[0,1]
    prevalence=predictions.label.mean()
    for name,d in predictions.groupby("feature_set_name", sort=False):
        precision,recall,_=precision_recall_curve(d.label,d.stacked_score); ap=pooled.loc[pooled.feature_set_name==name,"pr_auc"].iloc[0]
        ax.plot(recall,precision,lw=2,color=COLORS[name],label=f"{name} ({ap:.3f})")
    ax.axhline(prevalence,ls="--",color="#888888",lw=0.9); ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1)); ax.legend(loc="lower left"); panel_label(ax,"(b)")
    ax=axes[1,0]
    ax.plot([0,1],[0,1],"--",color="#777777",lw=0.9)
    for name,d in calibration.groupby("feature_set_name",sort=False):
        ax.plot(d.mean_score,d.observed_positive_fraction,marker="o",ms=4,lw=1.8,color=COLORS[name],label=name)
    ax.set(xlabel="Mean susceptibility score",ylabel="Observed positive fraction",xlim=(0,1),ylim=(0,1)); panel_label(ax,"(c)")
    ax=axes[1,1]
    metrics=[("balanced_accuracy","Balanced accuracy"),("f1","F1 score"),("brier","Brier score"),("ece_10bin","ECE")]
    width=0.23; x=np.arange(len(metrics))
    for j,(_,row) in enumerate(pooled.iterrows()):
        vals=[row[m] for m,_ in metrics]
        ax.bar(x+(j-1)*width,vals,width=width,color=COLORS[row.feature_set_name],label=row.feature_set_name)
        for xx,v in zip(x+(j-1)*width,vals): ax.text(xx,v+0.015,f"{v:.3f}",ha="center",va="bottom",fontsize=7,rotation=90)
    ax.set_xticks(x,[label for _,label in metrics]); ax.set_ylim(0,1.02); ax.set_ylabel("Metric value"); ax.legend(loc="upper right"); panel_label(ax,"(d)")
    save(fig,3,"nested_spatial_cv_validation")


def downsample_raster(path: Path, scale=8):
    with rasterio.open(path) as ds:
        height=max(1,ds.height//scale); width=max(1,ds.width//scale)
        data=ds.read(1,out_shape=(height,width),resampling=rasterio.enums.Resampling.bilinear).astype(float)
        data[data==ds.nodata]=np.nan
        extent=(ds.bounds.left,ds.bounds.right,ds.bounds.bottom,ds.bounds.top)
    return data,extent


def figure_4_maps() -> None:
    paths=[MAP_DIR/"cpec_baseline_conventional_stacked_susceptibility_score_250m.tif",MAP_DIR/"cpec_baseline_alphaearth_embeddings_stacked_susceptibility_score_250m.tif",MAP_DIR/"cpec_baseline_conventional_alphaearth_embeddings_stacked_susceptibility_score_250m.tif"]
    arrays=[]
    for p in paths:
        a,extent=downsample_raster(p); arrays.append(a)
    difference=arrays[2]-arrays[0]; vmax=float(np.nanquantile(np.abs(difference),0.99))
    fig,axes=plt.subplots(2,2,figsize=(10.8,8.4),constrained_layout=True)
    score_norm=PowerNorm(gamma=0.45,vmin=0,vmax=1)
    for i,(ax,data) in enumerate(zip(axes.flat[:3],arrays)):
        image=ax.imshow(data,extent=extent,origin="upper",cmap="magma",norm=score_norm,interpolation="bilinear")
        plot_domains(ax,alpha=0.0); plot_boundary(ax,lw=0.8); ax.set_xlim(59.8,80.6); ax.set_ylim(22.9,42.0); ax.set_aspect("equal"); panel_label(ax,f"({chr(97+i)})")
        ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
    plot_lines(axes.flat[2],KKH,"#00FFFF",lw=1.1,zorder=20)
    ax=axes.flat[3]
    diff=ax.imshow(difference,extent=extent,origin="upper",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),interpolation="bilinear")
    plot_boundary(ax,lw=0.8); ax.set_xlim(59.8,80.6); ax.set_ylim(22.9,42.0); ax.set_aspect("equal"); panel_label(ax,"(d)"); ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
    c1=fig.colorbar(image,ax=list(axes.flat[:3]),orientation="horizontal",fraction=0.035,pad=0.03); c1.set_label("Case-control susceptibility score")
    c2=fig.colorbar(diff,ax=ax,orientation="horizontal",fraction=0.07,pad=0.08); c2.set_label("Fused minus Conventional score")
    save(fig,4,"baseline_susceptibility_maps")


def standardize(values):
    values=np.asarray(values,float); finite=np.isfinite(values)
    out=np.zeros_like(values)
    if finite.any():
        lo,hi=np.nanquantile(values,[0.05,0.95]); out[finite]=np.clip((values[finite]-lo)/(hi-lo if hi>lo else 1),0,1)
    return out


def figure_5_explainability_robustness() -> None:
    meta=pd.read_csv(MODEL_DIR/"nested_spatial_cv_meta_coefficients.csv")
    importance=pd.read_csv(SHAP_DIR/"xgboost_treeshap_global_importance.csv")
    shap_values=pd.read_parquet(SHAP_DIR/"xgboost_treeshap_values_conventional_alphaearth_embeddings.parquet")
    feature_values=pd.read_parquet(SHAP_DIR/"xgboost_feature_values_conventional_alphaearth_embeddings.parquet")
    robustness=pd.read_csv(ROBUST_DIR/"robustness_experiment_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(11.2,8.6),constrained_layout=True)
    ax=axes[0,0]
    grouped=meta.groupby(["feature_set_name","base_model_name"]).meta_coefficient.agg(["mean","std"]).reset_index()
    models=list(grouped.base_model_name.unique()); y=np.arange(len(models)); offsets=[-0.22,0,0.22]
    for off,(name,d) in zip(offsets,grouped.groupby("feature_set_name",sort=False)):
        d=d.set_index("base_model_name").reindex(models); ax.errorbar(d["mean"],y+off,xerr=d["std"],fmt="o",capsize=2,color=COLORS[name],label=name)
    ax.axvline(0,color="#777",lw=0.8); ax.set_yticks(y,models); ax.set_xlabel("Meta-learner coefficient (mean +/- SD)"); ax.legend(fontsize=7.5,loc="lower left",bbox_to_anchor=(0.0,1.005),ncol=1); panel_label(ax,"(a)")
    ax=axes[0,1]
    fused=importance[importance.feature_set=="conventional_alphaearth_embeddings"].nlargest(12,"mean_absolute_treeshap").sort_values("mean_absolute_treeshap")
    ax.barh(fused.factor_name,fused.mean_absolute_treeshap,color=["#D55E00" if f.startswith("A") else "#0072B2" for f in fused.factor]); ax.set_xlabel("Mean |TreeSHAP| (model-margin units)"); panel_label(ax,"(b)")
    ax=axes[1,0]
    top=list(importance[importance.feature_set=="conventional_alphaearth_embeddings"].nlargest(10,"mean_absolute_treeshap").factor)
    remaining_ae=[c for c in shap_values.columns if c.startswith("A") and c not in top]
    display={r.factor:r.factor_name for r in importance[importance.feature_set=="conventional_alphaearth_embeddings"].itertuples()}
    rows=[]
    for factor in top:
        rows.append((display[factor],shap_values[factor].to_numpy(),standardize(feature_values[factor])))
    if remaining_ae:
        rows.append(("Remaining AE dimensions (aggregate)",shap_values[remaining_ae].sum(axis=1).to_numpy(),None))
    rng=np.random.default_rng(SEED)
    for yi,(label,sv,colorv) in enumerate(rows[::-1]):
        jitter=rng.normal(0,0.11,len(sv)); yy=np.full(len(sv),yi)+jitter
        if colorv is None: ax.scatter(sv,yy,s=4,c="#888888",alpha=0.35,rasterized=True)
        else: ax.scatter(sv,yy,s=4,c=colorv,cmap="coolwarm",vmin=0,vmax=1,alpha=0.45,rasterized=True)
    ax.axvline(0,color="#666",lw=0.8); ax.set_yticks(np.arange(len(rows)),[r[0] for r in rows[::-1]]); ax.set_xlabel("TreeSHAP contribution (model-margin units)"); panel_label(ax,"(c)")
    ax=axes[1,1]
    labels={"accessibility_matched_controls_repeat_01":"Matched controls 1","accessibility_matched_controls_repeat_02":"Matched controls 2","accessibility_matched_controls_repeat_03":"Matched controls 3","inventory_typology_landslide_only":"Landslide-only","inventory_typology_rockfall_only":"Rockfall-only","road_distance_ablation":"No road distance"}
    robustness["label"]=robustness.experiment.map(labels); robustness=robustness.iloc[::-1]
    y=np.arange(len(robustness)); ax.scatter(robustness.roc_auc,y,label="ROC-AUC",color="#0072B2",s=42); ax.scatter(robustness.pr_auc,y,label="PR-AUC",color="#D55E00",s=42,marker="s")
    primary=pd.read_csv(MODEL_DIR/"nested_spatial_cv_pooled_metrics.csv").query("feature_set == 'conventional_alphaearth_embeddings'").iloc[0]
    ax.axvline(primary.roc_auc,color="#0072B2",ls="--",lw=0.9); ax.axvline(primary.pr_auc,color="#D55E00",ls="--",lw=0.9)
    ax.set_yticks(y,robustness.label); ax.set_xlim(0.88,0.99); ax.set_xlabel("Nested spatial-CV discrimination"); ax.legend(); panel_label(ax,"(d)")
    save(fig,5,"explainability_and_robustness")


def figure_6_transferability_aoa() -> None:
    dem,extent=read_dem(); metrics=pd.read_csv(LODO_DIR/"leave_one_domain_out_metrics.csv"); ci=pd.read_csv(LODO_DIR/"leave_one_domain_out_block_bootstrap_ci.csv"); aoa=pd.read_csv(AOA_DIR/"harmonised_aoa_domain_summary.csv")
    fused=metrics[metrics.feature_set=="conventional_alphaearth_embeddings"].copy(); domains=["Balochistan","Gilgit-Baltistan","KPK-AJK","Kashgar","Punjab-Sindh"]
    fig,axes=plt.subplots(2,2,figsize=(11.0,8.2),constrained_layout=True)
    ax=axes[0,0]; terrain_background(ax,dem,extent); plot_domains(ax,alpha=0.55,labels=True); plot_boundary(ax,lw=1.0); ax.set_xlim(59.8,80.6); ax.set_ylim(22.9,42.0); ax.set_aspect("equal"); panel_label(ax,"(a)"); ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
    ax=axes[0,1]; y=np.arange(len(domains)); offsets=[-0.11,0.11]
    for off,metric,label,color in [(offsets[0],"roc_auc","ROC-AUC","#0072B2"),(offsets[1],"pr_auc","PR-AUC","#D55E00")]:
        vals=[]; lows=[]; highs=[]
        for d in domains:
            est=float(fused.loc[fused.held_out_domain==d,metric].iloc[0]); row=ci[(ci.feature_set=="conventional_alphaearth_embeddings")&(ci.held_out_domain==d)&(ci.metric==metric)].iloc[0]; vals.append(est); lows.append(est-row.ci95_low); highs.append(row.ci95_high-est)
        ax.errorbar(vals,y+off,xerr=[lows,highs],fmt="o",capsize=2,color=color,label=label)
    ax.set_yticks(y,domains); ax.set_xlim(0.78,1.005); ax.set_xlabel("Held-out-domain metric (95% block-bootstrap CI)"); ax.legend(); panel_label(ax,"(b)")
    ax=axes[1,0]; x=np.arange(len(domains)); ordered=fused.set_index("held_out_domain").reindex(domains); width=0.36
    ax.bar(x-width/2,ordered.brier,width,color="#4C78A8",label="Brier score"); ax.bar(x+width/2,ordered.ece_10bin,width,color="#F58518",label="ECE"); ax.set_xticks(x,domains,rotation=25,ha="right"); ax.set_ylabel("Calibration error (lower is better)"); ax.legend(); panel_label(ax,"(c)")
    ax=axes[1,1]; pivot=aoa.pivot(index="feature_set_name",columns="held_out_domain",values="aoa_coverage").reindex(index=["Conventional","AlphaEarth Embeddings","Conventional + AlphaEarth Embeddings"],columns=domains)
    im=ax.imshow(pivot.values,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]): ax.text(j,i,f"{100*pivot.iloc[i,j]:.1f}%",ha="center",va="center",fontsize=8,color="white" if pivot.iloc[i,j]>0.65 else "black")
    ax.set_xticks(np.arange(len(domains)),domains,rotation=25,ha="right"); ax.set_yticks(np.arange(len(pivot.index)),pivot.index); c=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.02); c.set_label("AoA coverage"); panel_label(ax,"(d)")
    save(fig,6,"transferability_and_aoa")


def figure_7_road_exposure() -> None:
    segments=pd.read_csv(ROAD_DIR/"road_segment_scores_250m.csv"); summary=pd.read_csv(ROAD_DIR/"road_network_exposure_summary.csv"); routes=pd.read_csv(ROAD_DIR/"named_route_exposure_ranking.csv").head(10)
    dem,dem_extent=read_dem()
    fig,axes=plt.subplots(2,2,figsize=(11.2,8.4),constrained_layout=True)
    line_segments=np.stack([segments[["start_lon","start_lat"]].to_numpy(),segments[["end_lon","end_lat"]].to_numpy()],axis=1)
    score_norm=PowerNorm(gamma=0.45,vmin=0,vmax=1)
    ax=axes[0,0]; terrain_background(ax,dem,dem_extent); plot_domains(ax,alpha=0.10); collection=LineCollection(line_segments,cmap="magma",norm=score_norm,linewidths=0.85); collection.set_array(segments.fused_score.to_numpy()); ax.add_collection(collection); plot_boundary(ax,lw=0.8); ax.set_xlim(59.8,80.6); ax.set_ylim(22.9,42.0); ax.set_aspect("equal"); ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)"); panel_label(ax,"(a)"); c=fig.colorbar(collection,ax=ax,fraction=0.045,pad=0.02); c.set_label("Susceptibility score")
    ax=axes[0,1]; terrain_background(ax,dem,dem_extent); k=segments[segments.is_kkh_proxy]; k_lines=np.stack([k[["start_lon","start_lat"]].to_numpy(),k[["end_lon","end_lat"]].to_numpy()],axis=1); kc=LineCollection(k_lines,cmap="magma",norm=score_norm,linewidths=1.8); kc.set_array(k.fused_score.to_numpy()); ax.add_collection(kc); p90=float(pd.read_json(ROAD_DIR/"run_manifest.json",typ="series")["score_thresholds"]["score_p90"]); hot=k[k.fused_score>=p90]; hlines=np.stack([hot[["start_lon","start_lat"]].to_numpy(),hot[["end_lon","end_lat"]].to_numpy()],axis=1); ax.add_collection(LineCollection(hlines,colors="#00FFFF",linewidths=1.1)); ax.set_xlim(72.3,79.2); ax.set_ylim(33.2,40.7); ax.set_aspect("equal"); ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)"); panel_label(ax,"(b)"); ax.legend(handles=[Line2D([0],[0],color="#00FFFF",lw=2,label="Segments >= map P90")],loc="lower left")
    ax=axes[1,0]; groups=["Full CPEC road network","KKH proxy route (N35/314)"]; d=summary.set_index("road_group").reindex(groups); total=d.total_length_km.to_numpy(); pct80=100*d.actual_length_ge_p80_km.to_numpy()/total; pct90=100*d.actual_length_ge_p90_km.to_numpy()/total; x=np.arange(2); width=0.35; ax.bar(x-width/2,pct80,width,label=">= P80",color="#E69F00"); ax.bar(x+width/2,pct90,width,label=">= P90",color="#D55E00"); ax.set_xticks(x,["Full network","KKH proxy"]); ax.set_ylabel("Share of road length (%)"); ax.legend(); panel_label(ax,"(c)")
    ax=axes[1,1]; routes=routes.sort_values("actual_length_ge_p90_km"); y=np.arange(len(routes)); ax.barh(y,routes.actual_length_ge_p90_km,color="#4C78A8",alpha=0.85); ax.set_yticks(y,routes.route_label); ax.set_xlabel("Actual road length >= map P90 (km)"); ax2=ax.twiny(); ax2.scatter(routes.length_weighted_mean_score_no_road_distance,y,color="#D55E00",marker="D",s=25,label="Mean score without road distance"); ax2.set_xlim(0,1); ax2.set_xlabel("Road-ablation mean score"); panel_label(ax,"(d)")
    save(fig,7,"road_exposure_and_ablation")


def main() -> None:
    configure()
    figure_1_study_area()
    figure_2_workflow()
    figure_3_validation()
    figure_4_maps()
    figure_5_explainability_robustness()
    figure_6_transferability_aoa()
    figure_7_road_exposure()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
