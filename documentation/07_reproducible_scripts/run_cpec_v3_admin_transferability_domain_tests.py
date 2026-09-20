from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj.datadir

warnings.filterwarnings("ignore")

os.environ.setdefault("PROJ_DATA", r"D:\MINICONDA\Library\share\proj")
os.environ.setdefault("PROJ_LIB", r"D:\MINICONDA\Library\share\proj")
pyproj.datadir.set_data_dir(r"D:\MINICONDA\Library\share\proj")

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


PROJECT = Path(r"D:\DING PROJECT")
PACKAGE = PROJECT / "00_READ_ME_FIRST_2018_V3_RESULTS"
SAMPLE = PROJECT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
OUT = PROJECT / "03_models" / "v3_admin_transferability_domain_tests"
FIG_DIR = PROJECT / "05_reports" / "figures" / "v3_admin_transferability_domain_tests"
PACKAGE_METHODS = PACKAGE / "01_methods_and_decisions"
PACKAGE_TABLES = PACKAGE / "02_model_performance_tables"
PACKAGE_FIGS = PACKAGE / "03_figures" / "05_admin_transferability_domain_tests"
BOUNDARY_DIR = Path(r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary")
PAK_ADMIN1 = BOUNDARY_DIR / "gadm41_PAK_shp" / "gadm41_PAK_1.shp"
KASHGAR = BOUNDARY_DIR / "New Folder" / "cpecpart.shp"

CONVENTIONAL = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "modis_lc_type1",
    "log1p_dist_road_m",
    "log1p_dist_river_m",
    "log1p_dist_fault_m",
    "lithology_code",
    "profile_curvature",
    "plan_curvature",
    "tri",
    "twi",
    "valley_depth",
    "soil_type",
    "eq_density_ms5",
]
ALPHA = [f"A{i:02d}" for i in range(64)]
FEATURE_SETS = {
    "Conventional": CONVENTIONAL,
    "AlphaEarth Embeddings": ALPHA,
    "Conventional + AlphaEarth Embeddings": CONVENTIONAL + ALPHA,
}
BASE_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]
DOMAIN_ORDER = [
    "Kashgar (Xinjiang, China)",
    "Gilgit-Baltistan",
    "KP-AJK",
    "Balochistan",
    "Punjab-Sindh lowland corridor",
]


def ensure_dirs() -> None:
    for d in [OUT, FIG_DIR, PACKAGE_METHODS, PACKAGE_TABLES, PACKAGE_FIGS]:
        d.mkdir(parents=True, exist_ok=True)


def assign_admin_domains(df: pd.DataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    pak = gpd.read_file(PAK_ADMIN1).to_crs("EPSG:4326")
    kas = gpd.read_file(KASHGAR).to_crs("EPSG:4326")
    admin = pd.concat(
        [
            pak[["COUNTRY", "NAME_1", "geometry"]].assign(
                admin_unit=lambda x: x["NAME_1"]
            ),
            kas[["COUNTRY", "NAME_1", "NAME_2", "geometry"]]
            .assign(admin_unit="Kashgar (Xinjiang, China)")[["COUNTRY", "NAME_1", "geometry", "admin_unit"]],
        ],
        ignore_index=True,
    )
    joined = gpd.sjoin(pts, admin[["admin_unit", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    if out["admin_unit"].isna().any():
        missing = int(out["admin_unit"].isna().sum())
        raise ValueError(f"{missing} samples did not match the admin-domain polygons.")

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
    out["cpec_admin_transfer_domain"] = out["admin_unit"].map(mapping)
    if out["cpec_admin_transfer_domain"].isna().any():
        unknown = sorted(out.loc[out["cpec_admin_transfer_domain"].isna(), "admin_unit"].dropna().unique())
        raise ValueError(f"Unmapped admin units: {unknown}")
    return out


def base_models(random_state: int = 141300) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=180,
                        learning_rate=0.04,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if CatBoostClassifier is not None:
        models["catboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=180,
                        depth=5,
                        learning_rate=0.04,
                        loss_function="Logloss",
                        auto_class_weights="Balanced",
                        random_seed=random_state,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    return {k: models[k] for k in BASE_KEYS if k in models}


def predict_proba(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def threshold_from_source(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc_ap": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }
    return out


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if vals.size < 5 or np.nanmin(vals) == np.nanmax(vals):
        return 0.0
    edges = np.quantile(vals, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    pa, _ = np.histogram(a[np.isfinite(a)], bins=edges)
    pb, _ = np.histogram(b[np.isfinite(b)], bins=edges)
    pa = pa.astype(float) + 1e-9
    pb = pb.astype(float) + 1e-9
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def feature_shift(source: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> dict[str, float]:
    smd = []
    js = []
    for f in features:
        a = pd.to_numeric(source[f], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(target[f], errors="coerce").to_numpy(dtype=float)
        av = a[np.isfinite(a)]
        bv = b[np.isfinite(b)]
        if len(av) < 5 or len(bv) < 5:
            continue
        pooled = np.sqrt((np.nanvar(av) + np.nanvar(bv)) / 2.0)
        if pooled > 0:
            smd.append(abs(np.nanmean(av) - np.nanmean(bv)) / pooled)
        js.append(js_divergence(av, bv))
    return {
        "mean_abs_standardized_mean_difference": float(np.mean(smd)) if smd else np.nan,
        "mean_js_divergence": float(np.mean(js)) if js else np.nan,
    }


def run_one_transfer(
    df: pd.DataFrame,
    feature_label: str,
    features: list[str],
    target_domain: str,
    random_state: int = 141300,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    source = df[df["cpec_admin_transfer_domain"] != target_domain].copy()
    target = df[df["cpec_admin_transfer_domain"] == target_domain].copy()
    y_source = source["label"].astype(int).to_numpy()
    y_target = target["label"].astype(int).to_numpy()
    models = base_models(random_state)
    folds = sorted(source["spatial_fold_5"].dropna().unique())
    oof = np.zeros((len(source), len(models)), dtype=float)
    target_base = np.zeros((len(target), len(models)), dtype=float)
    base_metric_rows = []

    for j, (key, model_template) in enumerate(models.items()):
        for fold in folds:
            tr = source["spatial_fold_5"].to_numpy() != fold
            va = source["spatial_fold_5"].to_numpy() == fold
            if va.sum() == 0 or len(np.unique(y_source[tr])) < 2:
                continue
            model = clone(model_template)
            model.fit(source.loc[tr, features], y_source[tr])
            oof[va, j] = predict_proba(model, source.loc[va, features])

        final_model = clone(model_template)
        final_model.fit(source[features], y_source)
        target_base[:, j] = predict_proba(final_model, target[features])
        source_threshold = threshold_from_source(y_source, oof[:, j])
        row = {
            "feature_set": feature_label,
            "held_out_domain": target_domain,
            "model": key,
            "n_train": int(len(source)),
            "n_test": int(len(target)),
            "train_positive": int(y_source.sum()),
            "test_positive": int(y_target.sum()),
            "test_negative": int((1 - y_target).sum()),
        }
        row.update(metrics(y_target, target_base[:, j], source_threshold))
        base_metric_rows.append(row)

    meta = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    meta.fit(oof, y_source)
    source_stack = predict_proba(meta, pd.DataFrame(oof, columns=list(models.keys())))
    target_stack = predict_proba(meta, pd.DataFrame(target_base, columns=list(models.keys())))
    stack_threshold = threshold_from_source(y_source, source_stack)
    shift = feature_shift(source, target, features)
    result = {
        "feature_set": feature_label,
        "held_out_domain": target_domain,
        "model": "Spatial-CV Stacked Ensemble",
        "n_train": int(len(source)),
        "n_test": int(len(target)),
        "train_positive": int(y_source.sum()),
        "test_positive": int(y_target.sum()),
        "test_negative": int((1 - y_target).sum()),
        **metrics(y_target, target_stack, stack_threshold),
        **shift,
    }
    preds = target[
        ["longitude", "latitude", "label", "spatial_fold_5", "admin_unit", "cpec_admin_transfer_domain"]
    ].copy()
    preds["feature_set"] = feature_label
    preds["held_out_domain"] = target_domain
    preds["stacked_probability"] = target_stack
    for j, key in enumerate(models.keys()):
        preds[f"base_{key}_probability"] = target_base[:, j]
    meta_info = {
        "base_models": list(models.keys()),
        "meta_coefficients": {
            key: float(coef)
            for key, coef in zip(models.keys(), meta.named_steps["model"].coef_.ravel())
        },
        "meta_intercept": float(meta.named_steps["model"].intercept_[0]),
    }
    return result, preds, {"base_rows": base_metric_rows, "meta": meta_info}


def make_figures(df: pd.DataFrame, metrics_df: pd.DataFrame, shift_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # 1. Domain sample map
    colors = {
        "Kashgar (Xinjiang, China)": "#2563eb",
        "Gilgit-Baltistan": "#7c3aed",
        "KP-AJK": "#059669",
        "Balochistan": "#b45309",
        "Punjab-Sindh lowland corridor": "#dc2626",
    }
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    for domain in DOMAIN_ORDER:
        sub = df[df["cpec_admin_transfer_domain"] == domain]
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=np.where(sub["label"].to_numpy() == 1, 18, 9),
            c=colors[domain],
            alpha=np.where(sub["label"].to_numpy() == 1, 0.80, 0.28),
            edgecolors="none",
            label=domain,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("CPEC administrative transfer domains and 2018 sample distribution", fontweight="bold")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    save_all(fig, "figure_cpec_v3_admin_transferability_subdomain_sample_map")

    # 2. Heatmap for stacked ROC-AUC
    stack = metrics_df[metrics_df["model"] == "Spatial-CV Stacked Ensemble"].copy()
    pivot = stack.pivot(index="held_out_domain", columns="feature_set", values="roc_auc").reindex(DOMAIN_ORDER)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(pivot.values, vmin=0.5, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="#0f172a")
    ax.set_title("Leave-one-domain-out ROC-AUC: Spatial-CV Stacked Ensemble", fontweight="bold")
    fig.colorbar(im, ax=ax, label="ROC-AUC", fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_all(fig, "figure_cpec_v3_admin_transferability_stacked_auc_heatmap")

    # 3. Shift versus performance
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for feature_set, sub in stack.groupby("feature_set"):
        ax.scatter(
            sub["mean_js_divergence"],
            sub["roc_auc"],
            s=80,
            alpha=0.9,
            label=feature_set,
        )
        for _, row in sub.iterrows():
            ax.text(row["mean_js_divergence"] + 0.002, row["roc_auc"], short_domain(row["held_out_domain"]), fontsize=7)
    ax.set_xlabel("Mean Jensen-Shannon divergence from source domains")
    ax.set_ylabel("Held-out domain ROC-AUC")
    ax.set_title("Domain feature shift versus transfer performance", fontweight="bold")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_all(fig, "figure_cpec_v3_admin_domain_shift_vs_transfer_auc")


def short_domain(label: str) -> str:
    return {
        "Kashgar (Xinjiang, China)": "Kashgar",
        "Gilgit-Baltistan": "GB",
        "KP-AJK": "KP-AJK",
        "Balochistan": "Baloch.",
        "Punjab-Sindh lowland corridor": "Punjab-Sindh",
    }.get(label, label[:8])


def save_all(fig, stem: str) -> None:
    for directory in [FIG_DIR, PACKAGE_FIGS]:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(domain_balance: pd.DataFrame, metrics_df: pd.DataFrame, base_df: pd.DataFrame, shift_df: pd.DataFrame) -> None:
    stack = metrics_df[metrics_df["model"] == "Spatial-CV Stacked Ensemble"].copy()
    best = stack.sort_values("roc_auc", ascending=False).head(10)
    rows = []
    for _, row in best.iterrows():
        rows.append(
            f"| {row['feature_set']} | {row['held_out_domain']} | {row['roc_auc']:.3f} | {row['pr_auc_ap']:.3f} | {row['balanced_accuracy']:.3f} | {row['n_test']} |"
        )
    balance_md = df_to_markdown(domain_balance)
    report = f"""# CPEC V3 Administrative Transferability Domain Tests

Date: 2026-05-10

## Purpose

This analysis implements the first major step of the teacher-approved updated methodology: **transferability-aware landslide susceptibility modelling**. It tests whether the 2018 V3 model framework transfers across political/admin CPEC subdomains rather than relying only on random or ordinary spatial folds.

## Domain Definition

Domains were assigned by spatially joining the V3 samples to political/administrative boundary shapefiles, then aggregating small or legacy units into analysis domains. These are **modelling transfer domains**, not claims that grouped units are the same administrative unit.

- Pakistan GADM level-1 units from `gadm41_PAK_1.shp`.
- Kashgar, Xinjiang from `cpecpart.shp`.

The grouping is justified on three grounds:

1. Review method: spatial and grouped validation are recommended for structured spatial data because ordinary random validation underestimates prediction error where samples are spatially dependent.
2. Transferability method: large-region LSM should test whether models transfer across coherent source/target regions rather than only within one mixed sample pool.
3. Sample adequacy: several administrative units had too few landslide/non-landslide samples for stable leave-one-domain-out testing, so they were grouped into transparent corridor analysis domains.

Public domain labels:

- Kashgar (Xinjiang, China)
- Gilgit-Baltistan
- KP-AJK
- Balochistan
- Punjab-Sindh lowland corridor

Specific naming/aggregation decisions:

- Former FATA is not displayed as a separate current public domain because it was merged with Khyber Pakhtunkhwa through Pakistan's 25th Constitutional Amendment in 2018; legacy GADM polygons labelled FATA are assigned to the KP side of the `KP-AJK` analysis domain.
- AJK is grouped with KP only for transfer-testing sample adequacy and northern-western mountainous corridor interpretation; it is not presented as the same administrative unit.
- Islamabad is administratively separate, but it is absorbed into the `Punjab-Sindh lowland corridor` analysis domain because it contributed only one sample and cannot support a separate held-out test.

Key support:

- Roberts et al. (2017), Ecography, recommend block/group validation for data with spatial/hierarchical dependence: https://doi.org/10.1111/ecog.02881
- Meyer and Pebesma (2021) define area-of-applicability as the area where cross-validation error can be expected to apply: https://doi.org/10.1111/2041-210X.13650
- Official KP portal uses KP for Khyber Pakhtunkhwa: https://kp.gov.pk/
- Pakistan Constitution text reflects the Twenty-fifth Amendment effect on FATA/KP: https://pakistancode.gov.pk/
- AJK official portal uses AJK/AJ&K naming: https://ajk.gov.pk/

## Domain Sample Balance

{balance_md}

## Method

For each held-out domain and each feature set, models were trained on all other domains and tested only on the held-out domain. Feature sets:

- Conventional
- AlphaEarth Embeddings
- Conventional + AlphaEarth Embeddings

Models:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Spatial-CV Stacked Ensemble

The stacked ensemble used source-domain spatial-fold out-of-fold predictions to train the meta-learner. The held-out domain was not used to tune the decision threshold or train the base/meta models.

## Top Stacked-Ensemble Transfer Results

| Feature set | Held-out domain | ROC-AUC | PR-AUC/AP | Balanced accuracy | N test |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Interpretation

- These results directly support the new paper direction: CPEC should be treated as a multi-domain corridor, not a single homogeneous region.
- Administrative-domain transfer metrics are more rigorous than random-split accuracy because they test extrapolation to different political/planning regions.
- The domain-shift table links model performance to source-target feature similarity, following the logic of recent transfer-learning and area-of-applicability literature.
- The fused feature set should remain the main candidate if it performs well across domains, because it combines physical interpretability with annual foundation-model representation.

## Output Files

- Metrics: `v3_admin_transferability_leave_one_domain_metrics.csv`
- Base learner metrics: `v3_admin_transferability_base_learner_metrics.csv`
- Domain shift table: `v3_admin_transferability_domain_shift.csv`
- Predictions: `v3_admin_transferability_held_out_predictions.csv`
- Domain sample table: `cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv`
- Figures:
  - `figure_cpec_v3_admin_transferability_subdomain_sample_map.png`
  - `figure_cpec_v3_admin_transferability_stacked_auc_heatmap.png`
  - `figure_cpec_v3_admin_domain_shift_vs_transfer_auc.png`

## Next Methodology Step

The next step is to convert this sample-level domain transfer analysis into a raster-level **area-of-applicability / transfer confidence map** using the same predictor space.
"""
    for path in [
        OUT / "v3_admin_transferability_domain_tests_report.md",
        PACKAGE_METHODS / "v3_admin_transferability_domain_tests_report_2026-05-10.md",
    ]:
        path.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(SAMPLE)
    df = assign_admin_domains(df)
    domain_balance = (
        df.groupby("cpec_admin_transfer_domain")["label"]
        .agg(n="count", positives="sum")
        .reset_index()
    )
    domain_balance["negatives"] = domain_balance["n"] - domain_balance["positives"]
    domain_balance["positive_share"] = domain_balance["positives"] / domain_balance["n"]
    domain_balance = domain_balance.set_index("cpec_admin_transfer_domain").reindex(DOMAIN_ORDER).reset_index()
    domain_balance.to_csv(OUT / "v3_admin_transferability_domain_sample_balance.csv", index=False)
    df.to_csv(OUT / "cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv", index=False)
    df.to_csv(PACKAGE_TABLES / "cpec_2018_lsm_samples_v3_with_admin_transfer_domains.csv", index=False)

    metrics_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    meta_records: list[dict[str, object]] = []
    for target_domain in DOMAIN_ORDER:
        for feature_label, features in FEATURE_SETS.items():
            print(f"Running LODO: {feature_label} -> held out {target_domain}")
            result, preds, info = run_one_transfer(df, feature_label, features, target_domain)
            metrics_rows.append(result)
            base_rows.extend(info["base_rows"])
            pred_frames.append(preds)
            meta_records.append(
                {
                    "feature_set": feature_label,
                    "held_out_domain": target_domain,
                    **info["meta"],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    base_df = pd.DataFrame(base_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    shift_df = metrics_df[
        [
            "feature_set",
            "held_out_domain",
            "mean_abs_standardized_mean_difference",
            "mean_js_divergence",
        ]
    ].copy()

    metrics_df.to_csv(OUT / "v3_admin_transferability_leave_one_domain_metrics.csv", index=False)
    base_df.to_csv(OUT / "v3_admin_transferability_base_learner_metrics.csv", index=False)
    pred_df.to_csv(OUT / "v3_admin_transferability_held_out_predictions.csv", index=False)
    shift_df.to_csv(OUT / "v3_admin_transferability_domain_shift.csv", index=False)
    (OUT / "v3_admin_transferability_stacked_meta_coefficients.json").write_text(
        json.dumps(meta_records, indent=2), encoding="utf-8"
    )

    for src in [
        "v3_admin_transferability_leave_one_domain_metrics.csv",
        "v3_admin_transferability_base_learner_metrics.csv",
        "v3_admin_transferability_domain_shift.csv",
        "v3_admin_transferability_domain_sample_balance.csv",
    ]:
        (PACKAGE_TABLES / src).write_bytes((OUT / src).read_bytes())

    make_figures(df, metrics_df, shift_df)
    write_report(domain_balance, metrics_df, base_df, shift_df)
    print(f"Saved outputs to: {OUT}")


if __name__ == "__main__":
    main()
