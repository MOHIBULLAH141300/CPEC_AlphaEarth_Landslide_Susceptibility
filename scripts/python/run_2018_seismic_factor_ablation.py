from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
BASE_SAMPLE = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
V3_FACTOR_LIST = (
    PROJECT_ROOT
    / "03_models"
    / "multicollinearity_assessment_2018_v3"
    / "v3_final_selected_conventional_factors.csv"
)
OUT_DIR = PROJECT_ROOT / "03_models" / "seismic_factor_ablation_2018"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures" / "seismic_factor_ablation_2018"
REPORT = PROJECT_ROOT / "05_reports" / "seismic_factor_ablation_2018.md"
SAMPLE_OUT = OUT_DIR / "cpec_2018_lsm_samples_v3_seismic.csv"

SEISMIC_RASTERS = {
    "gem_pga_475yr_rock_g": PROJECT_ROOT
    / r"01_clean_data\new_step_required_data\B3_seismic_pga"
    / r"GEM-GSHM_PGA-475y-rock_v2023\v2023_1_pga_475_rock_3min.tif",
    "annual_eq_kernel_density_m4plus_per_1000km2": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_annual_earthquake_kernel_density_m4plus_per_1000km2_250m.tif",
    "annual_mag_weighted_eq_density_m4plus_per_1000km2": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_annual_magnitude_weighted_earthquake_kernel_density_m4plus_per_1000km2_250m.tif",
    "distance_to_annual_eq_m4plus_km": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_distance_to_nearest_annual_earthquake_m4plus_km_250m.tif",
    "distance_to_annual_eq_m5plus_km": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_distance_to_nearest_annual_earthquake_m5plus_km_250m.tif",
    "annual_max_shakemap_pga_percent_g": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_annual_max_shakemap_pga_percent_g_250m.tif",
    "annual_max_shakemap_mmi": PROJECT_ROOT
    / r"04_maps\annual_dynamic_2017_2024\2018\01_predictor_rasters_250m\annual_seismicity"
    / "cpec_2018_annual_max_shakemap_mmi_250m.tif",
}

SEISMIC_NAMES = {
    "gem_pga_475yr_rock_g": "GEM PGA 475-year hazard",
    "annual_eq_kernel_density_m4plus_per_1000km2": "Annual earthquake density M4+",
    "annual_mag_weighted_eq_density_m4plus_per_1000km2": "Annual magnitude-weighted earthquake density M4+",
    "distance_to_annual_eq_m4plus_km": "Distance to annual M4+ earthquake",
    "distance_to_annual_eq_m5plus_km": "Distance to annual M5+ earthquake",
    "annual_max_shakemap_pga_percent_g": "Annual maximum ShakeMap PGA",
    "annual_max_shakemap_mmi": "Annual maximum ShakeMap MMI",
}

SCREENED_SEISMIC = [
    "gem_pga_475yr_rock_g",
    "annual_mag_weighted_eq_density_m4plus_per_1000km2",
    "distance_to_annual_eq_m5plus_km",
    "annual_max_shakemap_pga_percent_g",
]

MODEL_NAMES = {
    "logistic_l2": "Logistic Regression (L2)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "stacked_l2_logistic": "Spatial-CV Stacked Ensemble",
}
BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]


def log(message: str) -> None:
    print(message, flush=True)


def sample_raster(df: pd.DataFrame, path: Path) -> np.ndarray:
    coords = list(zip(df["longitude"].astype(float), df["latitude"].astype(float)))
    values: list[float] = []
    with rasterio.open(path) as src:
        nodata = src.nodata
        for val in src.sample(coords, indexes=1):
            x = float(val[0])
            if nodata is not None and np.isclose(x, float(nodata)):
                values.append(np.nan)
            elif not np.isfinite(x):
                values.append(np.nan)
            else:
                values.append(x)
    return np.asarray(values, dtype=float)


def build_seismic_sample_table() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE_SAMPLE, encoding="utf-8-sig")
    for name, path in SEISMIC_RASTERS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        log(f"Sampling {name}: {path.name}")
        df[name] = sample_raster(df, path)
    df.to_csv(SAMPLE_OUT, index=False, encoding="utf-8-sig")

    qa = []
    for name in SEISMIC_RASTERS:
        s = pd.to_numeric(df[name], errors="coerce")
        qa.append(
            {
                "factor": name,
                "factor_name": SEISMIC_NAMES[name],
                "missing_count": int(s.isna().sum()),
                "missing_percent": float(s.isna().mean() * 100.0),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
                "mean": float(s.mean(skipna=True)),
                "p90": float(s.quantile(0.90)),
                "source_raster": str(SEISMIC_RASTERS[name]),
            }
        )
    pd.DataFrame(qa).to_csv(OUT_DIR / "seismic_sample_factor_qa.csv", index=False, encoding="utf-8-sig")
    return df


def numeric_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=cols)


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = numeric_table(df, cols)
    rows = []
    for col in cols:
        other = [c for c in cols if c != col]
        model = LinearRegression()
        model.fit(x[other], x[col])
        r2 = float(model.score(x[other], x[col]))
        rows.append(
            {
                "factor": col,
                "factor_name": SEISMIC_NAMES.get(col, col),
                "r2_against_other_factors": r2,
                "vif": np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def high_corr_table(df: pd.DataFrame, cols: list[str], threshold: float = 0.85) -> pd.DataFrame:
    x = numeric_table(df, cols)
    corr = x.corr(method="spearman")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = float(corr.loc[a, b])
            if abs(val) >= threshold:
                rows.append(
                    {
                        "factor_a": a,
                        "factor_a_name": SEISMIC_NAMES.get(a, a),
                        "factor_b": b,
                        "factor_b_name": SEISMIC_NAMES.get(b, b),
                        "spearman_r": val,
                    }
                )
    return pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)


def run_multicollinearity(df: pd.DataFrame, conventional: list[str]) -> None:
    all_seismic = list(SEISMIC_RASTERS.keys())
    candidate_cols = conventional + all_seismic
    screened_cols = conventional + SCREENED_SEISMIC

    vif_table(df, all_seismic).to_csv(OUT_DIR / "seismic_only_vif.csv", index=False, encoding="utf-8-sig")
    high_corr_table(df, all_seismic, threshold=0.80).to_csv(
        OUT_DIR / "seismic_only_high_correlation_pairs_abs_ge_0_80.csv", index=False, encoding="utf-8-sig"
    )
    vif_table(df, candidate_cols).to_csv(OUT_DIR / "conventional_plus_all_seismic_vif.csv", index=False, encoding="utf-8-sig")
    high_corr_table(df, candidate_cols).to_csv(
        OUT_DIR / "conventional_plus_all_seismic_high_correlation_pairs_abs_ge_0_85.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif_table(df, screened_cols).to_csv(
        OUT_DIR / "conventional_plus_screened_seismic_vif.csv", index=False, encoding="utf-8-sig"
    )

    decisions = []
    drop_reasons = {
        "annual_eq_kernel_density_m4plus_per_1000km2": "Dropped from screened set because the magnitude-weighted density encodes both event frequency and magnitude.",
        "distance_to_annual_eq_m4plus_km": "Dropped from screened set because M5+ distance better represents stronger earthquake forcing.",
        "annual_max_shakemap_mmi": "Dropped from screened set because ShakeMap PGA is the direct seismic-shaking variable requested by the teacher; MMI is retained as scenario/interpretation support.",
    }
    for factor in all_seismic:
        decisions.append(
            {
                "factor": factor,
                "factor_name": SEISMIC_NAMES[factor],
                "screening_decision": "keep_for_ablation" if factor in SCREENED_SEISMIC else "supporting_or_drop_for_ablation",
                "reason": "Retained as a distinct seismic-hazard/seismic-forcing diagnostic."
                if factor in SCREENED_SEISMIC
                else drop_reasons[factor],
            }
        )
    pd.DataFrame(decisions).to_csv(OUT_DIR / "seismic_factor_screening_decisions.csv", index=False, encoding="utf-8-sig")


def make_models(seed: int = 141300) -> dict[str, Callable[[], object]]:
    return {
        "logistic_l2": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=3.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=600,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=3.0,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "catboost": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=700,
                        depth=5,
                        learning_rate=0.03,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        auto_class_weights="Balanced",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=-1,
                    ),
                ),
            ]
        ),
    }


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def threshold_for_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def metric_row(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
    }


def feature_sets(conventional: list[str], alpha: list[str]) -> dict[str, dict[str, object]]:
    screened = SCREENED_SEISMIC
    all_seismic = list(SEISMIC_RASTERS.keys())
    return {
        "conventional_current": {
            "display": "Conventional",
            "features": conventional,
        },
        "conventional_gem_pga": {
            "display": "Conventional + GEM PGA",
            "features": conventional + ["gem_pga_475yr_rock_g"],
        },
        "conventional_usgs_annual_seismic": {
            "display": "Conventional + annual USGS seismic factors",
            "features": conventional + screened[1:],
        },
        "conventional_gem_usgs_seismic": {
            "display": "Conventional + GEM PGA + annual USGS seismic factors",
            "features": conventional + screened,
        },
        "fused_current": {
            "display": "Conventional + AlphaEarth Embeddings",
            "features": conventional + alpha,
        },
        "fused_gem_usgs_seismic": {
            "display": "Conventional + AlphaEarth Embeddings + seismic factors",
            "features": conventional + alpha + screened,
        },
        "conventional_all_seismic_diagnostic": {
            "display": "Conventional + all seismic factors (diagnostic)",
            "features": conventional + all_seismic,
        },
    }


def run_base_models(df: pd.DataFrame, fsets: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = make_models()
    y = df["label"].to_numpy()
    folds = sorted(df["spatial_fold_5"].unique())
    metrics = []
    preds = []
    for fs_key, cfg in fsets.items():
        cols = list(cfg["features"])
        x = df[cols]
        log(f"Feature set {fs_key}: {len(cols)} features")
        for model_key, factory in models.items():
            log(f"  model {model_key}")
            for fold in folds:
                train_idx = df["spatial_fold_5"] != fold
                test_idx = df["spatial_fold_5"] == fold
                model = factory()
                model.fit(x.loc[train_idx], y[train_idx.to_numpy()])
                train_prob = predict_positive(model, x.loc[train_idx])
                threshold = threshold_for_f1(y[train_idx.to_numpy()], train_prob)
                prob = predict_positive(model, x.loc[test_idx])
                y_test = y[test_idx.to_numpy()]
                metrics.append(
                    {
                        "feature_set": fs_key,
                        "feature_set_name": str(cfg["display"]),
                        "model": model_key,
                        "model_name": MODEL_NAMES[model_key],
                        "fold": int(fold),
                        "n_train": int(train_idx.sum()),
                        "n_test": int(test_idx.sum()),
                        "n_test_pos": int(y_test.sum()),
                        "n_test_neg": int((y_test == 0).sum()),
                        **metric_row(y_test, prob, threshold),
                    }
                )
                p = df.loc[
                    test_idx,
                    ["inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"],
                ].copy()
                p["sample_id"] = p.index.astype(int)
                p["feature_set"] = fs_key
                p["feature_set_name"] = str(cfg["display"])
                p["model"] = model_key
                p["model_name"] = MODEL_NAMES[model_key]
                p["probability"] = prob
                p["threshold"] = threshold
                p["prediction"] = (prob >= threshold).astype(int)
                preds.append(p)
            final_model = factory()
            final_model.fit(x, y)
            joblib.dump(final_model, OUT_DIR / f"model_{fs_key}_{model_key}.joblib")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)


def make_meta_matrix(oof: pd.DataFrame, fs_key: str) -> pd.DataFrame:
    d = oof[oof["feature_set"] == fs_key].copy()
    wide = d.pivot_table(index="sample_id", columns="model", values="probability", aggfunc="first").reset_index()
    wide.columns.name = None
    meta = d[
        ["sample_id", "inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"]
    ].drop_duplicates("sample_id")
    return meta.merge(wide, on="sample_id", how="inner")


def run_stacking(oof: pd.DataFrame, fsets: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stacked_preds = []
    stacked_metrics = []
    weights = []
    for fs_key, cfg in fsets.items():
        wide = make_meta_matrix(oof, fs_key)
        folds = sorted(wide["spatial_fold_5"].unique())
        for fold in folds:
            train_idx = wide["spatial_fold_5"] != fold
            test_idx = wide["spatial_fold_5"] == fold
            x_train = wide.loc[train_idx, BASE_MODEL_KEYS]
            y_train = wide.loc[train_idx, "label"].astype(int).to_numpy()
            x_test = wide.loc[test_idx, BASE_MODEL_KEYS]
            y_test = wide.loc[test_idx, "label"].astype(int).to_numpy()
            meta = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=5000,
                            random_state=141300,
                        ),
                    ),
                ]
            )
            meta.fit(x_train, y_train)
            train_prob = predict_positive(meta, x_train)
            threshold = threshold_for_f1(y_train, train_prob)
            prob = predict_positive(meta, x_test)
            stacked_metrics.append(
                {
                    "feature_set": fs_key,
                    "feature_set_name": str(cfg["display"]),
                    "model": "stacked_l2_logistic",
                    "model_name": MODEL_NAMES["stacked_l2_logistic"],
                    "fold": int(fold),
                    "n_train": int(train_idx.sum()),
                    "n_test": int(test_idx.sum()),
                    "n_test_pos": int(y_test.sum()),
                    "n_test_neg": int((y_test == 0).sum()),
                    **metric_row(y_test, prob, threshold),
                }
            )
            p = wide.loc[
                test_idx,
                ["sample_id", "inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"],
            ].copy()
            p["feature_set"] = fs_key
            p["feature_set_name"] = str(cfg["display"])
            p["model"] = "stacked_l2_logistic"
            p["model_name"] = MODEL_NAMES["stacked_l2_logistic"]
            p["probability"] = prob
            p["threshold"] = threshold
            p["prediction"] = (prob >= threshold).astype(int)
            stacked_preds.append(p)
            for model_key, coef in zip(BASE_MODEL_KEYS, meta.named_steps["model"].coef_.ravel()):
                weights.append(
                    {
                        "feature_set": fs_key,
                        "feature_set_name": str(cfg["display"]),
                        "fold": int(fold),
                        "base_model": model_key,
                        "base_model_name": MODEL_NAMES[model_key],
                        "meta_coefficient": float(coef),
                    }
                )
        final_meta = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=141300,
                    ),
                ),
            ]
        )
        final_meta.fit(wide[BASE_MODEL_KEYS], wide["label"].astype(int).to_numpy())
        joblib.dump(final_meta, OUT_DIR / f"model_{fs_key}_stacked_l2_logistic.joblib")
    return pd.concat(stacked_preds, ignore_index=True), pd.DataFrame(stacked_metrics), pd.DataFrame(weights)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["feature_set", "feature_set_name", "model", "model_name"])
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .reset_index()
        .sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
    )


def write_stacked_figures(pred: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stack = pred[pred["model"] == "stacked_l2_logistic"].copy()
    order = [
        "fused_gem_usgs_seismic",
        "fused_current",
        "conventional_gem_usgs_seismic",
        "conventional_usgs_annual_seismic",
        "conventional_gem_pga",
        "conventional_current",
    ]

    plt.figure(figsize=(7.6, 6.2))
    for fs in order:
        d = stack[stack["feature_set"] == fs]
        if d.empty:
            continue
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{d['feature_set_name'].iloc[0]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Seismic-Factor Ablation: Spatial-CV Stacked Ensemble ROC")
    plt.legend(loc="lower right", fontsize=7, frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_seismic_ablation_stacked_roc_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_seismic_ablation_stacked_roc_2018.pdf")
    plt.close()

    stack_summary = summary[summary["model"] == "stacked_l2_logistic"].copy()
    stack_summary["delta_pr_auc_vs_conventional"] = (
        stack_summary["pr_auc_mean"]
        - float(stack_summary.loc[stack_summary["feature_set"] == "conventional_current", "pr_auc_mean"].iloc[0])
    )
    stack_summary["delta_roc_auc_vs_conventional"] = (
        stack_summary["roc_auc_mean"]
        - float(stack_summary.loc[stack_summary["feature_set"] == "conventional_current", "roc_auc_mean"].iloc[0])
    )
    stack_summary.to_csv(OUT_DIR / "seismic_ablation_stacked_summary_with_deltas.csv", index=False, encoding="utf-8-sig")

    plot_df = stack_summary.set_index("feature_set").loc[[fs for fs in order if fs in stack_summary["feature_set"].values]]
    plt.figure(figsize=(8.4, 4.8))
    bars = plt.barh(plot_df["feature_set_name"], plot_df["pr_auc_mean"], color="#2f6f8f")
    plt.xlabel("Mean PR-AUC")
    plt.title("Seismic-Factor Ablation: Stacked Ensemble Performance")
    plt.xlim(max(0, plot_df["pr_auc_mean"].min() - 0.02), min(1, plot_df["pr_auc_mean"].max() + 0.02))
    for bar, val in zip(bars, plot_df["pr_auc_mean"]):
        plt.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_seismic_ablation_stacked_pr_auc_bar_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_seismic_ablation_stacked_pr_auc_bar_2018.pdf")
    plt.close()


def write_report(df: pd.DataFrame, conventional: list[str], fsets: dict[str, dict[str, object]], summary: pd.DataFrame) -> None:
    stack_summary = summary[summary["model"] == "stacked_l2_logistic"].copy()
    sample_qa = pd.read_csv(OUT_DIR / "seismic_sample_factor_qa.csv")
    screened_vif = pd.read_csv(OUT_DIR / "conventional_plus_screened_seismic_vif.csv")
    all_vif = pd.read_csv(OUT_DIR / "conventional_plus_all_seismic_vif.csv")

    lines = [
        "# 2018 Seismic-Factor Ablation",
        "",
        "## Purpose",
        "",
        "This experiment tests whether adding static probabilistic PGA and annual USGS seismic forcing improves the existing 2018 CPEC landslide-susceptibility model.",
        "",
        "## Inputs",
        "",
        f"- Base sample table: `{BASE_SAMPLE}`",
        f"- Derived seismic sample table: `{SAMPLE_OUT}`",
        f"- Current final Conventional factors: `{V3_FACTOR_LIST}`",
        f"- Number of samples: `{len(df)}`",
        f"- Positive samples: `{int(df['label'].sum())}`",
        f"- Negative samples: `{int((df['label'] == 0).sum())}`",
        "",
        "## Added Seismic Factors",
        "",
        "| Factor | Missing % | Min | Max | Mean | P90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in sample_qa.iterrows():
        lines.append(
            f"| {row['factor_name']} (`{row['factor']}`) | {row['missing_percent']:.2f} | "
            f"{row['min']:.4g} | {row['max']:.4g} | {row['mean']:.4g} | {row['p90']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Screened Seismic Factors Used In Main Ablation",
            "",
        ]
    )
    for factor in SCREENED_SEISMIC:
        lines.append(f"- {SEISMIC_NAMES[factor]} (`{factor}`)")

    lines.extend(
        [
            "",
            "The all-seismic set is retained as a diagnostic comparison, while the screened set avoids duplicating paired variables such as density versus magnitude-weighted density and PGA versus MMI.",
            "",
            "## VIF Diagnostics",
            "",
            f"- All-seismic VIF table: `{OUT_DIR / 'conventional_plus_all_seismic_vif.csv'}`",
            f"- Screened-seismic VIF table: `{OUT_DIR / 'conventional_plus_screened_seismic_vif.csv'}`",
            f"- Largest VIF with all seismic factors: `{float(all_vif['vif'].replace(np.inf, np.nan).max()):.2f}`",
            f"- Largest VIF with screened seismic factors: `{float(screened_vif['vif'].replace(np.inf, np.nan).max()):.2f}`",
            "",
            "## Spatial-CV Stacked Ensemble Results",
            "",
            "| Feature set | ROC-AUC | PR-AUC | Balanced accuracy | F1 | Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in stack_summary.sort_values("pr_auc_mean", ascending=False).iterrows():
        lines.append(
            f"| {row['feature_set_name']} | {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} | "
            f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f} | "
            f"{row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_std']:.3f} | "
            f"{row['f1_mean']:.3f} +/- {row['f1_std']:.3f} | "
            f"{row['brier_mean']:.3f} +/- {row['brier_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Saved Outputs",
            "",
            f"- Model/output folder: `{OUT_DIR}`",
            f"- Figures: `{FIG_DIR}`",
            f"- Stacked summary with deltas: `{OUT_DIR / 'seismic_ablation_stacked_summary_with_deltas.csv'}`",
            f"- Base and stacked predictions: `{OUT_DIR / 'seismic_ablation_spatial_cv_predictions.csv'}`",
            f"- Fold metrics: `{OUT_DIR / 'seismic_ablation_fold_metrics.csv'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = build_seismic_sample_table()
    conventional = pd.read_csv(V3_FACTOR_LIST)["factor"].tolist()
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    numeric = sorted(set(conventional + alpha + list(SEISMIC_RASTERS) + ["label", "spatial_fold_5"]))
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)

    run_multicollinearity(df, conventional)
    fsets = feature_sets(conventional, alpha)

    pd.DataFrame(
        [
            {
                "feature_set": key,
                "feature_set_name": cfg["display"],
                "feature_count": len(cfg["features"]),
                "features": ";".join(cfg["features"]),
            }
            for key, cfg in fsets.items()
        ]
    ).to_csv(OUT_DIR / "seismic_ablation_feature_sets.csv", index=False, encoding="utf-8-sig")

    base_pred, base_metrics = run_base_models(df, fsets)
    stacked_pred, stacked_metrics, weights = run_stacking(base_pred, fsets)
    all_pred = pd.concat([base_pred, stacked_pred], ignore_index=True)
    all_metrics = pd.concat([base_metrics, stacked_metrics], ignore_index=True)
    summary = summarize(all_metrics)

    base_pred.to_csv(OUT_DIR / "seismic_ablation_base_oof_predictions.csv", index=False, encoding="utf-8-sig")
    stacked_pred.to_csv(OUT_DIR / "seismic_ablation_stacked_oof_predictions.csv", index=False, encoding="utf-8-sig")
    all_pred.to_csv(OUT_DIR / "seismic_ablation_spatial_cv_predictions.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(OUT_DIR / "seismic_ablation_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "seismic_ablation_model_summary.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT_DIR / "seismic_ablation_meta_coefficients.csv", index=False, encoding="utf-8-sig")
    write_stacked_figures(all_pred, summary)
    write_report(df, conventional, fsets, summary)

    run_summary = {
        "input_table": str(BASE_SAMPLE),
        "derived_table": str(SAMPLE_OUT),
        "rows": int(len(df)),
        "feature_sets": {str(cfg["display"]): len(cfg["features"]) for cfg in fsets.values()},
        "base_models": [MODEL_NAMES[k] for k in BASE_MODEL_KEYS],
        "meta_model": MODEL_NAMES["stacked_l2_logistic"],
        "validation": "5-fold spatial block CV using spatial_fold_5 and out-of-fold base probabilities for meta-learning",
        "best_stacked_by_pr_auc": summary[summary["model"] == "stacked_l2_logistic"].iloc[0].to_dict(),
    }
    (OUT_DIR / "seismic_ablation_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(summary[summary["model"] == "stacked_l2_logistic"].to_string(index=False))
    log(str(REPORT))


if __name__ == "__main__":
    main()
