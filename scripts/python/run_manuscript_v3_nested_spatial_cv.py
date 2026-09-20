"""Leakage-resistant nested spatial-CV stacked ensemble for manuscript v3.

The outer spatial folds estimate generalisation. Within each outer-training set,
inner spatial folds generate base-learner predictions for the meta-learner. A
20 km exclusion buffer is applied between every training and validation set.
Categorical predictors are one-hot encoded and aspect is represented by sine
and cosine components. Reported values are case-control susceptibility scores,
not population-calibrated landslide occurrence probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = (
    PROJECT_ROOT
    / "01_clean_data"
    / "manuscript_v3_corrected_terrain"
    / "cpec_baseline_samples_corrected_terrain_lithology.csv"
)
FACTOR_LIST = (
    PROJECT_ROOT
    / "03_models"
    / "manuscript_v3_multicollinearity"
    / "final_conventional_factors.csv"
)
OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_nested_spatial_cv"

SEED = 141300
EARTH_RADIUS_KM = 6371.0088
OUTER_BUFFER_KM = 20.0
INNER_BUFFER_KM = 20.0

FEATURE_SET_NAMES = {
    "conventional": "Conventional",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_alphaearth_embeddings": "Conventional + AlphaEarth Embeddings",
}

MODEL_NAMES = {
    "logistic_l2": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
BASE_MODEL_KEYS = list(MODEL_NAMES)
CATEGORICAL_FACTORS = ["modis_lc_type1", "lithology_code", "soil_type", "eq_density_ms5"]


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    display_name: str
    continuous: list[str]
    categorical: list[str]

    @property
    def columns(self) -> list[str]:
        return self.continuous + self.categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--outer-buffer-km", type=float, default=OUTER_BUFFER_KM)
    parser.add_argument("--inner-buffer-km", type=float, default=INNER_BUFFER_KM)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Use smaller ensembles for a smoke test.")
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, dict[str, FeatureSpec], list[str]]:
    df = pd.read_csv(DATA, encoding="utf-8-sig").reset_index(drop=True)
    conventional_raw = pd.read_csv(FACTOR_LIST)["factor"].astype(str).tolist()
    required = {
        "label",
        "spatial_fold_5",
        "spatial_block_1deg",
        "longitude",
        "latitude",
        "hazard_type",
        "cpec_admin_transfer_domain",
    }
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    alpha = sorted(c for c in df.columns if c.startswith("A") and c[1:].isdigit())
    numeric = set(conventional_raw + alpha + ["label", "spatial_fold_5", "longitude", "latitude"])
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5", "longitude", "latitude"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)

    theta = np.deg2rad(df["aspect_deg"].astype(float))
    df["aspect_sin"] = np.sin(theta)
    df["aspect_cos"] = np.cos(theta)
    conventional = [c for c in conventional_raw if c != "aspect_deg"] + ["aspect_sin", "aspect_cos"]
    categorical = [c for c in CATEGORICAL_FACTORS if c in conventional]
    continuous = [c for c in conventional if c not in categorical]

    specs = {
        "conventional": FeatureSpec(
            "conventional", FEATURE_SET_NAMES["conventional"], continuous, categorical
        ),
        "alphaearth_embeddings": FeatureSpec(
            "alphaearth_embeddings", FEATURE_SET_NAMES["alphaearth_embeddings"], alpha, []
        ),
        "conventional_alphaearth_embeddings": FeatureSpec(
            "conventional_alphaearth_embeddings",
            FEATURE_SET_NAMES["conventional_alphaearth_embeddings"],
            continuous + alpha,
            categorical,
        ),
    }
    return df.reset_index(drop=True), specs, conventional_raw


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if spec.continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                spec.continuous,
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                spec.categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def model_factories(spec: FeatureSpec, quick: bool = False) -> dict[str, Callable[[], Pipeline]]:
    n_small = 80 if quick else 350
    n_large = 100 if quick else 450

    def pipe(model: object) -> Pipeline:
        return Pipeline([("preprocess", make_preprocessor(spec)), ("model", model)])

    return {
        "logistic_l2": lambda: pipe(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                random_state=SEED,
            )
        ),
        "random_forest": lambda: pipe(
            RandomForestClassifier(
                n_estimators=n_small,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            )
        ),
        "extra_trees": lambda: pipe(
            ExtraTreesClassifier(
                n_estimators=n_large,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            )
        ),
        "xgboost": lambda: pipe(
            XGBClassifier(
                n_estimators=n_small,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "lightgbm": lambda: pipe(
            LGBMClassifier(
                n_estimators=n_small,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
        ),
        "catboost": lambda: pipe(
            CatBoostClassifier(
                iterations=n_large,
                depth=5,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
        ),
    }


def make_meta_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def buffered_indices(
    df: pd.DataFrame, candidate_idx: np.ndarray, validation_idx: np.ndarray, buffer_km: float
) -> np.ndarray:
    if buffer_km <= 0 or not len(candidate_idx) or not len(validation_idx):
        return candidate_idx
    val_coords = np.deg2rad(df.loc[validation_idx, ["latitude", "longitude"]].to_numpy(float))
    train_coords = np.deg2rad(df.loc[candidate_idx, ["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(val_coords, metric="haversine")
    distance_km = tree.query(train_coords, k=1, return_distance=True)[0].ravel() * EARTH_RADIUS_KM
    return candidate_idx[distance_km >= buffer_km]


def predict_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def threshold_for_f1(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def expected_calibration_error(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if use.any():
            ece += use.mean() * abs(float(y[use].mean()) - float(score[use].mean()))
    return float(ece if total else math.nan)


def metric_row(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, np.clip(score, 1e-8, 1 - 1e-8))),
        "ece_10bin": expected_calibration_error(y, score),
        "threshold": float(threshold),
    }


def pooled_metric_row(
    y: np.ndarray, score: np.ndarray, prediction: np.ndarray, threshold: float
) -> dict[str, float]:
    """Aggregate outer-fold results while retaining fold-specific decisions."""
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, np.clip(score, 1e-8, 1 - 1e-8))),
        "ece_10bin": expected_calibration_error(y, score),
        "threshold": float(threshold),
    }


def nested_feature_set(
    df: pd.DataFrame,
    spec: FeatureSpec,
    outer_buffer_km: float,
    inner_buffer_km: float,
    quick: bool,
    fit_final_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    y = df["label"].to_numpy(int)
    folds = df["spatial_fold_5"].to_numpy(int)
    all_idx = np.arange(len(df))
    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    factories = model_factories(spec, quick=quick)

    for outer_fold in sorted(np.unique(folds)):
        test_idx = all_idx[folds == outer_fold]
        raw_outer_train = all_idx[folds != outer_fold]
        outer_train = buffered_indices(df, raw_outer_train, test_idx, outer_buffer_km)
        inner_fold_values = sorted(np.unique(folds[outer_train]))
        inner_oof = np.full((len(outer_train), len(BASE_MODEL_KEYS)), np.nan, dtype=float)
        position = {sample_idx: pos for pos, sample_idx in enumerate(outer_train)}

        for inner_fold in inner_fold_values:
            inner_val = outer_train[folds[outer_train] == inner_fold]
            inner_candidates = outer_train[folds[outer_train] != inner_fold]
            inner_train = buffered_indices(df, inner_candidates, inner_val, inner_buffer_km)
            if len(np.unique(y[inner_train])) != 2 or len(np.unique(y[inner_val])) != 2:
                raise RuntimeError(
                    f"Class missing in feature set {spec.key}, outer {outer_fold}, inner {inner_fold}."
                )
            for model_col, model_key in enumerate(BASE_MODEL_KEYS):
                model = factories[model_key]()
                model.fit(df.loc[inner_train, spec.columns], y[inner_train])
                pred = predict_positive(model, df.loc[inner_val, spec.columns])
                rows = [position[i] for i in inner_val]
                inner_oof[rows, model_col] = pred

        if np.isnan(inner_oof).any():
            raise RuntimeError(f"Incomplete inner OOF matrix for {spec.key}, outer fold {outer_fold}.")

        meta = make_meta_model()
        meta.fit(inner_oof, y[outer_train])
        meta_train_score = predict_positive(meta, inner_oof)
        threshold = threshold_for_f1(y[outer_train], meta_train_score)

        outer_base = np.empty((len(test_idx), len(BASE_MODEL_KEYS)), dtype=float)
        for model_col, model_key in enumerate(BASE_MODEL_KEYS):
            model = factories[model_key]()
            model.fit(df.loc[outer_train, spec.columns], y[outer_train])
            outer_base[:, model_col] = predict_positive(model, df.loc[test_idx, spec.columns])
        outer_score = predict_positive(meta, outer_base)

        fold_rows.append(
            {
                "feature_set": spec.key,
                "feature_set_name": spec.display_name,
                "outer_fold": int(outer_fold),
                "n_train_before_buffer": int(len(raw_outer_train)),
                "n_train_after_buffer": int(len(outer_train)),
                "n_test": int(len(test_idx)),
                "n_test_positive": int(y[test_idx].sum()),
                "n_test_negative": int((y[test_idx] == 0).sum()),
                "outer_buffer_km": float(outer_buffer_km),
                "inner_buffer_km": float(inner_buffer_km),
                **metric_row(y[test_idx], outer_score, threshold),
            }
        )

        coef = meta.named_steps["model"].coef_.ravel()
        for model_key, value in zip(BASE_MODEL_KEYS, coef):
            coefficient_rows.append(
                {
                    "feature_set": spec.key,
                    "feature_set_name": spec.display_name,
                    "outer_fold": int(outer_fold),
                    "base_model": model_key,
                    "base_model_name": MODEL_NAMES[model_key],
                    "meta_coefficient": float(value),
                }
            )

        out = df.loc[
            test_idx,
            [
                "inventory_id",
                "label",
                "hazard_type",
                "spatial_block_1deg",
                "spatial_fold_5",
                "longitude",
                "latitude",
                "cpec_admin_transfer_domain",
            ],
        ].copy()
        out["sample_id"] = test_idx
        out["feature_set"] = spec.key
        out["feature_set_name"] = spec.display_name
        for model_col, model_key in enumerate(BASE_MODEL_KEYS):
            out[f"base_{model_key}_score"] = outer_base[:, model_col]
        out["stacked_score"] = outer_score
        out["threshold"] = threshold
        out["prediction"] = (outer_score >= threshold).astype(int)
        prediction_rows.append(out)

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("sample_id")
    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)

    bundle = None
    if fit_final_models:
        # The deployable meta-learner uses the complete outer OOF base matrix.
        oof_base_cols = [f"base_{key}_score" for key in BASE_MODEL_KEYS]
        final_meta = make_meta_model()
        final_meta.fit(predictions[oof_base_cols].to_numpy(float), predictions["label"].to_numpy(int))
        final_models: dict[str, object] = {}
        for model_key in BASE_MODEL_KEYS:
            model = factories[model_key]()
            model.fit(df[spec.columns], y)
            final_models[model_key] = model
        bundle = {
            "feature_set": spec.key,
            "feature_set_name": spec.display_name,
            "feature_spec": {
                "key": spec.key,
                "display_name": spec.display_name,
                "continuous": spec.continuous,
                "categorical": spec.categorical,
                "columns": spec.columns,
            },
            "base_model_order": BASE_MODEL_KEYS,
            "base_models": final_models,
            "meta_model": final_meta,
            "score_definition": "case-control susceptibility score",
            "outer_buffer_km": outer_buffer_km,
            "inner_buffer_km": inner_buffer_km,
            "random_seed": SEED,
        }
    return predictions, fold_metrics, coefficients, bundle


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fs_key, d in predictions.groupby("feature_set", sort=False):
        work = d.copy()
        work["bin"] = pd.qcut(work["stacked_score"], q=bins, duplicates="drop")
        for rank, (_, b) in enumerate(work.groupby("bin", observed=True), start=1):
            rows.append(
                {
                    "feature_set": fs_key,
                    "feature_set_name": b["feature_set_name"].iloc[0],
                    "bin": rank,
                    "n": len(b),
                    "mean_score": b["stacked_score"].mean(),
                    "observed_positive_fraction": b["label"].mean(),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metric_intervals(
    predictions: pd.DataFrame, repetitions: int, seed: int = SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]
    for fs_key, d in predictions.groupby("feature_set", sort=False):
        blocks = d["spatial_block_1deg"].astype(str).unique()
        block_rows = {b: np.flatnonzero(d["spatial_block_1deg"].astype(str).to_numpy() == b) for b in blocks}
        values = {m: [] for m in metric_names}
        y_all = d["label"].to_numpy(int)
        score_all = d["stacked_score"].to_numpy(float)
        prediction_all = d["prediction"].to_numpy(int)
        for _ in range(repetitions):
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            idx = np.concatenate([block_rows[b] for b in sampled])
            y = y_all[idx]
            score = score_all[idx]
            prediction = prediction_all[idx]
            if len(np.unique(y)) < 2:
                continue
            values["roc_auc"].append(roc_auc_score(y, score))
            values["pr_auc"].append(average_precision_score(y, score))
            values["balanced_accuracy"].append(balanced_accuracy_score(y, prediction))
            values["f1"].append(f1_score(y, prediction, zero_division=0))
            values["brier"].append(brier_score_loss(y, score))
        point = {
            "roc_auc": roc_auc_score(y_all, score_all),
            "pr_auc": average_precision_score(y_all, score_all),
            "balanced_accuracy": balanced_accuracy_score(y_all, prediction_all),
            "f1": f1_score(y_all, prediction_all, zero_division=0),
            "brier": brier_score_loss(y_all, score_all),
        }
        for metric in metric_names:
            arr = np.asarray(values[metric], dtype=float)
            rows.append(
                {
                    "feature_set": fs_key,
                    "feature_set_name": d["feature_set_name"].iloc[0],
                    "metric": metric,
                    "estimate": point[metric],
                    "ci95_low": np.quantile(arr, 0.025),
                    "ci95_high": np.quantile(arr, 0.975),
                    "bootstrap_repetitions": len(arr),
                    "resampling_unit": "1-degree spatial block",
                }
            )
    return pd.DataFrame(rows)


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fs_key, d in predictions.groupby("feature_set", sort=False):
        y = d["label"].to_numpy(int)
        score = d["stacked_score"].to_numpy(float)
        prediction = d["prediction"].to_numpy(int)
        threshold = float(np.median(d["threshold"]))
        rows.append(
            {
                "feature_set": fs_key,
                "feature_set_name": d["feature_set_name"].iloc[0],
                "n": len(d),
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
                **pooled_metric_row(y, score, prediction, threshold),
            }
        )
    return pd.DataFrame(rows)


def domain_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fs_key, domain), d in predictions.groupby(
        ["feature_set", "cpec_admin_transfer_domain"], sort=False
    ):
        y = d["label"].to_numpy(int)
        score = d["stacked_score"].to_numpy(float)
        prediction = d["prediction"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        threshold = float(np.median(d["threshold"]))
        rows.append(
            {
                "feature_set": fs_key,
                "feature_set_name": d["feature_set_name"].iloc[0],
                "domain": domain,
                "n": len(d),
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
                **pooled_metric_row(y, score, prediction, threshold),
            }
        )
    return pd.DataFrame(rows)


def write_run_report(
    out_dir: Path,
    df: pd.DataFrame,
    specs: dict[str, FeatureSpec],
    pooled: pd.DataFrame,
    intervals: pd.DataFrame,
    conventional_raw: list[str],
    elapsed_seconds: float,
) -> None:
    ci_lookup = intervals.set_index(["feature_set", "metric"])
    lines = [
        "# Manuscript v3 nested spatial-CV analysis",
        "",
        "## Design freeze",
        "",
        f"- Samples: {len(df)} ({int(df.label.sum())} positives; {int((df.label == 0).sum())} controls).",
        "- Validation: five outer one-degree spatial-block folds.",
        "- Stacking: base learners generated within outer-training data using inner spatial folds.",
        f"- Separation buffer: {OUTER_BUFFER_KM:.0f} km between training and validation samples.",
        "- Aspect: sine and cosine components.",
        "- Land cover, lithology and soil: categorical one-hot encoding.",
        "- Metric uncertainty: 95% confidence intervals from spatial-block bootstrap.",
        "- Output interpretation: case-control susceptibility score, not population occurrence probability.",
        "",
        "## Conventional predictor source fields",
        "",
    ]
    lines.extend(f"- `{factor}`" for factor in conventional_raw)
    lines.extend(
        [
            "",
            "## Primary stacked-ensemble performance",
            "",
            "| Feature set | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | F1 | Balanced accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in pooled.iterrows():
        roc = ci_lookup.loc[(row.feature_set, "roc_auc")]
        pr = ci_lookup.loc[(row.feature_set, "pr_auc")]
        lines.append(
            f"| {row.feature_set_name} | {row.roc_auc:.3f} ({roc.ci95_low:.3f}-{roc.ci95_high:.3f}) | "
            f"{row.pr_auc:.3f} ({pr.ci95_low:.3f}-{pr.ci95_high:.3f}) | {row.brier:.3f} | "
            f"{row.f1:.3f} | {row.balanced_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Python: {platform.python_version()}",
            f"- Runtime: {elapsed_seconds / 60:.1f} minutes",
            f"- Random seed: {SEED}",
            f"- Feature sets: {', '.join(spec.display_name for spec in specs.values())}",
        ]
    )
    (out_dir / "README_nested_spatial_cv_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    df, specs, conventional_raw = load_data()

    all_predictions: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    for spec in specs.values():
        print(f"Running {spec.display_name}...", flush=True)
        predictions, fold_metrics, coefficients, bundle = nested_feature_set(
            df,
            spec,
            outer_buffer_km=args.outer_buffer_km,
            inner_buffer_km=args.inner_buffer_km,
            quick=args.quick,
        )
        all_predictions.append(predictions)
        all_fold_metrics.append(fold_metrics)
        all_coefficients.append(coefficients)
        if bundle is not None:
            joblib.dump(bundle, model_dir / f"nested_spatial_stack_{spec.key}.joblib", compress=3)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    pooled = pooled_metrics(predictions)
    intervals = bootstrap_metric_intervals(predictions, repetitions=args.bootstrap_reps)
    calibration = calibration_table(predictions)
    domains = domain_metrics(predictions)

    predictions.to_csv(args.out_dir / "nested_spatial_cv_predictions.csv", index=False)
    fold_metrics.to_csv(args.out_dir / "nested_spatial_cv_fold_metrics.csv", index=False)
    coefficients.to_csv(args.out_dir / "nested_spatial_cv_meta_coefficients.csv", index=False)
    pooled.to_csv(args.out_dir / "nested_spatial_cv_pooled_metrics.csv", index=False)
    intervals.to_csv(args.out_dir / "nested_spatial_cv_block_bootstrap_ci.csv", index=False)
    calibration.to_csv(args.out_dir / "nested_spatial_cv_calibration_bins.csv", index=False)
    domains.to_csv(args.out_dir / "nested_spatial_cv_domain_metrics.csv", index=False)

    manifest = {
        "analysis": "manuscript_v3_nested_spatial_cv",
        "input_table": str(DATA),
        "factor_list": str(FACTOR_LIST),
        "n_samples": len(df),
        "n_positive": int(df.label.sum()),
        "n_negative": int((df.label == 0).sum()),
        "outer_folds": sorted(df.spatial_fold_5.unique().astype(int).tolist()),
        "outer_buffer_km": args.outer_buffer_km,
        "inner_buffer_km": args.inner_buffer_km,
        "bootstrap_repetitions_requested": args.bootstrap_reps,
        "quick": args.quick,
        "random_seed": SEED,
        "feature_sets": {key: spec.columns for key, spec in specs.items()},
        "categorical_predictors": CATEGORICAL_FACTORS,
        "aspect_encoding": ["aspect_sin", "aspect_cos"],
        "base_models": MODEL_NAMES,
        "meta_model": "L2-regularised logistic regression",
        "score_definition": "case-control susceptibility score",
        "elapsed_seconds": time.time() - start,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_report(
        args.out_dir,
        df,
        specs,
        pooled,
        intervals,
        conventional_raw,
        time.time() - start,
    )
    print(pooled.to_string(index=False), flush=True)
    print(f"Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
