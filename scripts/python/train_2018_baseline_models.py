"""Train 2018 CPEC LSM baseline models with spatial-block CV.

Experiments:
- conventional_full
- conventional_reduced
- alphaearth_only
- fused_full
- fused_reduced

Models:
- regularized logistic regression
- random forest
- extra trees
- XGBoost
- LightGBM
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v2.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "baseline_2018_spatial_cv"
REPORT = PROJECT_ROOT / "05_reports" / "baseline_2018_modelling_report.md"

CONVENTIONAL_FULL = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_annual_total",
    "rain_monsoon_total",
    "rain_max_1day",
    "rain_max_3day",
    "rain_max_7day",
    "ndvi_median",
    "ndvi_max",
    "ndvi_amplitude",
    "evi_median",
    "lst_day_mean_c",
    "lst_day_max_c",
    "modis_lc_type1",
]

CONVENTIONAL_REDUCED = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rain_monsoon_total",
    "rain_max_1day",
    "ndvi_median",
    "ndvi_amplitude",
    "lst_day_mean_c",
    "modis_lc_type1",
]


def threshold_for_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    if len(thresholds) == 0:
        return 0.5
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
    }


def make_models(seed: int = 141300) -> dict[str, Callable[[], object]]:
    return {
        "logistic_l2": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": lambda: XGBClassifier(
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
        "lightgbm": lambda: LGBMClassifier(
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
    }


def predict_proba_positive(model, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def feature_importance(model, features: list[str]) -> pd.DataFrame:
    fitted = model
    if isinstance(model, Pipeline):
        fitted = model.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        vals = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        vals = np.abs(fitted.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": features, "importance": vals}).sort_values("importance", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    feature_sets = {
        "conventional_full": CONVENTIONAL_FULL,
        "conventional_reduced": CONVENTIONAL_REDUCED,
        "alphaearth_only": alpha,
        "fused_full": CONVENTIONAL_FULL + alpha,
        "fused_reduced": CONVENTIONAL_REDUCED + alpha,
    }

    for col in sorted(set().union(*[set(v) for v in feature_sets.values()])) + ["label", "spatial_fold_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)

    all_fold_metrics = []
    all_predictions = []
    all_importance = []
    model_factories = make_models()

    for feature_set_name, features in feature_sets.items():
        x_all = df[features]
        y_all = df["label"].to_numpy()
        folds = sorted(df["spatial_fold_5"].unique())
        for model_name, factory in model_factories.items():
            for fold in folds:
                train_idx = df["spatial_fold_5"] != fold
                test_idx = df["spatial_fold_5"] == fold
                x_train = x_all.loc[train_idx]
                y_train = y_all[train_idx.to_numpy()]
                x_test = x_all.loc[test_idx]
                y_test = y_all[test_idx.to_numpy()]

                model = factory()
                model.fit(x_train, y_train)
                train_prob = predict_proba_positive(model, x_train)
                threshold = threshold_for_f1(y_train, train_prob)
                test_prob = predict_proba_positive(model, x_test)
                row = {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold": int(fold),
                    "n_train": int(train_idx.sum()),
                    "n_test": int(test_idx.sum()),
                    "n_test_pos": int(y_test.sum()),
                    "n_test_neg": int((y_test == 0).sum()),
                    **metrics(y_test, test_prob, threshold),
                }
                all_fold_metrics.append(row)

                pred_df = df.loc[test_idx, ["inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"]].copy()
                pred_df["feature_set"] = feature_set_name
                pred_df["model"] = model_name
                pred_df["probability"] = test_prob
                pred_df["threshold"] = threshold
                pred_df["prediction"] = (test_prob >= threshold).astype(int)
                all_predictions.append(pred_df)

            # Fit on all data for feature importance and a reusable artifact.
            final_model = factory()
            final_model.fit(x_all, y_all)
            imp = feature_importance(final_model, features)
            if not imp.empty:
                imp["feature_set"] = feature_set_name
                imp["model"] = model_name
                all_importance.append(imp)
            joblib.dump(final_model, OUT_DIR / f"model_{feature_set_name}_{model_name}.joblib")

    fold_metrics = pd.DataFrame(all_fold_metrics)
    summary = (
        fold_metrics.groupby(["feature_set", "model"])
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

    fold_metrics.to_csv(OUT_DIR / "spatial_cv_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "spatial_cv_summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "spatial_cv_predictions.csv", index=False)
    if all_importance:
        pd.concat(all_importance, ignore_index=True).to_csv(OUT_DIR / "final_model_feature_importance.csv", index=False)

    best = summary.iloc[0].to_dict()
    run_summary = {
        "input_table": str(DATA),
        "rows": int(len(df)),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "models": list(model_factories.keys()),
        "validation": "5-fold spatial block CV using spatial_fold_5",
        "best_by_pr_auc": best,
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    top_lines = summary.head(10).copy()
    lines = [
        "# 2018 Baseline Spatial-CV Modelling Report",
        "",
        f"- Input table: `{DATA}`",
        f"- Rows: {len(df)}",
        "- Validation: 5-fold spatial block cross-validation",
        "- Models: logistic L2, Random Forest, Extra Trees, XGBoost, LightGBM",
        "- Feature sets: conventional full, conventional reduced, AlphaEarth Embeddings, fused full, Conventional + AlphaEarth Embeddings",
        "",
        "## Top Results By Mean PR-AUC",
        "",
        "| Rank | Feature set | Model | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top_lines.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature_set']} | {row['model']} | "
            f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f} | "
            f"{row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} | "
            f"{row['balanced_accuracy_mean']:.3f} | "
            f"{row['f1_mean']:.3f} | "
            f"{row['brier_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are first baseline results, not final manuscript results.",
            "- Scores are based on spatial folds, which are more conservative than random CV.",
            "- Conventional redundancy was handled by testing both full and conventional feature sets.",
            "- AlphaEarth bands were kept intact for the first fair comparison.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

