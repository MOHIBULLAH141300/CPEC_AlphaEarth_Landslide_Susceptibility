"""Run 2018 CPEC meta-ensemble upgrade.

This script adds:
- CatBoost as an additional tree-boosting learner.
- A spatial-CV stacked/meta-learning model.

It writes to a separate folder so the already checked 2018 baseline package
remains intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.impute import SimpleImputer

from run_2018_reduced_strategy import (
    CONVENTIONAL_REDUCED,
    DATA,
    FACTOR_NAMES,
    FEATURE_SET_NAMES,
    MODEL_NAMES,
    PROJECT_ROOT,
    load_data,
    make_models,
    predict_positive,
    threshold_for_f1,
)


OUT_DIR = PROJECT_ROOT / "03_models" / "meta_ensemble_2018_spatial_cv"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "meta_ensemble_2018_modelling_report.md"
CURVE_REPORT = PROJECT_ROOT / "05_reports" / "meta_ensemble_2018_roc_pr_curves.md"

BASE_MODEL_KEYS = ["logistic_l2", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]

PUBLIC_MODEL_NAMES = {
    **MODEL_NAMES,
    "catboost": "CatBoost",
    "stacked_l2_logistic": "Spatial-CV Stacked Ensemble",
}


def catboost_factory(seed: int = 141300) -> CatBoostClassifier:
    return CatBoostClassifier(
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
    )


def all_model_factories(seed: int = 141300) -> dict[str, Callable[[], object]]:
    models = make_models(seed)
    models["catboost"] = lambda: catboost_factory(seed)
    return models


def feature_sets(alpha: list[str]) -> dict[str, list[str]]:
    return {
        "conventional_reduced": CONVENTIONAL_REDUCED,
        "alphaearth_only": alpha,
        "fused_reduced": CONVENTIONAL_REDUCED + alpha,
    }


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


def fold_summary(metrics: pd.DataFrame) -> pd.DataFrame:
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


def run_base_oof(df: pd.DataFrame, alpha: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = all_model_factories()
    y = df["label"].astype(int).to_numpy()
    folds = sorted(df["spatial_fold_5"].unique())
    pred_frames = []
    metric_rows = []

    for fs_key, cols in feature_sets(alpha).items():
        x = df[cols]
        for model_key in BASE_MODEL_KEYS:
            factory = models[model_key]
            for fold in folds:
                train_idx = df["spatial_fold_5"] != fold
                test_idx = df["spatial_fold_5"] == fold
                model = factory()
                model.fit(x.loc[train_idx], y[train_idx.to_numpy()])
                train_prob = predict_positive(model, x.loc[train_idx])
                threshold = threshold_for_f1(y[train_idx.to_numpy()], train_prob)
                prob = predict_positive(model, x.loc[test_idx])
                y_test = y[test_idx.to_numpy()]
                metric_rows.append(
                    {
                        "feature_set": fs_key,
                        "feature_set_name": FEATURE_SET_NAMES[fs_key],
                        "model": model_key,
                        "model_name": PUBLIC_MODEL_NAMES[model_key],
                        "fold": int(fold),
                        "n_train": int(train_idx.sum()),
                        "n_test": int(test_idx.sum()),
                        "n_test_pos": int(y_test.sum()),
                        "n_test_neg": int((y_test == 0).sum()),
                        **metric_row(y_test, prob, threshold),
                    }
                )
                pred = df.loc[
                    test_idx,
                    ["inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"],
                ].copy()
                pred["sample_id"] = pred.index.astype(int)
                pred["feature_set"] = fs_key
                pred["feature_set_name"] = FEATURE_SET_NAMES[fs_key]
                pred["model"] = model_key
                pred["model_name"] = PUBLIC_MODEL_NAMES[model_key]
                pred["probability"] = prob
                pred["threshold"] = threshold
                pred["prediction"] = (prob >= threshold).astype(int)
                pred_frames.append(pred)

            final_model = factory()
            final_model.fit(x, y)
            joblib.dump(final_model, OUT_DIR / f"model_{fs_key}_{model_key}.joblib")

    return pd.concat(pred_frames, ignore_index=True), pd.DataFrame(metric_rows)


def make_meta_matrix(oof: pd.DataFrame, fs_key: str) -> pd.DataFrame:
    d = oof[oof["feature_set"] == fs_key].copy()
    wide = d.pivot_table(
        index="sample_id",
        columns="model",
        values="probability",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    meta = d[
        ["sample_id", "inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"]
    ].drop_duplicates("sample_id")
    return meta.merge(wide, on="sample_id", how="inner")


def run_stacking(df: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = sorted(df["spatial_fold_5"].unique())
    stacked_preds = []
    stacked_metrics = []
    weight_rows = []

    for fs_key in FEATURE_SET_NAMES:
        wide = make_meta_matrix(oof, fs_key)
        meta_cols = BASE_MODEL_KEYS
        for fold in folds:
            train_idx = wide["spatial_fold_5"] != fold
            test_idx = wide["spatial_fold_5"] == fold
            x_train = wide.loc[train_idx, meta_cols]
            y_train = wide.loc[train_idx, "label"].astype(int).to_numpy()
            x_test = wide.loc[test_idx, meta_cols]
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
                    "feature_set_name": FEATURE_SET_NAMES[fs_key],
                    "model": "stacked_l2_logistic",
                    "model_name": PUBLIC_MODEL_NAMES["stacked_l2_logistic"],
                    "fold": int(fold),
                    "n_train": int(train_idx.sum()),
                    "n_test": int(test_idx.sum()),
                    "n_test_pos": int(y_test.sum()),
                    "n_test_neg": int((y_test == 0).sum()),
                    **metric_row(y_test, prob, threshold),
                }
            )
            pred = wide.loc[test_idx, ["sample_id", "inventory_id", "label", "spatial_fold_5", "hazard_type", "use_role", "longitude", "latitude"]].copy()
            pred["feature_set"] = fs_key
            pred["feature_set_name"] = FEATURE_SET_NAMES[fs_key]
            pred["model"] = "stacked_l2_logistic"
            pred["model_name"] = PUBLIC_MODEL_NAMES["stacked_l2_logistic"]
            pred["probability"] = prob
            pred["threshold"] = threshold
            pred["prediction"] = (prob >= threshold).astype(int)
            stacked_preds.append(pred)

            coefs = meta.named_steps["model"].coef_.ravel()
            for col, coef in zip(meta_cols, coefs):
                weight_rows.append(
                    {
                        "feature_set": fs_key,
                        "feature_set_name": FEATURE_SET_NAMES[fs_key],
                        "fold": int(fold),
                        "base_model": col,
                        "base_model_name": PUBLIC_MODEL_NAMES[col],
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
        final_meta.fit(wide[meta_cols], wide["label"].astype(int).to_numpy())
        joblib.dump(final_meta, OUT_DIR / f"model_{fs_key}_stacked_l2_logistic.joblib")

    return pd.concat(stacked_preds, ignore_index=True), pd.DataFrame(stacked_metrics), pd.DataFrame(weight_rows)


def write_reports(summary: pd.DataFrame, weights: pd.DataFrame, all_predictions: pd.DataFrame) -> None:
    def fmt(mean: float, std: float) -> str:
        return f"{float(mean):.3f} +/- {float(std):.3f}"

    lines = [
        "# 2018 Meta-Ensemble Modelling Upgrade",
        "",
        "This report adds CatBoost and a true spatial-CV stacked/meta-learning model to the 2018 CPEC susceptibility benchmark.",
        "",
        "## Design",
        "",
        "- Base learners: Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost.",
        "- Meta-learner: L2 Logistic Regression.",
        "- Meta-learning inputs: spatial out-of-fold probabilities from base learners.",
        "- Validation: 5-fold spatial block cross-validation using `spatial_fold_5`.",
        "",
        "## Top Results By Mean PR-AUC",
        "",
        "| Rank | Feature set | Model | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(summary.head(15).iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature_set_name']} | {row['model_name']} | "
            f"{fmt(row['pr_auc_mean'], row['pr_auc_std'])} | "
            f"{fmt(row['roc_auc_mean'], row['roc_auc_std'])} | "
            f"{fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} | "
            f"{fmt(row['f1_mean'], row['f1_std'])} | "
            f"{fmt(row['brier_mean'], row['brier_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Extra Trees is included as a high-randomization tree ensemble for noisy, high-dimensional, and spatially heterogeneous predictors.",
            "- CatBoost adds an independent ordered-boosting style benchmark to XGBoost and LightGBM.",
            "- The stacked model is trained from out-of-fold probabilities, so each meta-training value was generated by a base model that did not train on that sample.",
            "- The final map model should be selected only if stacking improves performance and calibration without making interpretation weaker.",
            "",
            "## Saved Outputs",
            "",
            f"- Fold metrics: `{OUT_DIR / 'meta_ensemble_2018_fold_metrics.csv'}`",
            f"- Ranked summary: `{OUT_DIR / 'meta_ensemble_2018_summary.csv'}`",
            f"- OOF predictions: `{OUT_DIR / 'meta_ensemble_2018_spatial_cv_predictions.csv'}`",
            f"- Meta coefficients: `{OUT_DIR / 'meta_ensemble_2018_meta_coefficients.csv'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    key = [
        ("conventional_reduced", "stacked_l2_logistic"),
        ("alphaearth_only", "stacked_l2_logistic"),
        ("fused_reduced", "stacked_l2_logistic"),
        ("fused_reduced", "xgboost"),
        ("fused_reduced", "catboost"),
        ("fused_reduced", "extra_trees"),
    ]
    plot_curves(all_predictions, key)

    coeff_summary = (
        weights.groupby(["feature_set", "feature_set_name", "base_model", "base_model_name"])
        .agg(meta_coefficient_mean=("meta_coefficient", "mean"), meta_coefficient_std=("meta_coefficient", "std"))
        .reset_index()
    )
    coeff_summary.to_csv(OUT_DIR / "meta_ensemble_2018_meta_coefficient_summary.csv", index=False)


def plot_curves(pred: pd.DataFrame, keep: list[tuple[str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    summary_rows = []

    plt.figure(figsize=(7.4, 6.2))
    for fs_key, model_key in keep:
        d = pred[(pred["feature_set"] == fs_key) & (pred["model"] == model_key)].copy()
        if d.empty:
            continue
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        label = f"{FEATURE_SET_NAMES[fs_key]} - {PUBLIC_MODEL_NAMES[model_key]} (AUC={roc_auc:.3f})"
        plt.plot(fpr, tpr, linewidth=2.0, label=label)
        for x, yy, th in zip(fpr, tpr, thresholds):
            curve_rows.append({"feature_set": fs_key, "model": model_key, "curve": "roc", "x": x, "y": yy, "threshold": th})
        summary_rows.append(
            {
                "feature_set": fs_key,
                "feature_set_name": FEATURE_SET_NAMES[fs_key],
                "model": model_key,
                "model_name": PUBLIC_MODEL_NAMES[model_key],
                "roc_auc": roc_auc,
                "pr_auc": average_precision_score(y, p),
                "n": len(d),
            }
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("2018 Meta-Ensemble Spatial-CV ROC Curves")
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_meta_ensemble_roc_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_meta_ensemble_roc_curves_2018.pdf")
    plt.close()

    plt.figure(figsize=(7.4, 6.2))
    for fs_key, model_key in keep:
        d = pred[(pred["feature_set"] == fs_key) & (pred["model"] == model_key)].copy()
        if d.empty:
            continue
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        label = f"{FEATURE_SET_NAMES[fs_key]} - {PUBLIC_MODEL_NAMES[model_key]} (AP={ap:.3f})"
        plt.plot(recall, precision, linewidth=2.0, label=label)
        for x, yy, th in zip(recall, precision, list(thresholds) + [float("nan")]):
            curve_rows.append({"feature_set": fs_key, "model": model_key, "curve": "precision_recall", "x": x, "y": yy, "threshold": th})
    prevalence = pred[pred["model"] == "xgboost"]["label"].astype(int).mean()
    plt.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2018 Meta-Ensemble Spatial-CV Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_meta_ensemble_pr_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_meta_ensemble_pr_curves_2018.pdf")
    plt.close()

    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "meta_ensemble_2018_curve_coordinates.csv", index=False)
    curve_summary = pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False)
    curve_summary.to_csv(OUT_DIR / "meta_ensemble_2018_curve_auc_summary.csv", index=False)

    lines = [
        "# 2018 Meta-Ensemble ROC And PR Curves",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC/AP | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in curve_summary.iterrows():
        lines.append(
            f"| {row['feature_set_name']} | {row['model_name']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            f"- ROC figure: `{FIG_DIR / 'figure_meta_ensemble_roc_curves_2018.png'}`",
            f"- PR figure: `{FIG_DIR / 'figure_meta_ensemble_pr_curves_2018.png'}`",
        ]
    )
    CURVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, alpha = load_data()
    base_pred, base_metrics = run_base_oof(df, alpha)
    stacked_pred, stacked_metrics, weights = run_stacking(df, base_pred)
    all_predictions = pd.concat([base_pred, stacked_pred], ignore_index=True)
    all_metrics = pd.concat([base_metrics, stacked_metrics], ignore_index=True)
    summary = fold_summary(all_metrics)

    base_pred.to_csv(OUT_DIR / "meta_ensemble_2018_base_oof_predictions.csv", index=False)
    stacked_pred.to_csv(OUT_DIR / "meta_ensemble_2018_stacked_oof_predictions.csv", index=False)
    all_predictions.to_csv(OUT_DIR / "meta_ensemble_2018_spatial_cv_predictions.csv", index=False)
    all_metrics.to_csv(OUT_DIR / "meta_ensemble_2018_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "meta_ensemble_2018_summary.csv", index=False)
    weights.to_csv(OUT_DIR / "meta_ensemble_2018_meta_coefficients.csv", index=False)
    write_reports(summary, weights, all_predictions)

    run_summary = {
        "input_table": str(DATA),
        "rows": int(len(df)),
        "feature_sets": {FEATURE_SET_NAMES[k]: len(v) for k, v in feature_sets(alpha).items()},
        "base_models": [PUBLIC_MODEL_NAMES[k] for k in BASE_MODEL_KEYS],
        "meta_model": PUBLIC_MODEL_NAMES["stacked_l2_logistic"],
        "validation": "5-fold spatial block CV using spatial_fold_5 and out-of-fold base probabilities for meta-learning",
        "best_by_pr_auc": summary.iloc[0].to_dict(),
    }
    (OUT_DIR / "meta_ensemble_2018_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
