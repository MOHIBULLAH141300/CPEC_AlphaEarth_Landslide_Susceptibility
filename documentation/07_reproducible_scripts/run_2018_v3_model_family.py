"""Run corrected 2018 v3 model family after literature/multicollinearity gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(r"D:\DING PROJECT")
DATA = PROJECT_ROOT / "03_models" / "cpec_2018_lsm_samples_v3.csv"
V3_FACTOR_LIST = PROJECT_ROOT / "03_models" / "multicollinearity_assessment_2018_v3" / "v3_final_selected_conventional_factors.csv"
OUT_DIR = PROJECT_ROOT / "03_models" / "v3_model_family_2018_spatial_cv"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "v3_model_family_2018_modelling_report.md"
CURVE_REPORT = PROJECT_ROOT / "05_reports" / "v3_model_family_2018_roc_pr_curves.md"
STATUS_REPORT = PROJECT_ROOT / "05_reports" / "v3_model_family_status_2026-05-03.md"

FEATURE_SET_NAMES = {
    "conventional_v3": "Conventional v3",
    "alphaearth_embeddings": "AlphaEarth Embeddings",
    "conventional_v3_alphaearth_embeddings": "Conventional v3 + AlphaEarth Embeddings",
}

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


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    conventional = pd.read_csv(V3_FACTOR_LIST)["factor"].tolist()
    alpha = sorted([c for c in df.columns if c.startswith("A") and c[1:].isdigit()])
    numeric = sorted(set(conventional + alpha + ["label", "spatial_fold_5"]))
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label", "spatial_fold_5"]).copy()
    df["label"] = df["label"].astype(int)
    df["spatial_fold_5"] = df["spatial_fold_5"].astype(int)
    return df, conventional, alpha


def feature_sets(conventional: list[str], alpha: list[str]) -> dict[str, list[str]]:
    return {
        "conventional_v3": conventional,
        "alphaearth_embeddings": alpha,
        "conventional_v3_alphaearth_embeddings": conventional + alpha,
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


def run_base_models(df: pd.DataFrame, fsets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = make_models()
    y = df["label"].to_numpy()
    folds = sorted(df["spatial_fold_5"].unique())
    metrics = []
    preds = []
    for fs_key, cols in fsets.items():
        x = df[cols]
        for model_key, factory in models.items():
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
                        "feature_set_name": FEATURE_SET_NAMES[fs_key],
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
                p["feature_set_name"] = FEATURE_SET_NAMES[fs_key]
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


def run_stacking(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stacked_preds = []
    stacked_metrics = []
    weights = []
    for fs_key in FEATURE_SET_NAMES:
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
                    "feature_set_name": FEATURE_SET_NAMES[fs_key],
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
            p["feature_set_name"] = FEATURE_SET_NAMES[fs_key]
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
                        "feature_set_name": FEATURE_SET_NAMES[fs_key],
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


def write_curves(pred: pd.DataFrame) -> None:
    keep = [
        ("conventional_v3", "stacked_l2_logistic"),
        ("alphaearth_embeddings", "stacked_l2_logistic"),
        ("conventional_v3_alphaearth_embeddings", "stacked_l2_logistic"),
        ("conventional_v3_alphaearth_embeddings", "xgboost"),
        ("conventional_v3_alphaearth_embeddings", "catboost"),
        ("conventional_v3_alphaearth_embeddings", "extra_trees"),
    ]
    curve_rows = []
    summary_rows = []
    plt.figure(figsize=(7.4, 6.2))
    for fs_key, model_key in keep:
        d = pred[(pred["feature_set"] == fs_key) & (pred["model"] == model_key)]
        if d.empty:
            continue
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{FEATURE_SET_NAMES[fs_key]} - {MODEL_NAMES[model_key]} (AUC={roc_auc:.3f})")
        for x, yy, th in zip(fpr, tpr, thresholds):
            curve_rows.append({"feature_set": fs_key, "model": model_key, "curve": "roc", "x": x, "y": yy, "threshold": th})
        summary_rows.append(
            {
                "feature_set": fs_key,
                "feature_set_name": FEATURE_SET_NAMES[fs_key],
                "model": model_key,
                "model_name": MODEL_NAMES[model_key],
                "roc_auc": roc_auc,
                "pr_auc": average_precision_score(y, p),
                "n": len(d),
            }
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("2018 v3 Spatial-CV ROC Curves")
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_model_family_roc_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_v3_model_family_roc_curves_2018.pdf")
    plt.close()

    plt.figure(figsize=(7.4, 6.2))
    for fs_key, model_key in keep:
        d = pred[(pred["feature_set"] == fs_key) & (pred["model"] == model_key)]
        if d.empty:
            continue
        y = d["label"].astype(int).to_numpy()
        p = d["probability"].astype(float).to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(recall, precision, linewidth=2.0, label=f"{FEATURE_SET_NAMES[fs_key]} - {MODEL_NAMES[model_key]} (AP={ap:.3f})")
        for x, yy, th in zip(recall, precision, list(thresholds) + [float("nan")]):
            curve_rows.append({"feature_set": fs_key, "model": model_key, "curve": "precision_recall", "x": x, "y": yy, "threshold": th})
    prevalence = pred[pred["model"] == "xgboost"]["label"].astype(int).mean()
    plt.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2018 v3 Spatial-CV Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_v3_model_family_pr_curves_2018.png", dpi=300)
    plt.savefig(FIG_DIR / "figure_v3_model_family_pr_curves_2018.pdf")
    plt.close()

    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "v3_model_family_curve_coordinates.csv", index=False)
    curve_summary = pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False)
    curve_summary.to_csv(OUT_DIR / "v3_model_family_curve_auc_summary.csv", index=False)

    lines = [
        "# 2018 v3 ROC And Precision-Recall Curves",
        "",
        "| Feature set | Model | ROC-AUC | PR-AUC/AP | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in curve_summary.iterrows():
        lines.append(f"| {row['feature_set_name']} | {row['model_name']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {int(row['n'])} |")
    lines.extend(
        [
            "",
            f"- ROC figure: `{FIG_DIR / 'figure_v3_model_family_roc_curves_2018.png'}`",
            f"- PR figure: `{FIG_DIR / 'figure_v3_model_family_pr_curves_2018.png'}`",
        ]
    )
    CURVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: pd.DataFrame, conventional: list[str]) -> None:
    def fmt(mean: float, std: float) -> str:
        return f"{float(mean):.3f} +/- {float(std):.3f}"

    lines = [
        "# 2018 v3 Model Family Report",
        "",
        "## Literature And Diagnostics Gate",
        "",
        "- v3 uses the literature-supported corrected factor set.",
        "- Missing road, river/stream, fault, lithology, terrain morphology, soil, and seismic-background factors were added before modelling.",
        "- Final conventional v3 factors were selected after multicollinearity reduction; all final selected VIF values are below 4.",
        "",
        "## Conventional v3 Factors",
        "",
    ]
    for f in conventional:
        lines.append(f"- `{f}`")
    lines.extend(
        [
            "",
            "## Top Results By Mean PR-AUC",
            "",
            "| Rank | Feature set | Model | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, (_, row) in enumerate(summary.head(18).iterrows(), start=1):
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
            "## Saved Outputs",
            "",
            f"- Model folder: `{OUT_DIR}`",
            f"- Summary: `{OUT_DIR / 'v3_model_family_summary.csv'}`",
            f"- Fold metrics: `{OUT_DIR / 'v3_model_family_fold_metrics.csv'}`",
            f"- OOF predictions: `{OUT_DIR / 'v3_model_family_spatial_cv_predictions.csv'}`",
            f"- Meta coefficients: `{OUT_DIR / 'v3_model_family_meta_coefficients.csv'}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, conventional, alpha = load_data()
    fsets = feature_sets(conventional, alpha)
    base_pred, base_metrics = run_base_models(df, fsets)
    stacked_pred, stacked_metrics, weights = run_stacking(base_pred)
    all_pred = pd.concat([base_pred, stacked_pred], ignore_index=True)
    all_metrics = pd.concat([base_metrics, stacked_metrics], ignore_index=True)
    summary = summarize(all_metrics)

    base_pred.to_csv(OUT_DIR / "v3_model_family_base_oof_predictions.csv", index=False)
    stacked_pred.to_csv(OUT_DIR / "v3_model_family_stacked_oof_predictions.csv", index=False)
    all_pred.to_csv(OUT_DIR / "v3_model_family_spatial_cv_predictions.csv", index=False)
    all_metrics.to_csv(OUT_DIR / "v3_model_family_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "v3_model_family_summary.csv", index=False)
    weights.to_csv(OUT_DIR / "v3_model_family_meta_coefficients.csv", index=False)
    write_curves(all_pred)
    write_report(summary, conventional)
    run_summary = {
        "input_table": str(DATA),
        "rows": int(len(df)),
        "feature_sets": {FEATURE_SET_NAMES[k]: len(v) for k, v in fsets.items()},
        "base_models": [MODEL_NAMES[k] for k in BASE_MODEL_KEYS],
        "meta_model": MODEL_NAMES["stacked_l2_logistic"],
        "validation": "5-fold spatial block CV using spatial_fold_5 and out-of-fold base probabilities for meta-learning",
        "best_by_pr_auc": summary.iloc[0].to_dict(),
    }
    (OUT_DIR / "v3_model_family_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    STATUS_REPORT.write_text(
        "\n".join(
            [
                "# v3 Model Family Status",
                "",
                "The corrected v3 model family has completed.",
                "",
                f"- Input table: `{DATA}`",
                f"- Output folder: `{OUT_DIR}`",
                f"- Best by PR-AUC: {summary.iloc[0]['feature_set_name']} / {summary.iloc[0]['model_name']}",
                f"- ROC-AUC: {summary.iloc[0]['roc_auc_mean']:.3f} +/- {summary.iloc[0]['roc_auc_std']:.3f}",
                f"- PR-AUC: {summary.iloc[0]['pr_auc_mean']:.3f} +/- {summary.iloc[0]['pr_auc_std']:.3f}",
                f"- Brier: {summary.iloc[0]['brier_mean']:.3f} +/- {summary.iloc[0]['brier_std']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.head(18).to_string(index=False))


if __name__ == "__main__":
    main()
