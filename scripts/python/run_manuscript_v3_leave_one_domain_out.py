"""Leakage-resistant leave-one-domain-out transfer evaluation for manuscript v3."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    BASE_MODEL_KEYS,
    INNER_BUFFER_KM,
    MODEL_NAMES,
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    buffered_indices,
    expected_calibration_error,
    load_data,
    make_meta_model,
    metric_row,
    model_factories,
    predict_positive,
    threshold_for_f1,
)


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_leave_one_domain_out"
DOMAIN_DISPLAY = {
    "Balochistan": "Balochistan",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "KP-AJK": "KPK-AJK",
    "Kashgar (Xinjiang, China)": "Kashgar",
    "Punjab-Sindh lowland corridor": "Punjab-Sindh",
}


def run_transfer(df: pd.DataFrame, spec, held_out_domain: str):
    y = df.label.to_numpy(int)
    fold = df.spatial_fold_5.to_numpy(int)
    domain = df.cpec_admin_transfer_domain.astype(str).to_numpy()
    all_idx = np.arange(len(df))
    test_idx = all_idx[domain == held_out_domain]
    raw_train = all_idx[domain != held_out_domain]
    train_idx = buffered_indices(df, raw_train, test_idx, OUTER_BUFFER_KM)
    factories = model_factories(spec, quick=False)
    inner_folds = sorted(np.unique(fold[train_idx]))
    inner_oof = np.full((len(train_idx), len(BASE_MODEL_KEYS)), np.nan, dtype=float)
    position = {sample: pos for pos, sample in enumerate(train_idx)}

    for inner_fold in inner_folds:
        validation = train_idx[fold[train_idx] == inner_fold]
        candidates = train_idx[fold[train_idx] != inner_fold]
        inner_train = buffered_indices(df, candidates, validation, INNER_BUFFER_KM)
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[validation])) < 2:
            raise RuntimeError(
                f"Missing class for {spec.key}, held out {held_out_domain}, inner fold {inner_fold}."
            )
        for model_col, model_key in enumerate(BASE_MODEL_KEYS):
            model = factories[model_key]()
            model.fit(df.loc[inner_train, spec.columns], y[inner_train])
            score = predict_positive(model, df.loc[validation, spec.columns])
            inner_oof[[position[i] for i in validation], model_col] = score
    if np.isnan(inner_oof).any():
        raise RuntimeError(f"Incomplete inner OOF matrix for {spec.key}, {held_out_domain}.")

    meta = make_meta_model()
    meta.fit(inner_oof, y[train_idx])
    training_score = predict_positive(meta, inner_oof)
    threshold = threshold_for_f1(y[train_idx], training_score)
    test_base = np.empty((len(test_idx), len(BASE_MODEL_KEYS)), dtype=float)
    for model_col, model_key in enumerate(BASE_MODEL_KEYS):
        model = factories[model_key]()
        model.fit(df.loc[train_idx, spec.columns], y[train_idx])
        test_base[:, model_col] = predict_positive(model, df.loc[test_idx, spec.columns])
    test_score = predict_positive(meta, test_base)

    prediction = df.loc[
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
    prediction["sample_id"] = test_idx
    prediction["feature_set"] = spec.key
    prediction["feature_set_name"] = spec.display_name
    prediction["held_out_domain"] = DOMAIN_DISPLAY[held_out_domain]
    prediction["stacked_score"] = test_score
    prediction["threshold"] = threshold
    prediction["prediction"] = (test_score >= threshold).astype(int)
    for model_col, model_key in enumerate(BASE_MODEL_KEYS):
        prediction[f"base_{model_key}_score"] = test_base[:, model_col]

    metrics = {
        "feature_set": spec.key,
        "feature_set_name": spec.display_name,
        "held_out_domain_source": held_out_domain,
        "held_out_domain": DOMAIN_DISPLAY[held_out_domain],
        "n_train_before_buffer": len(raw_train),
        "n_train_after_buffer": len(train_idx),
        "n_test": len(test_idx),
        "n_test_positive": int(y[test_idx].sum()),
        "n_test_negative": int((y[test_idx] == 0).sum()),
        "outer_buffer_km": OUTER_BUFFER_KM,
        "inner_buffer_km": INNER_BUFFER_KM,
        **metric_row(y[test_idx], test_score, threshold),
    }
    coefficients = pd.DataFrame(
        {
            "feature_set": spec.key,
            "feature_set_name": spec.display_name,
            "held_out_domain": DOMAIN_DISPLAY[held_out_domain],
            "base_model": BASE_MODEL_KEYS,
            "base_model_name": [MODEL_NAMES[key] for key in BASE_MODEL_KEYS],
            "meta_coefficient": meta.named_steps["model"].coef_.ravel(),
        }
    )
    return prediction, metrics, coefficients


def block_bootstrap_ci(predictions: pd.DataFrame, repetitions: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (feature_set, held_out), data in predictions.groupby(["feature_set", "held_out_domain"]):
        blocks = data.spatial_block_1deg.astype(str).unique()
        block_rows = {
            block: np.flatnonzero(data.spatial_block_1deg.astype(str).to_numpy() == block)
            for block in blocks
        }
        y_all = data.label.to_numpy(int)
        score_all = data.stacked_score.to_numpy(float)
        values = {"roc_auc": [], "pr_auc": [], "brier": []}
        for _ in range(repetitions):
            sampled = rng.choice(blocks, len(blocks), replace=True)
            idx = np.concatenate([block_rows[block] for block in sampled])
            y, score = y_all[idx], score_all[idx]
            if len(np.unique(y)) < 2:
                continue
            values["roc_auc"].append(roc_auc_score(y, score))
            values["pr_auc"].append(average_precision_score(y, score))
            values["brier"].append(brier_score_loss(y, score))
        points = {
            "roc_auc": roc_auc_score(y_all, score_all),
            "pr_auc": average_precision_score(y_all, score_all),
            "brier": brier_score_loss(y_all, score_all),
        }
        for metric, samples in values.items():
            array = np.asarray(samples, dtype=float)
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_set_name": data.feature_set_name.iloc[0],
                    "held_out_domain": held_out,
                    "metric": metric,
                    "estimate": points[metric],
                    "ci95_low": np.quantile(array, 0.025),
                    "ci95_high": np.quantile(array, 0.975),
                    "bootstrap_repetitions": len(array),
                }
            )
    return pd.DataFrame(rows)


def calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, held_out), data in predictions.groupby(["feature_set", "held_out_domain"]):
        work = data.copy()
        work["bin"] = pd.qcut(work.stacked_score, 10, duplicates="drop")
        for rank, (_, group) in enumerate(work.groupby("bin", observed=True), start=1):
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_set_name": data.feature_set_name.iloc[0],
                    "held_out_domain": held_out,
                    "bin": rank,
                    "n": len(group),
                    "mean_score": group.stacked_score.mean(),
                    "observed_positive_fraction": group.label.mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    predictions, metrics, coefficients = [], [], []
    domains = sorted(df.cpec_admin_transfer_domain.dropna().astype(str).unique())
    for spec in specs.values():
        for domain in domains:
            print(f"Running {spec.display_name}; held out {DOMAIN_DISPLAY[domain]}", flush=True)
            pred, row, coef = run_transfer(df, spec, domain)
            predictions.append(pred)
            metrics.append(row)
            coefficients.append(coef)

    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_table = pd.DataFrame(metrics)
    coefficient_table = pd.concat(coefficients, ignore_index=True)
    intervals = block_bootstrap_ci(prediction_table)
    calibration = calibration_bins(prediction_table)
    prediction_table.to_csv(OUT_DIR / "leave_one_domain_out_predictions.csv", index=False)
    metric_table.to_csv(OUT_DIR / "leave_one_domain_out_metrics.csv", index=False)
    coefficient_table.to_csv(OUT_DIR / "leave_one_domain_out_meta_coefficients.csv", index=False)
    intervals.to_csv(OUT_DIR / "leave_one_domain_out_block_bootstrap_ci.csv", index=False)
    calibration.to_csv(OUT_DIR / "leave_one_domain_out_calibration_bins.csv", index=False)
    manifest = {
        "analysis": "manuscript_v3_leave_one_domain_out",
        "domains": [DOMAIN_DISPLAY[d] for d in domains],
        "feature_sets": [spec.display_name for spec in specs.values()],
        "outer_separation": "entire administrative-geographic domain held out",
        "outer_buffer_km": OUTER_BUFFER_KM,
        "inner_validation": "five spatial block folds within the source domains",
        "inner_buffer_km": INNER_BUFFER_KM,
        "bootstrap": "2000 repetitions resampling 1-degree blocks within each held-out domain",
        "score_definition": "case-control susceptibility score",
        "elapsed_seconds": time.time() - start,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metric_table.to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
