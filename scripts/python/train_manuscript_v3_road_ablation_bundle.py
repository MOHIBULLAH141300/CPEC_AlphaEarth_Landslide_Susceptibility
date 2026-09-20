"""Fit and archive the manuscript-v3 fused stack without road distance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    FeatureSpec,
    INNER_BUFFER_KM,
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    load_data,
    nested_feature_set,
    pooled_metrics,
)


OUT_DIR = PROJECT_ROOT / "03_models" / "manuscript_v3_road_distance_ablation_final"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    fused = specs["conventional_alphaearth_embeddings"]
    no_road = FeatureSpec(
        key="conventional_alphaearth_embeddings_no_road_distance",
        display_name="Conventional + AlphaEarth Embeddings (road-distance ablation)",
        continuous=[c for c in fused.continuous if c != "log1p_dist_road_m"],
        categorical=fused.categorical,
    )
    predictions, folds, coefficients, bundle = nested_feature_set(
        df,
        no_road,
        outer_buffer_km=OUTER_BUFFER_KM,
        inner_buffer_km=INNER_BUFFER_KM,
        quick=False,
        fit_final_models=True,
    )
    if bundle is None:
        raise RuntimeError("The deployable road-ablation bundle was not fitted.")
    predictions.to_csv(OUT_DIR / "nested_spatial_cv_predictions.csv", index=False)
    folds.to_csv(OUT_DIR / "nested_spatial_cv_fold_metrics.csv", index=False)
    coefficients.to_csv(OUT_DIR / "nested_spatial_cv_meta_coefficients.csv", index=False)
    pooled_metrics(predictions).to_csv(OUT_DIR / "nested_spatial_cv_pooled_metrics.csv", index=False)
    joblib.dump(bundle, OUT_DIR / "nested_spatial_stack_no_road_distance.joblib", compress=3)
    manifest = {
        "analysis": "manuscript_v3_fused_road_distance_ablation",
        "n_samples": len(df),
        "n_positive": int(df.label.sum()),
        "n_negative": int((df.label == 0).sum()),
        "removed_predictor": "log1p_dist_road_m",
        "outer_buffer_km": OUTER_BUFFER_KM,
        "inner_buffer_km": INNER_BUFFER_KM,
        "random_seed": SEED,
        "score_definition": "case-control susceptibility score",
        "purpose": "test road-exposure ranking stability without circular use of road proximity",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pooled_metrics(predictions).to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
