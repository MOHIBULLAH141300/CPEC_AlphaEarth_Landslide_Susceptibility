"""Create pixel-level source-only LODO AoA rasters for the three feature sets.

Each subdomain is evaluated using preprocessing, PCA, and a nearest-neighbour
threshold fitted only to samples from the other four domains. The five target
pieces are mosaicked into one 250 m dissimilarity-index raster per feature set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import shapefile
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_manuscript_v3_stacks_to_baseline_rasters import (  # noqa: E402
    AE_DIR,
    BASE,
    LITHOLOGY,
    LOCAL,
    TERRAIN,
    alpha_frame,
    conventional_frame,
    tile_offsets,
    verify_alignment,
    windows,
)
from run_manuscript_v3_harmonised_aoa import different_block_distance  # noqa: E402
from run_manuscript_v3_leave_one_domain_out import DOMAIN_DISPLAY  # noqa: E402
from run_manuscript_v3_nested_spatial_cv import (  # noqa: E402
    OUTER_BUFFER_KM,
    PROJECT_ROOT,
    SEED,
    buffered_indices,
    load_data,
    make_preprocessor,
)


DOMAIN_SHP = (
    PROJECT_ROOT
    / "FINAL PAPER"
    / "00_Study_Area"
    / "Figure_1_ArcMap_layers"
    / "02_CPEC_transfer_subdomains.shp"
)
OUT_DIR = PROJECT_ROOT / "04_maps" / "manuscript_v6_lodo_aoa_250m"
NODATA = np.float32(-9999.0)
LATENT_DIMENSIONS = 15
THRESHOLD_QUANTILE = 0.95
BLOCK_SIZE = 512
FEATURE_ORDER = [
    "conventional",
    "alphaearth_embeddings",
    "conventional_alphaearth_embeddings",
]
OUTPUTS = {
    feature_set: OUT_DIR / f"cpec_baseline_{feature_set}_lodo_dissimilarity_index_250m.tif"
    for feature_set in FEATURE_ORDER
}
POOLED_FUSION_OUTPUT = (
    OUT_DIR
    / "cpec_baseline_conventional_alphaearth_embeddings_pooled_dissimilarity_index_250m.tif"
)
RAW_TO_DISPLAY = {
    "Balochistan": "Balochistan",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "KP-AJK": "KPK-AJK",
    "Kashgar (Xinjiang, China)": "Kashgar",
    "Punjab-Sindh lowland corridor": "Punjab-Sindh",
}
DISPLAY_ORDER = ["Kashgar", "Gilgit-Baltistan", "KPK-AJK", "Balochistan", "Punjab-Sindh"]


@dataclass
class AoaModel:
    preprocess: object
    pca: PCA
    neighbour: NearestNeighbors
    threshold: float
    explained_variance_ratio: float
    n_source: int

    def dissimilarity(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocess.transform(frame)
        latent = self.pca.transform(transformed)
        distance = self.neighbour.kneighbors(latent, return_distance=True)[0].ravel()
        return (distance / self.threshold).astype("float32")


def domain_raster(reference) -> tuple[np.ndarray, dict[int, str]]:
    reader = shapefile.Reader(str(DOMAIN_SHP), encoding="latin1")
    field_names = [field[0] for field in reader.fields[1:]]
    domain_index = field_names.index("domain")
    shapes = []
    code_to_display: dict[int, str] = {}
    for code, record in enumerate(reader.iterShapeRecords(), start=1):
        raw = str(record.record[domain_index])
        display = RAW_TO_DISPLAY.get(raw, raw)
        shapes.append((record.shape.__geo_interface__, code))
        code_to_display[code] = display
    raster = rasterize(
        shapes,
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return raster, code_to_display


def fit_models(df: pd.DataFrame, specs) -> tuple[dict[str, dict[str, AoaModel]], list[dict]]:
    all_idx = np.arange(len(df))
    raw_domains = df.cpec_admin_transfer_domain.astype(str).to_numpy()
    models: dict[str, dict[str, AoaModel]] = {feature_set: {} for feature_set in FEATURE_ORDER}
    metadata = []
    for feature_set in FEATURE_ORDER:
        spec = specs[feature_set]
        for raw_domain in sorted(np.unique(raw_domains)):
            display = DOMAIN_DISPLAY[raw_domain]
            test_idx = all_idx[raw_domains == raw_domain]
            raw_train = all_idx[raw_domains != raw_domain]
            train_idx = buffered_indices(df, raw_train, test_idx, OUTER_BUFFER_KM)
            preprocess = make_preprocessor(spec)
            x_train = preprocess.fit_transform(df.loc[train_idx, spec.columns])
            dimensions = min(LATENT_DIMENSIONS, x_train.shape[1], len(train_idx) - 1)
            pca = PCA(n_components=dimensions, whiten=True, random_state=SEED)
            z_train = pca.fit_transform(x_train)
            blocks = df.loc[train_idx, "spatial_block_1deg"].astype(str).to_numpy()
            reference = different_block_distance(z_train, blocks)
            threshold = float(np.quantile(reference, THRESHOLD_QUANTILE))
            model = AoaModel(
                preprocess=preprocess,
                pca=pca,
                neighbour=NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_train),
                threshold=threshold,
                explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
                n_source=len(train_idx),
            )
            models[feature_set][display] = model
            metadata.append(
                {
                    "feature_set": feature_set,
                    "feature_set_name": spec.display_name,
                    "held_out_domain": display,
                    "n_source_after_buffer": len(train_idx),
                    "latent_dimensions": dimensions,
                    "explained_variance_ratio": model.explained_variance_ratio,
                    "reference_distance_threshold_p95": threshold,
                }
            )
            print(f"Fitted raster AoA: {spec.display_name} / {display}", flush=True)
    return models, metadata


def fit_pooled_fusion_model(df: pd.DataFrame, spec) -> tuple[AoaModel, dict[str, object]]:
    preprocess = make_preprocessor(spec)
    x_train = preprocess.fit_transform(df[spec.columns])
    dimensions = min(LATENT_DIMENSIONS, x_train.shape[1], len(df) - 1)
    pca = PCA(n_components=dimensions, whiten=True, random_state=SEED)
    z_train = pca.fit_transform(x_train)
    blocks = df.spatial_block_1deg.astype(str).to_numpy()
    reference = different_block_distance(z_train, blocks)
    threshold = float(np.quantile(reference, THRESHOLD_QUANTILE))
    model = AoaModel(
        preprocess=preprocess,
        pca=pca,
        neighbour=NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_train),
        threshold=threshold,
        explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
        n_source=len(df),
    )
    metadata = {
        "feature_set": "conventional_alphaearth_embeddings",
        "feature_set_name": spec.display_name,
        "support_mode": "pooled-deployment AoA",
        "n_source": len(df),
        "latent_dimensions": dimensions,
        "explained_variance_ratio": model.explained_variance_ratio,
        "reference_distance_threshold_p95": threshold,
    }
    return model, metadata


def output_profile(reference) -> dict:
    profile = reference.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=float(NODATA),
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def initialise(path: Path, reference) -> None:
    if path.exists():
        return
    with rasterio.open(path, "w", **output_profile(reference)) as dst:
        for window in windows(reference.width, reference.height, BLOCK_SIZE):
            dst.write(
                np.full((int(window.height), int(window.width)), NODATA, dtype="float32"),
                1,
                window=window,
            )
        dst.set_band_description(1, "lodo_dissimilarity_index")
        dst.update_tags(
            analysis="manuscript_v6",
            support_mode="source-to-target LODO transfer AoA",
            latent_dimensions=LATENT_DIMENSIONS,
            threshold_quantile=THRESHOLD_QUANTILE,
            inside_aoa="dissimilarity_index <= 1",
            output_interpretation="feature-space support, not predictive accuracy",
        )


def initialise_pooled(path: Path, reference) -> None:
    initialise(path, reference)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(
            support_mode="pooled-deployment AoA",
            domain_logic="preprocessing and reference distribution fitted to all labelled samples",
        )


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf, -9999.0, -32768.0], np.nan)
    return frame


def query_window(
    frame: pd.DataFrame,
    domain_codes: np.ndarray,
    code_to_display: dict[int, str],
    models: dict[str, AoaModel],
) -> np.ndarray:
    output = np.full(len(frame), NODATA, dtype="float32")
    for code in np.unique(domain_codes):
        if code == 0:
            continue
        use = np.flatnonzero(domain_codes == code)
        display = code_to_display[int(code)]
        output[use] = models[display].dissimilarity(frame.iloc[use])
    return output


def apply_conventional(
    reference,
    study: np.ndarray,
    domains: np.ndarray,
    code_to_display: dict[int, str],
    models: dict[str, AoaModel],
    base,
    local,
    terrain,
    lithology,
) -> None:
    path = OUTPUTS["conventional"]
    initialise(path, reference)
    with rasterio.open(path, "r+") as dst:
        for number, window in enumerate(windows(reference.width, reference.height, BLOCK_SIZE), start=1):
            row0, col0 = int(window.row_off), int(window.col_off)
            sub_study = study[row0 : row0 + int(window.height), col0 : col0 + int(window.width)]
            sub_domains = domains[row0 : row0 + int(window.height), col0 : col0 + int(window.width)]
            valid = sub_study & (sub_domains > 0)
            if not valid.any():
                continue
            existing = dst.read(1, window=window)
            if np.all(existing[valid] != NODATA):
                continue
            frame = clean_frame(conventional_frame(base, local, terrain, lithology, window, valid))
            rr, cc = np.nonzero(valid)
            values = query_window(frame, sub_domains[rr, cc], code_to_display, models)
            output = np.full(valid.shape, NODATA, dtype="float32")
            output[rr, cc] = values
            dst.write(output, 1, window=window)
            if number % 100 == 0:
                print(f"Conventional raster window {number}", flush=True)


def apply_alpha_or_fusion(
    feature_set: str,
    reference,
    study: np.ndarray,
    domains: np.ndarray,
    code_to_display: dict[int, str],
    models: dict[str, AoaModel],
    base,
    local,
    terrain,
    lithology,
) -> None:
    path = OUTPUTS[feature_set]
    initialise(path, reference)
    tiles = sorted(AE_DIR.glob("*.tif"))
    with rasterio.open(path, "r+") as dst:
        for tile_path in tiles:
            row_offset, col_offset = tile_offsets(tile_path)
            with rasterio.open(tile_path) as tile:
                for number, tile_window in enumerate(
                    windows(tile.width, tile.height, BLOCK_SIZE), start=1
                ):
                    global_window = Window(
                        col_offset + tile_window.col_off,
                        row_offset + tile_window.row_off,
                        tile_window.width,
                        tile_window.height,
                    )
                    row0, col0 = int(global_window.row_off), int(global_window.col_off)
                    sub_study = study[
                        row0 : row0 + int(global_window.height),
                        col0 : col0 + int(global_window.width),
                    ]
                    sub_domains = domains[
                        row0 : row0 + int(global_window.height),
                        col0 : col0 + int(global_window.width),
                    ]
                    valid = sub_study & (sub_domains > 0)
                    if not valid.any():
                        continue
                    existing = dst.read(1, window=global_window)
                    if np.all(existing[valid] != NODATA):
                        continue
                    alpha = clean_frame(alpha_frame(tile, tile_window, valid))
                    if feature_set == "alphaearth_embeddings":
                        frame = alpha
                    else:
                        conventional = clean_frame(
                            conventional_frame(
                                base, local, terrain, lithology, global_window, valid
                            )
                        )
                        frame = pd.concat(
                            [conventional.reset_index(drop=True), alpha.reset_index(drop=True)],
                            axis=1,
                        )
                    rr, cc = np.nonzero(valid)
                    values = query_window(
                        frame, sub_domains[rr, cc], code_to_display, models
                    )
                    output = np.full(valid.shape, NODATA, dtype="float32")
                    output[rr, cc] = values
                    dst.write(output, 1, window=global_window)
                    if number % 80 == 0:
                        print(f"{feature_set} {tile_path.name}: window {number}", flush=True)


def apply_pooled_fusion(
    reference,
    study: np.ndarray,
    pooled_model: AoaModel,
    base,
    local,
    terrain,
    lithology,
) -> None:
    path = POOLED_FUSION_OUTPUT
    initialise_pooled(path, reference)
    tiles = sorted(AE_DIR.glob("*.tif"))
    with rasterio.open(path, "r+") as dst:
        for tile_path in tiles:
            row_offset, col_offset = tile_offsets(tile_path)
            with rasterio.open(tile_path) as tile:
                for number, tile_window in enumerate(
                    windows(tile.width, tile.height, BLOCK_SIZE), start=1
                ):
                    global_window = Window(
                        col_offset + tile_window.col_off,
                        row_offset + tile_window.row_off,
                        tile_window.width,
                        tile_window.height,
                    )
                    row0, col0 = int(global_window.row_off), int(global_window.col_off)
                    valid = study[
                        row0 : row0 + int(global_window.height),
                        col0 : col0 + int(global_window.width),
                    ]
                    if not valid.any():
                        continue
                    existing = dst.read(1, window=global_window)
                    if np.all(existing[valid] != NODATA):
                        continue
                    alpha = clean_frame(alpha_frame(tile, tile_window, valid))
                    conventional = clean_frame(
                        conventional_frame(
                            base, local, terrain, lithology, global_window, valid
                        )
                    )
                    frame = pd.concat(
                        [conventional.reset_index(drop=True), alpha.reset_index(drop=True)], axis=1
                    )
                    values = pooled_model.dissimilarity(frame)
                    rr, cc = np.nonzero(valid)
                    output = np.full(valid.shape, NODATA, dtype="float32")
                    output[rr, cc] = values
                    dst.write(output, 1, window=global_window)
                    if number % 80 == 0:
                        print(f"pooled fusion {tile_path.name}: window {number}", flush=True)


def qa(outputs: dict[str, Path], study: np.ndarray, domains: np.ndarray, code_to_display) -> pd.DataFrame:
    rows = []
    for feature_set, path in outputs.items():
        with rasterio.open(path) as src:
            data = src.read(1)
            for code, display in code_to_display.items():
                mask = study & (domains == code)
                valid = mask & np.isfinite(data) & (data != NODATA)
                values = data[valid]
                rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_domain": display,
                        "domain_pixels": int(mask.sum()),
                        "valid_pixels": int(valid.sum()),
                        "inside_aoa_pixels": int((values <= 1).sum()),
                        "inside_aoa_percent": float(100 * np.mean(values <= 1)),
                        "median_dissimilarity_index": float(np.median(values)),
                        "p90_dissimilarity_index": float(np.quantile(values, 0.90)),
                        "inside_domain_nodata_pixels": int(mask.sum() - valid.sum()),
                        "raster": str(path),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, specs, _ = load_data()
    models, model_metadata = fit_models(df, specs)
    pooled_model, pooled_metadata = fit_pooled_fusion_model(
        df, specs["conventional_alphaearth_embeddings"]
    )
    model_metadata.append(pooled_metadata)
    pd.DataFrame(model_metadata).to_csv(OUT_DIR / "lodo_aoa_model_metadata.csv", index=False)

    with (
        rasterio.open(BASE) as base,
        rasterio.open(LOCAL) as local,
        rasterio.open(TERRAIN) as terrain,
        rasterio.open(LITHOLOGY) as lithology,
    ):
        verify_alignment(base, local, terrain, lithology)
        domains, code_to_display = domain_raster(base)
        study = domains > 0
        apply_conventional(
            base,
            study,
            domains,
            code_to_display,
            models["conventional"],
            base,
            local,
            terrain,
            lithology,
        )
        for feature_set in ["alphaearth_embeddings", "conventional_alphaearth_embeddings"]:
            apply_alpha_or_fusion(
                feature_set,
                base,
                study,
                domains,
                code_to_display,
                models[feature_set],
                base,
                local,
                terrain,
                lithology,
            )
        apply_pooled_fusion(
            base,
            study,
            pooled_model,
            base,
            local,
            terrain,
            lithology,
        )
        qa_table = qa(OUTPUTS, study, domains, code_to_display)
        pooled_qa = qa(
            {"pooled_conventional_alphaearth_embeddings": POOLED_FUSION_OUTPUT},
            study,
            domains,
            code_to_display,
        )
    qa_table.to_csv(OUT_DIR / "lodo_aoa_raster_qa_and_coverage.csv", index=False)
    pooled_qa.to_csv(OUT_DIR / "pooled_deployment_aoa_raster_qa_and_coverage.csv", index=False)
    manifest = {
        "analysis": "manuscript_v6_pixel_level_lodo_aoa",
        "support_mode": "source-to-target transfer support",
        "domain_logic": "for every pixel, preprocessing and AoA reference distribution exclude that pixel's complete subdomain",
        "latent_dimensions": LATENT_DIMENSIONS,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "source_target_buffer_km": OUTER_BUFFER_KM,
        "resolution": "250 m nominal geographic grid",
        "outputs": {key: str(path) for key, path in OUTPUTS.items()},
        "pooled_deployment_output": str(POOLED_FUSION_OUTPUT),
        "random_seed": SEED,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nPixel-level LODO AoA QA\n", qa_table.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
