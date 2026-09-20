from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString


PROJECT_ROOT = Path(r"D:\DING PROJECT")
YEAR_DIR = PROJECT_ROOT / "04_maps" / "annual_dynamic_2017_2024" / "2018"
SCENARIO_DIR = YEAR_DIR / "03_scenario_maps_250m" / "seismic_rainfall"
TABLE_DIR = YEAR_DIR / "04_year_specific_tables" / "seismic_rainfall_scenario_road_exposure"
QA_DIR = YEAR_DIR / "05_year_specific_qa"
FIG_DIR = PROJECT_ROOT / "05_reports" / "figures" / "seismic_rainfall_scenarios_2018"

ROAD_GPKG = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_250m_high_impact_outputs"
    / "road_segment_exposure_all_250m.gpkg"
)
RELIABILITY_RASTER = (
    PROJECT_ROOT
    / "04_maps"
    / "stacked_ensemble_250m_high_impact_outputs"
    / "cpec_2018_fused_reliability_score_250m.tif"
)

NODATA = -9999.0
SAMPLE_SPACING_M = 1000.0

SCENARIOS = {
    "baseline": {
        "label": "Baseline",
        "path": SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_baseline_probability_250m.tif",
    },
    "heavy_rainfall": {
        "label": "Heavy rainfall",
        "path": SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_heavy_rainfall_stress_probability_250m.tif",
    },
    "strong_earthquake": {
        "label": "Strong earthquake",
        "path": SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_strong_earthquake_stress_probability_250m.tif",
    },
    "compound": {
        "label": "Compound",
        "path": SCENARIO_DIR / "cpec_2018_fused_seismic_stacked_compound_heavy_rainfall_strong_earthquake_probability_250m.tif",
    },
}


def points_along_geometry(geom, spacing: float = SAMPLE_SPACING_M) -> list:
    if geom is None or geom.is_empty:
        return []
    lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
    points = []
    for line in lines:
        if not isinstance(line, LineString) or line.length <= 0:
            continue
        distances = list(np.arange(0.0, line.length, spacing))
        if not distances or not np.isclose(distances[-1], line.length):
            distances.append(float(line.length))
        points.extend(line.interpolate(distance) for distance in distances)
    return points


def valid_values(values: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= ~np.isclose(arr, nodata)
    mask &= ~np.isclose(arr, NODATA)
    return arr[mask]


def raster_percentiles(path: Path) -> dict[str, float]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).compressed()
    return {
        "p80": float(np.percentile(data, 80)),
        "p90": float(np.percentile(data, 90)),
        "median": float(np.percentile(data, 50)),
        "mean": float(np.mean(data)),
    }


def sample_raster_by_segment(path: Path, roads: gpd.GeoDataFrame, point_lists: list[list]) -> list[np.ndarray]:
    with rasterio.open(path) as src:
        roads_in_raster_crs = roads.to_crs(src.crs) if roads.crs != src.crs else roads
        all_values = []
        for geom, original_points in zip(roads_in_raster_crs.geometry, point_lists):
            points = original_points
            if roads.crs != src.crs:
                points = points_along_geometry(geom)
            coords = [(point.x, point.y) for point in points]
            if not coords:
                all_values.append(np.asarray([], dtype="float32"))
                continue
            sampled = np.asarray([value[0] for value in src.sample(coords)], dtype="float32")
            all_values.append(valid_values(sampled, src.nodata))
    return all_values


def add_segment_metrics(
    roads: gpd.GeoDataFrame,
    scenario_values: dict[str, list[np.ndarray]],
    reliability_values: list[np.ndarray],
    thresholds: dict[str, float],
) -> gpd.GeoDataFrame:
    out = roads.copy()
    reliability_means = []
    for values in reliability_values:
        reliability_means.append(float(np.mean(values)) if values.size else np.nan)
    out["mean_reliability_score"] = reliability_means

    baseline_means = None
    for key, values_by_segment in scenario_values.items():
        means = []
        maxes = []
        sample_counts = []
        p80_shares = []
        p90_shares = []
        reliable_weighted = []
        for values, rel_values in zip(values_by_segment, reliability_values):
            if values.size:
                means.append(float(np.mean(values)))
                maxes.append(float(np.max(values)))
                sample_counts.append(int(values.size))
                p80_shares.append(float(np.mean(values >= thresholds["p80"])))
                p90_shares.append(float(np.mean(values >= thresholds["p90"])))
                if rel_values.size == values.size:
                    reliable_weighted.append(float(np.mean(values * rel_values)))
                else:
                    rel = float(np.mean(rel_values)) if rel_values.size else np.nan
                    reliable_weighted.append(float(np.mean(values) * rel) if np.isfinite(rel) else np.nan)
            else:
                means.append(np.nan)
                maxes.append(np.nan)
                sample_counts.append(0)
                p80_shares.append(np.nan)
                p90_shares.append(np.nan)
                reliable_weighted.append(np.nan)

        out[f"{key}_sample_n"] = sample_counts
        out[f"{key}_mean_probability"] = means
        out[f"{key}_max_probability"] = maxes
        out[f"{key}_p80_share"] = p80_shares
        out[f"{key}_p90_share"] = p90_shares
        out[f"{key}_mean_reliability_weighted_probability"] = reliable_weighted
        if key == "baseline":
            baseline_means = np.asarray(means, dtype="float64")

    if baseline_means is not None:
        for key in SCENARIOS:
            out[f"{key}_mean_delta_from_baseline"] = (
                np.asarray(out[f"{key}_mean_probability"], dtype="float64") - baseline_means
            )
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def summarize_group(df: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows = []
    weights = df["length_km"].astype(float)
    for key, cfg in SCENARIOS.items():
        mean_col = f"{key}_mean_probability"
        delta_col = f"{key}_mean_delta_from_baseline"
        p80_col = f"{key}_p80_share"
        p90_col = f"{key}_p90_share"
        rel_col = f"{key}_mean_reliability_weighted_probability"
        rows.append(
            {
                "road_group": label,
                "scenario": cfg["label"],
                "scenario_key": key,
                "segment_count": int(df.shape[0]),
                "total_length_km": float(weights.sum()),
                "mean_probability_length_weighted": weighted_mean(df[mean_col], weights),
                "mean_delta_from_baseline_length_weighted": weighted_mean(df[delta_col], weights),
                "mean_reliability_weighted_probability": weighted_mean(df[rel_col], weights),
                "p80_share_weighted_length_km": float((df[p80_col].fillna(0.0) * weights).sum()),
                "p90_share_weighted_length_km": float((df[p90_col].fillna(0.0) * weights).sum()),
                "majority_p80_length_km": float(weights[df[p80_col].fillna(0.0) >= 0.5].sum()),
                "majority_p90_length_km": float(weights[df[p90_col].fillna(0.0) >= 0.5].sum()),
                "max_segment_mean_probability": float(df[mean_col].max()),
            }
        )
    return rows


def make_summary(segment_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    records = []
    records.extend(summarize_group(segment_gdf, "All clipped roads"))
    records.extend(summarize_group(segment_gdf[segment_gdf["is_kkh_proxy"].astype(bool)], "KKH proxy routes (N35/314)"))
    return pd.DataFrame(records)


def make_hotspots(segment_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [
        "segment_id",
        "length_km",
        "route_code",
        "is_kkh_proxy",
        "road_type",
        "road_rank",
        "baseline_mean_probability",
        "heavy_rainfall_mean_probability",
        "strong_earthquake_mean_probability",
        "compound_mean_probability",
        "compound_mean_delta_from_baseline",
        "compound_p80_share",
        "compound_p90_share",
        "compound_mean_reliability_weighted_probability",
        "mean_reliability_score",
    ]
    ranked = segment_gdf.copy()
    ranked["compound_scenario_priority_score"] = (
        ranked["compound_mean_probability"].fillna(0.0)
        + ranked["compound_p90_share"].fillna(0.0) * 0.1
        + ranked["compound_mean_reliability_weighted_probability"].fillna(0.0) * 0.2
    )
    ranked = ranked.sort_values("compound_scenario_priority_score", ascending=False).head(50)
    ranked["scenario_hotspot_rank"] = np.arange(1, ranked.shape[0] + 1)
    return ranked[["scenario_hotspot_rank", "compound_scenario_priority_score"] + cols].copy()


def make_increase_hotspots(segment_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [
        "segment_id",
        "length_km",
        "route_code",
        "is_kkh_proxy",
        "road_type",
        "road_rank",
        "baseline_mean_probability",
        "heavy_rainfall_mean_delta_from_baseline",
        "strong_earthquake_mean_delta_from_baseline",
        "compound_mean_delta_from_baseline",
        "compound_mean_probability",
        "compound_p80_share",
        "compound_p90_share",
        "compound_mean_reliability_weighted_probability",
        "mean_reliability_score",
    ]
    ranked = segment_gdf.copy()
    ranked["compound_increase_priority_score"] = (
        ranked["compound_mean_delta_from_baseline"].clip(lower=0.0).fillna(0.0)
        + ranked["compound_p90_share"].fillna(0.0) * 0.05
        + ranked["compound_mean_reliability_weighted_probability"].fillna(0.0) * 0.1
    )
    ranked = ranked.sort_values(
        ["compound_mean_delta_from_baseline", "compound_increase_priority_score"],
        ascending=False,
    ).head(50)
    ranked["scenario_increase_rank"] = np.arange(1, ranked.shape[0] + 1)
    return ranked[["scenario_increase_rank", "compound_increase_priority_score"] + cols].copy()


def make_figure(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [cfg["label"] for cfg in SCENARIOS.values()]
    colors = {
        "Baseline": "#2f5d8c",
        "Heavy rainfall": "#54a24b",
        "Strong earthquake": "#d27d2d",
        "Compound": "#8e3b46",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax, group in zip(axes, ["All clipped roads", "KKH proxy routes (N35/314)"]):
        part = summary[summary["road_group"] == group].set_index("scenario").loc[order]
        x = np.arange(len(order))
        ax.bar(
            x,
            part["mean_probability_length_weighted"],
            color=[colors[item] for item in order],
            edgecolor="#1f2937",
            linewidth=0.6,
        )
        ax.set_title(group, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Length-weighted mean probability")
        ax.set_ylim(0, max(0.1, float(part["mean_probability_length_weighted"].max()) * 1.18))
        for idx, value in enumerate(part["mean_probability_length_weighted"]):
            ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2018 seismic-rainfall scenario road exposure", fontsize=14, fontweight="bold")
    png = FIG_DIR / "figure_2018_seismic_rainfall_scenario_road_exposure.png"
    pdf = FIG_DIR / "figure_2018_seismic_rainfall_scenario_road_exposure.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)


def write_report(summary: pd.DataFrame, thresholds: dict[str, float], hotspot_path: Path, increase_path: Path) -> Path:
    report = QA_DIR / "cpec_2018_fused_seismic_rainfall_scenario_road_exposure_report.md"
    all_roads = summary[summary["road_group"] == "All clipped roads"].set_index("scenario")
    kkh = summary[summary["road_group"] == "KKH proxy routes (N35/314)"].set_index("scenario")
    lines = [
        "# 2018 Seismic-Rainfall Scenario Road Exposure",
        "",
        "Model: fused Conventional + AlphaEarth Embeddings + screened seismic factors, Spatial-CV Stacked Ensemble.",
        "",
        "Scenario thresholds are fixed from the scenario baseline probability raster for comparability:",
        f"- Baseline P80: {thresholds['p80']:.6f}",
        f"- Baseline P90: {thresholds['p90']:.6f}",
        "",
        "Key length-weighted mean probabilities:",
        "",
        "| Road group | Baseline | Heavy rainfall | Strong earthquake | Compound |",
        "|---|---:|---:|---:|---:|",
        (
            "| All clipped roads | "
            f"{all_roads.loc['Baseline', 'mean_probability_length_weighted']:.3f} | "
            f"{all_roads.loc['Heavy rainfall', 'mean_probability_length_weighted']:.3f} | "
            f"{all_roads.loc['Strong earthquake', 'mean_probability_length_weighted']:.3f} | "
            f"{all_roads.loc['Compound', 'mean_probability_length_weighted']:.3f} |"
        ),
        (
            "| KKH proxy routes (N35/314) | "
            f"{kkh.loc['Baseline', 'mean_probability_length_weighted']:.3f} | "
            f"{kkh.loc['Heavy rainfall', 'mean_probability_length_weighted']:.3f} | "
            f"{kkh.loc['Strong earthquake', 'mean_probability_length_weighted']:.3f} | "
            f"{kkh.loc['Compound', 'mean_probability_length_weighted']:.3f} |"
        ),
        "",
        "Outputs:",
        f"- Segment table: `{TABLE_DIR / 'road_segment_seismic_rainfall_scenario_exposure_250m.csv'}`",
        f"- Summary table: `{TABLE_DIR / 'road_seismic_rainfall_scenario_exposure_summary_250m.csv'}`",
        f"- Top hotspot table: `{hotspot_path}`",
        f"- Top scenario-increase hotspot table: `{increase_path}`",
        f"- Figure: `{FIG_DIR / 'figure_2018_seismic_rainfall_scenario_road_exposure.png'}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    roads = gpd.read_file(ROAD_GPKG)
    point_lists = [points_along_geometry(geom) for geom in roads.geometry]
    thresholds = raster_percentiles(SCENARIOS["baseline"]["path"])
    scenario_values = {
        key: sample_raster_by_segment(cfg["path"], roads, point_lists)
        for key, cfg in SCENARIOS.items()
    }
    reliability_values = sample_raster_by_segment(RELIABILITY_RASTER, roads, point_lists)
    segment_gdf = add_segment_metrics(roads, scenario_values, reliability_values, thresholds)
    summary = make_summary(segment_gdf)
    hotspots = make_hotspots(segment_gdf)
    increase_hotspots = make_increase_hotspots(segment_gdf)

    segment_csv = TABLE_DIR / "road_segment_seismic_rainfall_scenario_exposure_250m.csv"
    segment_gpkg = TABLE_DIR / "road_segment_seismic_rainfall_scenario_exposure_250m.gpkg"
    summary_csv = TABLE_DIR / "road_seismic_rainfall_scenario_exposure_summary_250m.csv"
    hotspots_csv = TABLE_DIR / "road_seismic_rainfall_scenario_hotspot_segments_top50_250m.csv"
    increase_hotspots_csv = TABLE_DIR / "road_seismic_rainfall_scenario_increase_hotspot_segments_top50_250m.csv"
    thresholds_json = TABLE_DIR / "road_seismic_rainfall_scenario_thresholds_250m.json"

    segment_gdf.drop(columns="geometry").to_csv(segment_csv, index=False, encoding="utf-8-sig")
    segment_gdf.to_file(segment_gpkg, driver="GPKG")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    hotspots.to_csv(hotspots_csv, index=False, encoding="utf-8-sig")
    increase_hotspots.to_csv(increase_hotspots_csv, index=False, encoding="utf-8-sig")
    thresholds_json.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    make_figure(summary)
    report = write_report(summary, thresholds, hotspots_csv, increase_hotspots_csv)

    print(report)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
