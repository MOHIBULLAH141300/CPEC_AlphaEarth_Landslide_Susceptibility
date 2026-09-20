"""Build the first clean point-based CPEC landslide inventory.

The output is a managed modelling inventory table. Raw shapefiles are read only.
This script avoids GeoPandas reprojection issues by using pyogrio bounds for
point coordinates and rasterio.warp for CRS conversion where needed.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyogrio
from rasterio.warp import transform


PROJECT_ROOT = Path(r"D:\DING PROJECT")
OUT_DIR = PROJECT_ROOT / "01_clean_data" / "inventory"
REPORT_DIR = PROJECT_ROOT / "05_reports"
SOURCE_PROBE = PROJECT_ROOT / "00_inventory" / "official_geohazard_paths.json"

ROCKFALL_POINTS = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\CPEC rockfall\rockfall points\rock_fall_points.shp"
)
LANDSLIDE_POINTS = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\RAW DATA\landslides points\landslide_points_CPEC.shp"
)
HMA_POINTS = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\HMA_LS_Cat_2-20260421_073021\HMA_LS_Cat_point_v02.0.shp"
)

# Bounding box from the uploaded official CPEC study area asset metadata.
CPEC_BOUNDS = {
    "min_lon": 60.8971,
    "max_lon": 79.8801,
    "min_lat": 23.6966,
    "max_lat": 41.4227,
}


def dms_to_decimal(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)", text)
    if not match:
        try:
            return float(text)
        except ValueError:
            return None
    deg, minute, second = map(float, match.groups())
    sign = -1 if deg < 0 else 1
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


def event_year_from_text(text: object) -> tuple[int | None, str]:
    if text is None or pd.isna(text):
        return None, ""
    s = str(text)
    years = re.findall(r"(19\d{2}|20\d{2})", s)
    if not years:
        return None, ""
    year = int(years[0])
    if "左右" in s or "约" in s:
        return year, "approximate"
    return year, "text_mentioned"


def in_cpec_bbox(lon: float | None, lat: float | None) -> bool:
    if lon is None or lat is None or math.isnan(lon) or math.isnan(lat):
        return False
    return (
        CPEC_BOUNDS["min_lon"] <= lon <= CPEC_BOUNDS["max_lon"]
        and CPEC_BOUNDS["min_lat"] <= lat <= CPEC_BOUNDS["max_lat"]
    )


def point_bounds(path: Path | str) -> pd.DataFrame:
    fids, bounds = pyogrio.read_bounds(path)
    return pd.DataFrame(
        {
            "fid": fids,
            "x": bounds[0],
            "y": bounds[1],
            "xmax": bounds[2],
            "ymax": bounds[3],
        }
    )


def read_attrs(path: Path | str) -> pd.DataFrame:
    return pyogrio.read_dataframe(path, read_geometry=False).reset_index(drop=True)


def base_record(
    inventory_id: str,
    source: str,
    source_path: str,
    source_fid: int,
    lon: float | None,
    lat: float | None,
    hazard_type: str,
    source_role: str,
    confidence: str,
) -> dict[str, object]:
    return {
        "inventory_id": inventory_id,
        "source": source,
        "source_path": source_path,
        "source_fid": source_fid,
        "longitude": lon,
        "latitude": lat,
        "geometry_type": "point",
        "hazard_type": hazard_type,
        "event_date": "",
        "event_year": "",
        "event_year_quality": "",
        "trigger": "",
        "country_region": "",
        "area_m2": "",
        "fatalities": "",
        "injuries": "",
        "source_role": source_role,
        "confidence": confidence,
        "use_role": "",
        "duplicate_cluster_id": "",
        "duplicate_priority": "",
        "notes": "",
    }


def load_rockfall() -> list[dict[str, object]]:
    attrs = read_attrs(ROCKFALL_POINTS)
    coords = point_bounds(ROCKFALL_POINTS)
    rows = []
    for i, row in attrs.iterrows():
        lon = dms_to_decimal(row.get("经度"))
        lat = dms_to_decimal(row.get("纬度"))
        if lon is None or lat is None:
            lon = float(coords.loc[i, "x"])
            lat = float(coords.loc[i, "y"])
        event_year, year_quality = event_year_from_text(row.get("其它特征"))
        rec = base_record(
            inventory_id=f"CPEC_ROCKFALL_{i + 1:04d}",
            source="CPEC_1970_2020_rockfall_points",
            source_path=str(ROCKFALL_POINTS),
            source_fid=int(coords.loc[i, "fid"]),
            lon=lon,
            lat=lat,
            hazard_type="rockfall",
            source_role="primary_train_candidate",
            confidence="high",
        )
        rec.update(
            {
                "event_year": event_year or "",
                "event_year_quality": year_quality,
                "trigger": row.get("诱发因素", "") or "",
                "country_region": row.get("国家", "") or "",
                "area_m2": float(row.get("面积_万m2", 0) or 0) * 10000 if str(row.get("面积_万m2", "")).strip() else "",
                "notes": row.get("其它特征", "") or "",
            }
        )
        rows.append(rec)
    return rows


def load_landslide_points() -> list[dict[str, object]]:
    coords = point_bounds(LANDSLIDE_POINTS)
    lon, lat = transform("EPSG:32642", "EPSG:4326", coords["x"].tolist(), coords["y"].tolist())
    rows = []
    for i, (x_lon, y_lat) in enumerate(zip(lon, lat)):
        rec = base_record(
            inventory_id=f"CPEC_LANDSLIDE_{i + 1:04d}",
            source="CPEC_1970_2020_landslide_points_geometry_only",
            source_path=str(LANDSLIDE_POINTS),
            source_fid=int(coords.loc[i, "fid"]),
            lon=float(x_lon),
            lat=float(y_lat),
            hazard_type="landslide",
            source_role="primary_train_candidate",
            confidence="medium_geometry_only",
        )
        rec["notes"] = "Attributes are corrupted/empty in local point shapefile; geometry retained as CPEC inventory evidence."
        rows.append(rec)
    return rows


def normalize_hazard_type(name: object) -> str:
    text = str(name or "").strip().lower()
    if "崩塌" in text or "rock" in text:
        return "rockfall"
    if "泥石流" in text or "debris" in text:
        return "debris_flow"
    if "滑坡" in text or "slide" in text:
        return "landslide"
    if "complex" in text:
        return "complex"
    return text or "unknown"


def load_official_geohazard() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = json.loads(SOURCE_PROBE.read_text(encoding="utf-8-sig"))
    # Use the broader full-CPEC file as the secondary source; the northern file
    # is retained in the probe metadata as a nested/overlap source.
    full_source = max(sources, key=lambda r: int(r["feature_count"]))
    path = full_source["path"]
    attrs = read_attrs(path)
    coords = point_bounds(path)
    for i, row in attrs.iterrows():
        lon = float(coords.loc[i, "x"])
        lat = float(coords.loc[i, "y"])
        if not in_cpec_bbox(lon, lat):
            continue
        hazard_type = normalize_hazard_type(row.get("Name"))
        rec = base_record(
            inventory_id=f"CPEC_GEOHAZARD_{i + 1:05d}",
            source="CPEC_geohazard_points_1990_2019_full",
            source_path=path,
            source_fid=int(coords.loc[i, "fid"]),
            lon=lon,
            lat=lat,
            hazard_type=hazard_type,
            source_role="secondary_context_candidate",
            confidence="medium_type_only",
        )
        rec.update(
            {
                "event_year_quality": "period_1990_2019_only",
                "notes": f"Original Name={row.get('Name', '')}; limited attributes, likely overlaps primary CPEC inventory.",
            }
        )
        rows.append(rec)
    return rows


def load_hma() -> list[dict[str, object]]:
    attrs = read_attrs(HMA_POINTS)
    coords = point_bounds(HMA_POINTS)
    rows = []
    for i, row in attrs.iterrows():
        lon = float(row.get("longitude") or coords.loc[i, "x"])
        lat = float(row.get("latitude") or coords.loc[i, "y"])
        if not in_cpec_bbox(lon, lat):
            continue
        if str(row.get("ctry_code", "")).upper() not in {"PK", "CN"}:
            continue
        rec = base_record(
            inventory_id=f"HMA_{int(row.get('ev_id')) if str(row.get('ev_id')).isdigit() else i + 1}",
            source="NASA_HMA_LS_Cat_V002",
            source_path=str(HMA_POINTS),
            source_fid=int(coords.loc[i, "fid"]),
            lon=lon,
            lat=lat,
            hazard_type=normalize_hazard_type(row.get("ls_cat")),
            source_role="temporal_validation_enrichment",
            confidence="high_dated_public_catalog",
        )
        event_date = str(row.get("ev_date", "") or "")
        rec.update(
            {
                "event_date": event_date,
                "event_year": event_date[:4] if len(event_date) >= 4 else "",
                "event_year_quality": "exact_or_catalog_date",
                "trigger": row.get("ls_trig", "") or "",
                "country_region": f"{row.get('ctry_name', '')}; {row.get('div_name', '')}",
                "fatalities": row.get("fatalities", "") if not pd.isna(row.get("fatalities", "")) else "",
                "injuries": row.get("injuries", "") if not pd.isna(row.get("injuries", "")) else "",
                "notes": row.get("ev_title", "") or "",
            }
        )
        rows.append(rec)
    return rows


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assign_duplicate_clusters(df: pd.DataFrame, threshold_m: float = 100.0) -> pd.DataFrame:
    priority = {
        "primary_train_candidate": 1,
        "temporal_validation_enrichment": 2,
        "secondary_context_candidate": 3,
    }
    df = df.copy()
    df["_grid_lon"] = (df["longitude"].astype(float) * 100).round().astype(int)
    df["_grid_lat"] = (df["latitude"].astype(float) * 100).round().astype(int)
    cluster_ids = [""] * len(df)
    duplicate_priority = [""] * len(df)
    cluster_no = 0
    for idx, row in df.iterrows():
        if cluster_ids[idx]:
            continue
        candidates = df[
            (df["_grid_lon"].between(row["_grid_lon"] - 1, row["_grid_lon"] + 1))
            & (df["_grid_lat"].between(row["_grid_lat"] - 1, row["_grid_lat"] + 1))
        ].index.tolist()
        members = []
        for j in candidates:
            if cluster_ids[j]:
                continue
            d = haversine_m(float(row["longitude"]), float(row["latitude"]), float(df.at[j, "longitude"]), float(df.at[j, "latitude"]))
            if d <= threshold_m:
                members.append(j)
        if len(members) > 1:
            cluster_no += 1
            cid = f"DUP_{cluster_no:05d}"
            ranked = sorted(members, key=lambda k: priority.get(str(df.at[k, "source_role"]), 9))
            for rank, k in enumerate(ranked, start=1):
                cluster_ids[k] = cid
                duplicate_priority[k] = rank
        else:
            cluster_ids[idx] = f"UNQ_{idx + 1:05d}"
            duplicate_priority[idx] = 1
    df["duplicate_cluster_id"] = cluster_ids
    df["duplicate_priority"] = duplicate_priority
    df.drop(columns=["_grid_lon", "_grid_lat"], inplace=True)
    return df


def assign_use_role(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    roles = []
    for _, row in df.iterrows():
        if not in_cpec_bbox(float(row["longitude"]), float(row["latitude"])):
            roles.append("excluded_outside_boundary_bbox")
        elif int(row["duplicate_priority"]) != 1:
            roles.append("excluded_duplicate_lower_priority")
        elif row["source_role"] == "primary_train_candidate":
            roles.append("train_candidate")
        elif row["source_role"] == "temporal_validation_enrichment":
            roles.append("temporal_validation_candidate")
        else:
            roles.append("secondary_context")
    df["use_role"] = roles
    return df


def write_summary(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_records": int(len(df)),
        "by_source": df["source"].value_counts().to_dict(),
        "by_hazard_type": df["hazard_type"].value_counts().to_dict(),
        "by_use_role": df["use_role"].value_counts().to_dict(),
        "duplicate_clusters": int(df["duplicate_cluster_id"].str.startswith("DUP_").sum()),
        "records_with_event_year": int((df["event_year"].astype(str) != "").sum()),
    }
    (OUT_DIR / "clean_inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Clean Inventory QA Report",
        "",
        "This is the first managed point-based inventory for the clean CPEC LSM workflow. Raw source files were not modified.",
        "",
        "## Source Counts",
        "",
    ]
    for source, count in summary["by_source"].items():
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Use Roles", ""])
    for role, count in summary["by_use_role"].items():
        lines.append(f"- {role}: {count}")
    lines.extend(
        [
            "",
            "## Quality Notes",
            "",
            "- CPEC rockfall points preserve readable attributes, DMS coordinates, triggers, and some approximate event years.",
            "- CPEC landslide points preserve geometry but local attributes are corrupted/empty, so confidence is `medium_geometry_only`.",
            "- Full CPEC 1990-2019 geohazard points are retained as secondary context because attributes are limited and likely overlap primary records.",
            "- NASA HMA points inside the CPEC bounding box are retained mainly for dated temporal validation/enrichment.",
            "- Duplicate clusters are approximate, using a 100 m point-distance threshold and source priority.",
            "- Final exact boundary clipping should be confirmed in Earth Engine using the uploaded official boundary asset before model sampling.",
        ]
    )
    (REPORT_DIR / "clean_inventory_qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, object]] = []
    for loader in (load_rockfall, load_landslide_points, load_official_geohazard, load_hma):
        rows.extend(loader())
    df = pd.DataFrame(rows)
    df = df[df.apply(lambda r: in_cpec_bbox(float(r["longitude"]), float(r["latitude"])), axis=1)].reset_index(drop=True)
    df = assign_duplicate_clusters(df)
    df = assign_use_role(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "cpec_landslide_inventory_master_points.csv", index=False, encoding="utf-8-sig")
    training = df[df["use_role"].isin(["train_candidate", "temporal_validation_candidate"])].copy()
    training.to_csv(OUT_DIR / "cpec_landslide_inventory_model_candidates.csv", index=False, encoding="utf-8-sig")
    write_summary(df)
    print(f"Wrote {len(df)} master records")
    print(f"Wrote {len(training)} model/validation candidate records")


if __name__ == "__main__":
    main()
