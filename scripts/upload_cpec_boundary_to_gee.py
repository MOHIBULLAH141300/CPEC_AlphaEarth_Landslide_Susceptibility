"""Create a managed Earth Engine boundary asset from the local CPEC boundary.

This avoids unmanaged Cloud Storage staging files. The local conda pyproj
installation is currently unable to resolve EPSG definitions, so this script
reads the polygon coordinates directly from the shapefile and sends them to
Earth Engine with the source projection (`EPSG:32642`) attached.
"""

from pathlib import Path
import struct

import ee


PROJECT = "ee-mohibullah141300"
SOURCE = Path(
    r"C:\Users\Administrator\Desktop\cpec landslides\cpec boundary\cpec boundary\CPEC_BOUNDARY.shp"
)
ASSET_ID = "projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area"


def read_polygon_rings_from_shp(path: Path):
    """Read Polygon/MultiPatch-like ring coordinate arrays from a .shp file.

    Supports ESRI Shapefile Polygon/PolygonZ record bodies. Returns a list of
    ring coordinate lists in the source CRS.
    """
    data = path.read_bytes()
    offset = 100
    all_rings = []

    while offset < len(data):
        if offset + 8 > len(data):
            break
        _record_number, content_length_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_length = content_length_words * 2
        content = data[offset : offset + content_length]
        offset += content_length
        if len(content) < 44:
            continue

        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            continue
        if shape_type not in (5, 15):
            raise ValueError(f"Unsupported shape type {shape_type}; expected Polygon or PolygonZ")

        num_parts, num_points = struct.unpack("<2i", content[36:44])
        parts_start = 44
        points_start = parts_start + 4 * num_parts
        parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start]))
        points_raw = content[points_start : points_start + 16 * num_points]
        points = [
            [float(x), float(y)]
            for x, y in struct.iter_unpack("<2d", points_raw)
        ]

        for i, part_start in enumerate(parts):
            part_end = parts[i + 1] if i + 1 < len(parts) else len(points)
            ring = points[part_start:part_end]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            if len(ring) >= 4:
                all_rings.append(ring)

    if not all_rings:
        raise ValueError(f"No polygon rings found in {path}")
    return all_rings


def main():
    ee.Initialize(project=PROJECT)

    rings = read_polygon_rings_from_shp(SOURCE)
    geometry = ee.Geometry.Polygon(
        coords=rings,
        proj=ee.Projection("EPSG:32642"),
        geodesic=False,
        evenOdd=True,
    )

    props = {
        "source_path": str(SOURCE),
        "source_crs": "EPSG:32642",
        "feature_count": 1,
        "ring_count": len(rings),
        "study_area_name": "Pakistan plus Xinjiang Uygur/Kashgar CPEC study boundary",
        "note": "Master study area supplied by user for CPEC/KKH landslide susceptibility processing.",
    }

    fc = ee.FeatureCollection([ee.Feature(geometry, props)])

    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description="upload_cpec_boundary_official_study_area",
        assetId=ASSET_ID,
    )
    task.start()
    print("Started boundary asset export")
    print("asset_id", ASSET_ID)
    print("task_id", task.id)


if __name__ == "__main__":
    main()
