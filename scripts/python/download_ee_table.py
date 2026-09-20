"""Download an Earth Engine FeatureCollection table to local CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import pandas as pd


PROJECT = "ee-mohibullah141300"
DEFAULT_ASSET = "projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2"
DEFAULT_OUT = Path(r"D:\DING PROJECT\03_models\cpec_2018_lsm_samples_v2.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    fc = ee.FeatureCollection(args.asset)
    size = fc.size().getInfo()
    features = fc.toList(size).getInfo()
    rows = []
    for feature in features:
        props = feature.get("properties", {}).copy()
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        props["longitude"] = coords[0]
        props["latitude"] = coords[1]
        rows.append(props)
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Downloaded {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
