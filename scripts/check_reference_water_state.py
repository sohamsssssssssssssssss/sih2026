"""Check the GSW permanent-water reference against independent clear-sky S2 scenes.

For each low-cloud S2 L2A item in a date range, report the fraction of
unobstructed reference-water pixels whose NDWI exceeds the frozen threshold,
per column band of a case grid written by scripts/acquire_pc_s1s2_case.py.
A band near 0 means the "permanent water" reference does not hold there.
Research tooling; requires pystac-client and planetary-computer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from eval.sensor_necessity import OBSTRUCTED_SCL, frozen_rule, reference_masks

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--datetime", required=True, help="STAC interval, e.g. 2020-05-01/2020-07-10")
    parser.add_argument("--bands", required=True, type=json.loads,
                        help='JSON {"name": [col_start, col_end], ...}')
    parser.add_argument("--max-cloud", type=float, default=5.0)
    args = parser.parse_args()

    import planetary_computer
    import pystac_client

    provenance = json.loads((args.case / "provenance.json").read_text())
    with rasterio.open(args.case / "gsw_occurrence.tif") as dataset:
        occurrence, bounds = dataset.read(1), dataset.bounds
    water, _ = reference_masks(occurrence)
    threshold = frozen_rule()["ndwi_strictly_greater_than"]
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=["sentinel-2-l2a"], bbox=provenance["requested_bbox_lonlat"],
                           datetime=args.datetime).items()
    tile = provenance["s2"]["mgrs_tile"]
    rows = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
        for item in sorted(items, key=lambda value: value.datetime):
            if item.properties.get("s2:mgrs_tile") != tile or item.properties["eo:cloud_cover"] >= args.max_cloud:
                continue
            arrays = {}
            for band in ("B03", "B08", "SCL"):
                with rasterio.open(item.assets[band].href) as dataset:
                    window = from_bounds(*bounds, transform=dataset.transform)
                    # SCL is 20 m: nearest-neighbour upsampling to the 10 m case grid.
                    arrays[band] = dataset.read(1, window=window, out_shape=occurrence.shape).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = (arrays["B03"] - arrays["B08"]) / (arrays["B03"] + arrays["B08"])
            clear = ~np.isin(arrays["SCL"], OBSTRUCTED_SCL)
            row = {"item_id": item.id, "datetime": item.properties["datetime"],
                   "cloud_cover": item.properties["eo:cloud_cover"], "bands": {}}
            for name, (start, end) in args.bands.items():
                mask = np.zeros_like(water)
                mask[:, start:end] = True
                mask &= water & clear
                row["bands"][name] = {"reference_water_pixels": int(mask.sum()),
                                      "ndwi_water_fraction": float((ndwi[mask] > threshold).mean()) if mask.any() else None}
            rows.append(row)
    print(json.dumps({"case": args.case.name, "ndwi_threshold": threshold, "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
