"""Audit geographic leakage in the official BigEarthNet v2 split and the pilot.

Patch footprints are derived from patch IDs, which need no imagery:
`..._T<tile>_<x>_<y>` is the 1200 m patch at x index <x> and y index <y>
from the upper-left corner of Sentinel-2 tile <tile>. The derivation is checked
against every tracked reference-map footprint before it is used.

Research tooling. Requires pystac-client + planetary-computer (tile origins, once) and the
official Zenodo `metadata.parquet` (record 10891137; SHA-256 in
data/manifests/bigearthnet/source-registry.v1.json).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rasterio.warp import transform

from data.bigearthnet_pilot import sha256_file
from data.spatial_blocking import leakage_audit, nearest_distances_m

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "data" / "manifests" / "bigearthnet"
PATCH_M = 1200
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TILE_ORIGINS_PATH = MANIFEST_DIR / "s2-tile-origins.v1.json"
RADII_M = (600, 1_200, 1_700, 10_000, 50_000)


def patch_location(patch_id: str) -> tuple[str, int, int]:
    tile, x_index, y_index = patch_id.rsplit("_", 3)[1:]
    return tile.removeprefix("T"), int(x_index), int(y_index)


def tile_origins_from_stac(tiles: list[str]) -> dict:
    """Exact (EPSG, ULX, ULY) per S2 tile from the 10 m band geotransform of one
    Planetary Computer sentinel-2-l2a product of that tile. S2 tile origins are
    fixed per tile but are not always on MGRS 100 km corners (offsets of tens of metres)."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    origins = {}
    for tile in tiles:
        search = catalog.search(collections=["sentinel-2-l2a"], max_items=1, filter_lang="cql2-json",
                                filter={"op": "=", "args": [{"property": "s2:mgrs_tile"}, tile]})
        item = next(iter(search.items()), None)
        if item is None:
            raise LookupError(f"no sentinel-2-l2a item for tile {tile}")
        band = item.assets["B02"].extra_fields
        a, _, ulx, _, e, uly = band["proj:transform"][:6]
        if (a, e) != (10.0, -10.0):
            raise ValueError(f"unexpected B02 pixel size for {tile}")
        epsg = int(str(item.properties.get("proj:code") or item.properties["proj:epsg"]).split(":")[-1])
        origins[tile] = {"epsg": epsg, "ulx": ulx, "uly": uly, "source_item": item.id}
    return origins


def patch_bounds(patch_id: str, corners: dict) -> tuple[int, list[float]]:
    tile, x_index, y_index = patch_location(patch_id)
    origin = corners[tile]
    epsg, ulx, uly = origin["epsg"], origin["ulx"], origin["uly"]
    left, top = ulx + x_index * PATCH_M, uly - y_index * PATCH_M
    return epsg, [left, top - PATCH_M, left + PATCH_M, top]


def verify_against_reference_maps(corners: dict) -> dict:
    audit = json.loads((MANIFEST_DIR / "reference-map-audit.v1.json").read_text())
    mismatches = []
    for record in audit["records"]:
        epsg, bounds = patch_bounds(record["patch_id"], corners)
        if f"EPSG:{epsg}" != record["crs"] or bounds != record["bounds"]:
            mismatches.append({"patch_id": record["patch_id"], "derived": [epsg, bounds],
                               "reference": [record["crs"], record["bounds"]]})
    return {"checked": len(audit["records"]), "mismatches": mismatches}


def centres_lonlat(patch_ids: list[str], corners: dict) -> np.ndarray:
    by_epsg: dict[int, list[int]] = defaultdict(list)
    xs, ys = np.empty(len(patch_ids)), np.empty(len(patch_ids))
    for index, patch_id in enumerate(patch_ids):
        epsg, (left, bottom, _, _) = patch_bounds(patch_id, corners)
        by_epsg[epsg].append(index)
        xs[index], ys[index] = left + PATCH_M / 2, bottom + PATCH_M / 2
    lonlat = np.empty((len(patch_ids), 2))
    for epsg, indices in by_epsg.items():
        lons, lats = transform(f"EPSG:{epsg}", "EPSG:4326", xs[indices].tolist(), ys[indices].tolist())
        lonlat[indices] = np.column_stack([lons, lats])
    return lonlat


def same_footprint_across_splits(patch_ids: list[str], splits: list[str]) -> dict:
    """Identical tile + x/y index (same ground footprint, different acquisition) in two splits."""
    locations: dict[tuple, set[str]] = defaultdict(set)
    for patch_id, split in zip(patch_ids, splits):
        locations[patch_location(patch_id)].add(split)
    shared = Counter("+".join(sorted(value)) for value in locations.values() if len(value) > 1)
    return {"unique_footprints": len(locations), "footprints_in_multiple_splits": dict(shared)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=MANIFEST_DIR / "split-leakage-audit.v1.json")
    args = parser.parse_args()

    registry = json.loads((MANIFEST_DIR / "source-registry.v1.json").read_text())
    expected = next(s["sha256"] for s in registry["sources"] if s["name"] == "bigearthnet_v2_metadata")
    if sha256_file(args.metadata) != expected:
        raise ValueError("metadata.parquet does not match the registered SHA-256")
    table = pq.read_table(args.metadata, columns=["patch_id", "split"])
    patch_ids, splits = table.column("patch_id").to_pylist(), table.column("split").to_pylist()
    pilot = json.loads((MANIFEST_DIR / "pilot-candidates.v1.json").read_text())["pairs"]

    tiles = sorted({patch_location(p)[0] for p in patch_ids})
    if TILE_ORIGINS_PATH.exists():
        corners = json.loads(TILE_ORIGINS_PATH.read_text())["origins"]
    else:
        corners = tile_origins_from_stac(tiles)
        TILE_ORIGINS_PATH.write_text(json.dumps({
            "version": "1.0", "source": STAC_URL,
            "definition": "UL corner of the 10 m B02 grid of one sentinel-2-l2a product per tile",
            "origins": corners}, indent=2, sort_keys=True) + "\n")
    if sorted(corners) != tiles:
        raise ValueError("tile origin manifest does not cover exactly the metadata tiles")
    verification = verify_against_reference_maps(corners)
    if verification["mismatches"]:
        raise ValueError(f"footprint derivation disagrees with reference maps: {verification['mismatches'][:3]}")

    centres = centres_lonlat(patch_ids, corners)
    official = leakage_audit(centres[:, 0], centres[:, 1], splits, radii_m=RADII_M)

    pilot_ids = [row["patch_id"] for row in pilot]
    pilot_centres = centres_lonlat(pilot_ids, corners)
    pilot_declared = np.array([[row["longitude"], row["latitude"]] for row in pilot])
    declared_offset = nearest_distances_m(pilot_declared, pilot_centres)  # sanity: declared vs derived
    pilot_split = [row["split"] for row in pilot]
    pilot_internal = leakage_audit(pilot_centres[:, 0], pilot_centres[:, 1], pilot_split, radii_m=RADII_M)
    train_centres = centres[np.asarray(splits) == "train"]
    pilot_vs_full_train = {}
    for split in ("validation", "test", "bench"):
        mask = np.asarray(pilot_split) == split
        distances = nearest_distances_m(pilot_centres[mask], train_centres)
        pilot_vs_full_train[split] = {
            "samples": int(mask.sum()),
            "nearest_official_train_m": {"min": float(distances.min()), "median": float(np.median(distances))},
            "within_radius": {str(r): int(np.count_nonzero(distances <= r)) for r in RADII_M},
        }

    report = {
        "version": "1.0",
        "inputs": {"metadata_parquet_sha256": expected, "patches": len(patch_ids), "tiles": len(corners),
                   "pilot_manifest_sha256": json.loads((MANIFEST_DIR / "pilot-candidates.v1.json").read_text())["manifest_sha256"]},
        "method": {
            "footprint": "S2 tile UL (s2-tile-origins.v1.json) + x_index*1200 m east, y_index*1200 m south",
            "footprint_verified_against_reference_maps": verification,
            "distance": "great-circle distance between patch centres (spherical Earth, R=6371008.8 m)",
            "radius_meaning_m": {"600": "footprints overlap by >= half a patch on both axes",
                                 "1200": "edge-adjacent or overlapping", "1700": "includes diagonal neighbours",
                                 "10000": "same local landscape", "50000": "same region"},
            "scope": "main metadata.parquet only (480,038 patches); the 69,450 snow/cloud/shadow patches are excluded",
        },
        "official_split": {"nearest_train_by_split": official,
                           "same_footprint_across_splits": same_footprint_across_splits(patch_ids, splits)},
        "pilot": {"declared_vs_derived_centre_m": {"max": float(declared_offset.max()), "median": float(np.median(declared_offset))},
                  "internal_split": pilot_internal, "versus_full_official_train": pilot_vs_full_train},
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2)[:6000])


if __name__ == "__main__":
    main()
