"""Check whether the exact source products of each BigEarthNet pilot pair are
available on Microsoft Planetary Computer (STAC metadata only; no pixels).

S2: a sentinel-2-l2a item for the same MGRS tile and sensing time (+-1 h).
S1: a sentinel-1-rtc item whose ID starts with the pilot's S1 product prefix
(platform, mode, product, polarisation, start time) and covers the patch.

Availability does NOT make the pixels BigEarthNet pixels: BigEarthNet S2
patches carry processing baseline N9999 (a dataset-specific processing run)
and BigEarthNet S1 patches were preprocessed by the dataset authors, whereas
Planetary Computer serves ESA/ESRI L2A and gamma0 RTC. Pixel equality is
unverified and not expected.
Research tooling; requires pystac-client and planetary-computer.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "data" / "manifests" / "bigearthnet"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _interval(center: datetime, delta: timedelta) -> str:
    return f"{(center - delta).isoformat()}/{(center + delta).isoformat()}".replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MANIFEST_DIR / "pc-availability.v1.json")
    args = parser.parse_args()

    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    pairs = json.loads((MANIFEST_DIR / "pilot-candidates.v1.json").read_text())["pairs"]
    rows = []
    for pair in pairs:
        patch_id, s1_name = pair["patch_id"], pair["s1_name"]
        tile = patch_id.rsplit("_", 3)[1].removeprefix("T")
        point = {"type": "Point", "coordinates": [pair["longitude"], pair["latitude"]]}
        s2_items = list(catalog.search(
            collections=["sentinel-2-l2a"], intersects=point,
            datetime=_interval(_utc(patch_id.split("_")[2]), timedelta(hours=1)),
            filter_lang="cql2-json", filter={"op": "=", "args": [{"property": "s2:mgrs_tile"}, tile]},
        ).items())
        s1_prefix = "_".join(s1_name.split("_")[:5])
        s1_items = [item for item in catalog.search(
            collections=["sentinel-1-rtc"], intersects=point,
            datetime=_interval(_utc(s1_name.split("_")[4]), timedelta(minutes=2)),
        ).items() if item.id.startswith(s1_prefix)]
        rows.append({
            "patch_id": patch_id, "split": pair["split"],
            "s2_items": [item.id for item in s2_items],
            "s2_processing_baselines": sorted({item.properties.get("s2:processing_baseline") for item in s2_items}),
            "s1_items": [item.id for item in s1_items],
        })
    summary = {
        "pairs": len(rows),
        "s2_available": sum(bool(row["s2_items"]) for row in rows),
        "s1_available": sum(bool(row["s1_items"]) for row in rows),
        "both_available": sum(bool(row["s2_items"]) and bool(row["s1_items"]) for row in rows),
    }
    report = {"version": "1.0", "source": STAC_URL, "checked_at": datetime.now(timezone.utc).isoformat(),
              "caveat": __doc__.split("\n\n")[2].replace("\n", " "), "summary": summary, "rows": rows}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
