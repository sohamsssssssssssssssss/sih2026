# Village boundaries

`villages.v1.geojson` holds the village and town polygons used to name flooded villages in Sentinel-1 water-change answers (`models/change/villages.py`).

## Source and licence

- Data: [DataMeet Indian Village Boundaries](https://github.com/datameet/indian_village_boundaries), at commit `85060b611724af4d9def71e16b1aa0de941fa6f2`.
- Licence: [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). This subset is a derived database and is shared under the same licence. Attribution: © DataMeet contributors.
- The polygons are community-digitised Census 2001 village boundaries. They are not an official government layer. Names, codes and extents may have changed since 2001.

| Input file | SHA-256 |
|---|---|
| `br/br.geojson` (Bihar, 45,648 features) | `2f2eff14835123baedf4bd959fd7e7bf87c223746141e7717f0869e28c9e49c9` |
| `kl/kl.geojson` (Kerala, 1,533 features) | `0f396cbdc176e520870f97a93c778df1cf46b8d030f0fa27ae995c0c6abfee53` |

## How the subset was made

```bash
python -m data.village_boundaries br.geojson kl.geojson
```

The script keeps every feature whose bounding box overlaps the grid that `backend.sentinel1` requests for an event in `data/manifests/flood_events.v1.json`. That grid is the AOI projected to UTM and snapped outward, so it is a little wider than the AOI. Properties and coordinates are copied unchanged. The result has 94 features (Bihar 54, Kerala 40; 78 villages and 16 towns). The SHA-256 of `villages.v1.geojson` is `fc9d7527430f663a9319f7fdc8ef84a9d52dca6522ca21f4f4400c04fbf0c904`.

Properties: `NAME`, `TYPE` (Village or Town), `SUB_DIST`, `DISTRICT`, `STATE`, `CEN_2001` (Census 2001 location code).

## Gaps

DataMeet has no Assam file, so `silchar-2022` has no village boundaries. Answers for that area say villages are not named.
