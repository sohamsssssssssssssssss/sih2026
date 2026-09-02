"""Dataloaders. OWNER: B1.

Deliberately unimplemented. B1 ships the tile format by 25 SEPT and fills this
in. Stage 0 eval-only runs do not need it — smoke.py exits cleanly when it
hits the NotImplementedError below.

Tile schema v2 (frozen 25 Sept, bump data.tile_format_version on change):

    sample = {
      "id":          str,
      "optical":     float32[C, H, W],       # C=12 (S2) or 3 (high-res RGB)
      "sar":         float32[3, H, W],       # VV, VH, VV-VH in dB — NOT optional
      "optical_t2":  float32[C, H, W]|None,  # bitemporal second frame
      "gsd":         float,                  # metres/pixel — MANDATORY
      "sensor":      str,                    # S2|S1|cartosat|risat|dota|synthetic
      "meta":        dict,                   # bbox, CRS, dates, source

      # --- v2: BigEarthNet.txt annotation payload ---
      "ann_type":    str,                    # "caption"|"vqa"|"refexp" — MANDATORY
      "caption":     str|None,
      "qa":          list[dict]|None,        # [{"question":..., "answer":...}]
      "refexp":      list[dict]|None,        # [{"expression":..., "box":[x1,y1,x2,y2]}]
      "lulc_labels": list[str]|None,
    }

gsd and sensor are required on EVERY sample. The resolution ladder is
meaningless without them.

WHAT CHANGED IN v2
------------------
`sar` is no longer optional. BigEarthNet.txt ships 464,044 Sentinel-1 SAR +
Sentinel-2 multispectral pairs that are already CO-REGISTERED, so a sample
without SAR is a bug in ingestion, not a valid sparse case.

Bounding boxes in `refexp` are normalised to [0, 1] and axis-aligned.

WHY ann_type EXISTS
-------------------
One shard set serves four tracks. BEN.txt carries 9.6M annotations of three
kinds over the same imagery, so we ingest the pairs ONCE and hand each track a
filtered view rather than building a corpus per track:

    Stage 1  (A1, captioning)  filters ann_type == "caption"
    Stage 2a (A1, VQA)         filters ann_type == "vqa"
    Stage 2b (A1, grounding)   filters ann_type == "refexp"

One ingestion, four views. build_loader(ann_type=None) yields every sample;
passing a value filters to that annotation type.

SUBSAMPLING (READ BEFORE INGESTING)
-----------------------------------
464k pairs do not fit Kaggle's ~73 GB scratch. Target ~50k.

Stratify on all four of: annotation type, geography, LULC class, AND season.
Do NOT random-sample. Random sampling drops the rare complex LULC classes,
and those are exactly the classes the BEN.txt paper reports VLMs failing on —
so a randomly-sampled 50k makes our own numbers look better than they are by
deleting the hard cases.

The manually-verified benchmark split is NEVER subsampled and NEVER trained
on. It is the only clean measurement surface we have.
"""

from __future__ import annotations

ANN_TYPES = ("caption", "vqa", "refexp")


def build_loader(cfg: dict, split: str = "train", limit: int | None = None,
                 ann_type: str | None = None):
    """Build a loader over the tile shards.

    ann_type: None yields every sample; "caption"|"vqa"|"refexp" filters to
    that annotation view. Anything else is a ValueError — a typo here would
    otherwise silently train a track on an empty set.
    """
    if ann_type is not None and ann_type not in ANN_TYPES:
        raise ValueError(
            f"Unknown ann_type {ann_type!r}. Expected one of {ANN_TYPES} or None."
        )

    raise NotImplementedError(
        "B1 has not shipped the dataloader yet. "
        "Tile format freeze: 25 Sept. See docstring for schema v2."
    )
