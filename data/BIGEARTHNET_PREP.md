# BigEarthNet preparation index

Keep the local files outside this repository at `~/Datasets/BigEarthNet/`:

- `metadata.parquet`
- `BigEarthNet.txt.parquet`
- `V2/BigEarthNet-S2.tar.gzaa` and `.tar.gzab`
- `V2/BigEarthNet-S1.tar.gzaa` and `.tar.gzab`
- `V2/Reference_Maps.tar.gzaa`

The S1/S2 archives stay compressed. With about 96 GiB free, full extraction is
not viable. This index reads only parquet metadata and annotations; it never
opens image archives or decodes pixels. Do not copy dataset files into the repo.

Verified local counts (2026-10-03):

| Item | Count |
| --- | ---: |
| Annotation rows | 9,553,962 |
| Annotation patch IDs | 464,044 |
| Binary | 3,625,160 |
| MCQ | 3,259,184 |
| Bounding box | 2,205,686 |
| Captioning | 463,932 |
| Metadata patches | 480,038 |

All 463,932 caption patch IDs occur in metadata. The index keeps each
annotation row, joins on `patch_id`, and uses the official metadata `split`
and `country`. Unmatched rows remain visible with null metadata fields and
`_merge == "left_only"`.

Run the summary without reading text or image pixels:

```bash
python3 -m data.bigearthnet \
  ~/Datasets/BigEarthNet/metadata.parquet \
  ~/Datasets/BigEarthNet/BigEarthNet.txt.parquet
```

For code, `data.bigearthnet.iter_index(metadata_path, annotations_path,
annotation_types=("captioning",))` yields joined pandas DataFrames in bounded
annotation batches. The metadata table is loaded once. Omit `annotation_types`
to keep every annotation type. `summarize(...)` returns the CLI counts, with
unmatched rows grouped under `<unmatched>` for split and country.

## Caption geographic split (A.1)

`data.bigearthnet_split` parses and validates the complete Sentinel-2 patch
ID, then groups all caption patches by their MGRS tile. It ranks tiles by
`SHA256("26167:" + tile)` (hexadecimal digest, tile as tie-breaker), using seed
`26167`. The eval side takes the ranked tile prefix whose patch count is
closest to `ceil(0.10 × total captions)`; ties take the longer prefix. The
remaining tiles are train. Algorithm ID:
`sha256-tile-rank-closest-prefix-v1`. The official metadata split remains a
separate `official_split` column in the manifest.

Regenerate the compressed CSV and its JSON sidecar with:

```bash
python3 -m data.bigearthnet_split \
  ~/Datasets/BigEarthNet/metadata.parquet \
  ~/Datasets/BigEarthNet/BigEarthNet.txt.parquet \
  data/manifests/bigearthnet/caption-geo-split.v1.csv.gz
```

The manifest contains one sorted row per caption patch: `patch_id`,
`mgrs_tile`, `geo_split`, `official_split`, `country`. The 1.2 MiB gzip file
is small enough to version with the code. The sidecar records the algorithm,
seed, source paths, counts, and SHA256. Local result (2026-10-04): 422,809
train patches across 48 MGRS tiles; 41,123 eval patches across 6 tiles.
All 463,932 captions were assigned once; train/eval patch and MGRS sets are
disjoint. Source official-split counts among captions: train 229,114,
validation 118,095, test 116,723. Manifest SHA256:
`3e5b1d774125c7b48a9454dbb33e6144aee8d990d8966e830987836159bd83ee`.

Image decoding, Stage-1 training, and evaluation are outside this preparation
step.

## Stage A.2 sample and patch embedding infrastructure

`data.bigearthnet_s2.load_s2_patch(patch_dir)` reads an already accessible
patch directory with one `<patch_id>_<band>.tif` per band. Channel order is
`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`; B10 is
rejected. Each source must be single-band uint16. Rasterio bilinear resampling
maps each native grid onto 120×120. The result is a float32 tensor in CHW
layout `[12, 120, 120]` (batch as NCHW for Conv2d), retaining raw
digital-number scale without clipping or normalization. BigEarthNet V2 marks
zero as nodata; the loader preserves zero values and bilinear interpolation
includes them. Other declared nodata values are rejected. Freeze a nodata and
normalization policy before training.

`models.qwen_vl.stage1.convert_patch_embed` replaces the 3-channel Conv2d
with a 12-channel Conv2d. Each new channel receives the mean of the three
old channel weights multiplied by `3/12`, so repeating a common signal over
12 channels preserves the old activation scale from three copies. The same
module provides parameter groups at vision `1e-5`, patch embed `1e-4`, and
LM LoRA `1e-4` for a future trainer. Archives remain compressed outside the
repo. This A.2 infrastructure loop is not a training run.

## Stage A.2 real S2 validation (2026-10-04)

`scripts.validate_bigearthnet_s2` selected four SHA256-ranked MGRS tiles on
each frozen geographic side, then four SHA256-ranked caption patches per tile
(16 train, 16 eval). The exact 32 IDs, each source TIFF's metadata and zero
fraction, and train/eval/combined per-band output statistics are in
`data/manifests/bigearthnet/stage1-real-validation.v1.json`. Run with
`python3 -m scripts.validate_bigearthnet_s2 --root <extracted-BigEarthNet-S2-root>
--manifest data/manifests/bigearthnet/caption-geo-split.v1.csv.gz --output
<report.json>`. It reads only already extracted patch directories.

Exactly 384 TIFFs (about 5.3 MB of file content) were selectively extracted
to a temporary directory; the split archives remain compressed. All 32
patches passed the loader: 12 canonical channels, no B10, finite float32
`[12,120,120]` output. All 384 TIFFs are uint16 and declare `nodata=0`,
but none of their source pixels are zero in this sample. All observed native
grids match the expected pattern: B02/B03/B04/B08 are 120×120 at 10 m;
B05/B06/B07/B8A/B11/B12 are 60×60 at 20 m; B01/B09 are 20×20 at 60 m.
The resampled output also has zero fraction 0 for every band on both sides.

Combined output means range from 262.15 (B01) to 2317.46 (B09), while
maxima range from 2387 (B01) to 10432 (B04). Several train-band p1 values
are 1 even without zero pixels, and train/eval distributions differ. These
are sample observations, not population estimates. No normalization or
clipping was chosen or implemented in this validation loop.
