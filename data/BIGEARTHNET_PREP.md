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

Geographic split construction is the **next loop**. This module preserves the
official split and country fields; it does not construct or change splits.
