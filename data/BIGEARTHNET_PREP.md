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
layout `[12, 120, 120]`, retaining raw
digital-number scale without clipping or normalization. BigEarthNet V2 marks
zero as nodata. The raw loader preserves zeros; when explicitly given the
Stage-1 contract it rejects patches containing raw zeros. Other declared
nodata values are rejected.

`models.qwen_vl.stage1.convert_patch_embed` replaces a 3-channel patch
projection with a 12-channel projection. The pinned Qwen2.5-VL-3B uses a
Conv3d with a two-frame temporal kernel; the utility also supports the earlier
Conv2d synthetic tests. Each new channel receives the mean of the three
old channel weights multiplied by `3/12`, so repeating a common signal over
12 channels preserves the old activation scale from three copies. The same
module provides parameter groups at vision `1e-5`, patch embed `1e-4`, and
LM LoRA `1e-4` for a future trainer. Archives remain compressed outside the
repo. This A.2 infrastructure loop is not a training run.

For the loop-4 Qwen path, `pack_s2_pixel_values` receives the loader's
contract-normalized float32 `[12, 120, 120]` tensor. It resizes to 112×112
with bicubic interpolation, repeats the image across Qwen's two temporal
frames, and packs 64 flattened patches of 4,704 values each. The vision
tower merges these to 16 image tokens. This path bypasses the RGB processor's
1/255 rescale and CLIP mean/std; it does not alter the frozen S2 contract.

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

## Stage-1 normalization study and contract (loop 3)

The full caption manifest was rechecked against the metadata patch inventory
without reading pixels. The train/eval patch and MGRS tile sets do not overlap.
The machine-readable [split verification](manifests/bigearthnet/stage1-split-verification.v1.json)
records counts, source SHA, seed, timestamp, and git SHA. Inventory comparison
uses `metadata.parquet`, not a complete archive member listing.

The [train ID list](manifests/bigearthnet/stage1-norm-train-ids.v1.json) fixes
4,000 caption patches selected with seed 0 by round-robin tile quotas capped
at each tile's available count. Only their 48,000 TIFFs were extracted. The
[study JSON](manifests/bigearthnet/stage1-norm-study.v1.json) contains raw-DN
tails, source DN 0–50 histograms, per-patch `==1` fractions, three diagnostic
quicklooks, and three disjoint 1,000-patch subset comparisons. No eval pixels
were read for the fit. The [option comparison](manifests/bigearthnet/stage1-norm-options.v1.json)
and [loader check](manifests/bigearthnet/stage1-norm-loader-check.v1.json) keep
their own run provenance. [Check results](manifests/bigearthnet/stage1-loop3-checks.v1.json)
record the focused and full suite outcomes.

Source TIFFs themselves contain DN 1 and no zeros or 65535 in this sample.
There is a strong DN 1 pile-up. Some edge regions have all twelve bands at 1;
the quicklook shows one along a black scene boundary. B09 can also be entirely
1 while other bands vary. Thus 1 is not created by division or resampling,
and a single all-band fill explanation does not cover every case. The physical
reason for band-specific 1s, including any BOA offset clipping, is unverified.
Likewise, raw DN values above 10,000 already occur in source TIFFs; dividing
by 10,000 can therefore produce values near or above 1 without an additional
clipping bug. B01/B09 retain visibly coarse 60 m texture after 6× bilinear
upsampling; resampled float32 values are integer-quantized in this run.

The frozen [norm_contract.json](manifests/bigearthnet/norm_contract.json)
chooses option A: divide raw DN by 10,000, clip to [0, 0.9113], then apply
per-band train-fitted mean and std. The clip ceiling is the upper edge of the
one-DN histogram bin containing the pooled train p99.9. The exact per-band
parameters and clipped fractions are in the JSON artifacts. The pooled clipped
fraction is 0.0998%; B02 is highest at 0.2746%. Across disjoint train subsets,
option A's largest absolute mean offset is 0.0344 and largest std deviation
from 1 is 0.0580; its largest relative p99 spread is 16.36%. Option B (global
division and clip only) has largest relative mean/std/p99 spreads of
8.83%/9.28%/14.37%. A was chosen to equalize observed band scales, not for
a proven stability or model-quality advantage. In normalized
mode, the loader rejects any raw zero, sets pixels whose twelve resampled
bands all equal 1 to normalized zero, and retains band-specific 1 values.
Raw mode remains available by omitting `norm_contract`. No per-image mean or
std is computed. The 4,000-patch loader check used an absolute tolerance of
0.01 for per-band mean 0 and std 1 and passed.

Each quicklook has three 120×120 panels: RGB, B08, and a mask (red = any
B04–B09 band equals 1, yellow = all twelve bands equal 1). The all-band-1
mask addresses a visible scene-edge fill pattern; band-specific 1 values
remain an open source-data interpretation issue.

Reproduce the study from an explicitly extracted train-only patch root with
`python3 -m scripts.study_bigearthnet_s2 --root <S2-root> --ids
data/manifests/bigearthnet/stage1-norm-train-ids.v1.json --scratch <temp-dir>
--quicklooks results/stage1-normalization-quicklooks --output <study.json>`.
Then run `python3 -m scripts.freeze_bigearthnet_norm --study <study.json>
--ids <ids.json> --pixels <temp-dir>/resampled-float32.dat --contract
<norm_contract.json> --report <options.json>`. This loop did not train a model.

## Stage-1 Qwen engineering limitations

The pinned Transformers 4.49.0 vision path casts attention q/k to float32 in
`apply_rotary_pos_emb_vision`, even when the tower weights and input are
float64. The rotary frequency buffer starts as float32 after meta
initialization and becomes float64 when the tower is cast to double. Hooks in
`results/stage1-loop6-canary-explanation.json` record the actual operand
dtypes. The projection-only float64 perturbation response doubled exactly;
the full-tower comparison has a float32 precision boundary. Its exact-zero
result therefore cannot serve as a pure float64 equivalence proof. The
evidence for the 12-channel conversion is the genuine float64 projection
match, unchanged downstream weights, and the earlier fp32 noise-floor check.
The casts do not by themselves prove the cause of every nonlinear tower
perturbation ratio observed in loop 5.

## Stage-1 T4 evidence package (loop 7)

`packages/stage1-t4-smoke` now carries a five-stage runner (`run.py all`) that
produces all Stage-1 T4 evidence in one Kaggle session: verify (pinned
revision plus both weight SHA256; abort on mismatch), 50-step smoke (finite
losses and finite vision output, with one `config-vision-fp32.yaml` fallback
attempt that is recorded and then reused by later stages), resume continuity
(checkpoint at step 25, reload, finish; per-step losses compared against an
uninterrupted 0-50 run at max |diff| <= 0.05, per-step LR equality, canonical
optimizer-state digests), a 16-patch overfit sized from measured smoke
throughput to a 20-minute budget (final loss <= 0.2x initial, no NaN, no >2x
early spike, greedy decode reproduces >= 12/16 captions, and an image-shuffle
control measured eval-only at the trained state: shuffled-image loss > 1.10x
true-image loss), and an extrapolation of epoch hours for all 463,932 train
captions from the measured rate (2xT4 DDP is not extrapolated without a
measured run). Thresholds are pre-registered in
`results/stage1-loop7-preregistration.json`; each stage writes its own JSON
with full provenance and the session stops at the first failed stage (max two
attempts per problem, failures kept as separate files). The README documents
both weights sources (HF download at the pinned revision with internet on, or
a private Kaggle dataset with internet off); the runner verifies the pin
either way and no weights are bundled or downloaded in the repo. CPU unit
tests cover stage order, abort-on-mismatch, resume logic, and manifest
integrity with mocked runs; no training ran in this loop.

## Stage-1 shards and the full-run stage (loop 8)

[Stage-1 training shards](../results/stage1-loop8-preregistration.json) are built by
`scripts/build_stage1_shards.py`, which reads the frozen split manifest and the
`type=='captioning'` rows of `BigEarthNet.txt`, then makes **one** sequential
pass over `BigEarthNet-S2.tar.gz{aa,ab}`. Nothing is extracted to disk: the
archive decompresses to roughly 0.5 TB, so selected members are decoded from
memory and dropped.

The shard format is native per band, because BigEarthNet V2 stores each band at
its own ground sampling distance — measured from the archive, not assumed:
**10 m** `B02 B03 B04 B08` at 120×120, **20 m** `B05 B06 B07 B8A B11 B12` at
60×60, **60 m** `B01 B09` at 20×20. A patch is 160,000 uint16 bytes, not the
345,600 bytes a uniform 120×120 twelve-band cube would need. The frozen
[norm_contract.json](manifests/bigearthnet/norm_contract.json) is applied at
load time — rasterio bilinear to 120×120, divide by 10,000, clip to
[0, 0.9113], per-band train-fitted mean/std — through
`data/bigearthnet_s2.apply_contract`, and
`data/test_bigearthnet_shards.py` asserts that a shard row and the equivalent
extracted TIFF directory give identical normalized tensors.

A matching **eval-side** shard (2,000 patches from the 6 eval tiles only,
0 rejected, 0 shortfall, 320,002,878 bytes; measured tiles disjoint from
the train shard and 0 shared patch IDs) is what `--eval-shards` consumes
for the periodic held-out caption loss. Because the pixels are derived
and gitignored, both shard manifests are published under
`manifests/bigearthnet/` (`stage1-shards-train-2000.v1.json`,
`stage1-shards-eval-2000.v1.json`) as the integrity record.

The measured proof shard (`results/stage1-loop8-shard-build.json`) is 2,000
patches across all 48 train tiles, 40–42 patches per tile, 0 rejected, 0
shortfall, 320,002,878 bytes on disk (deflate would be 0.699×, not adopted).
Linear projections for the pre-registered N grid: 100k → 16.0 GB, 150k → 24.0
GB, 250k → 40.0 GB, 422,809 → 67.6 GB. Above N=150k these exceed the default
30 GB session budget, and the full train set exceeds the local 40 GiB buffer
rule, so **N is a parameter**, not a decision: the builder refuses a build that
would cross either limit.

Selection mirrors the loop-4 style — `numpy.default_rng(seed)`, sorted train
tiles, round-robin quotas capped by each tile's available count, no replacement —
at the new seed 26168. Rejected candidates (off-size band, missing band, raw DN
0, which the contract cannot normalize) are recorded with their reason and the
shortfall is reported; nothing is refilled silently.

`scripts/run_stage1_full.py` is the long-run stage: shard-backed samples whose
order is a pure function of (seed, epoch), a checkpoint every 500 steps holding
adapter, optimizer, scheduler, data position and RNG state, a wall-clock guard
that writes a final checkpoint and exits cleanly, and caption-only eval loss
every 2,000 steps on an eval-tile shard. It records measurements only; there is
no learning gate. Gates, the N selection rule, and the acceptance rules are
pre-registered in `results/stage1-loop8-preregistration.json`, together with the
three recorded amendments (native per-band GSD, selection ordering, B08
correction) and the two failed build attempts kept as separate files.

## Loop 9: the 150k train build, and a throughput gate on the full stage

Preregistration: `results/stage1-loop9-preregistration.json`, with two amendments
recorded before attempt 2 and the failed attempt kept as
`data/manifests/bigearthnet/stage1-shards-train-150000.attempt1-aborted-archive-order.v1.json`
plus `logs/loop9-build.attempt1-aborted.log`.

**Selection order is prefix-stratified.** `select_patch_ids` (loop 8, untouched) emits
tile-blocked prefixes: its first 10,000 patches touch 3 tiles with a 1,535 spread. The
loop-9 `select_patch_ids_interleaved` round-robins the selection across tiles, so measured
from the selection list: 10k → 48 tiles, min 40 / max 212, max deviation 1.946 pp;
50k → 48 tiles, min 40 / max 1,113, 1.459 pp; 100k → 48 tiles, min 40 / max 2,390, 0.791 pp;
150k → 48 tiles, min 40 / max 3,845, 0.0 pp. The pre-registered `max - min <= 1` gate was
unattainable by construction (availability spans 40 … 31,770) and was replaced *before any
build* by amendment 0, the proportional gate of 3 percentage points. Nothing else was
loosened.

**Attempt 1 aborted.** The first build wrote shards in *tar arrival order*, and the archive
is tile-major, so `shard-00000` held 2,000 patches of `T34VER` alone and every shard prefix
was tile-blocked. The selection was right; the builder ignored it. The 40 unusable shards
(12.8 GB) were deleted, the manifest kept as an attempt file.

**Fix: one archive pass per shard window.** Shard membership is fixed by selection rank, so
`build_shard_set` now buffers `shard_window` shards at once (default 30% of physical memory ÷
320 MB = 16 here) and assigns each accepted patch to `shard = selection_rank //
patches_per_shard`, sorting rows inside a shard by selection rank. That costs
`ceil(75/16) = 5` passes over 63.5 GB instead of one — recorded as amendment 1, with the
0.288 s/patch speed guard *not* relaxed for it. `test_build_shard_set_membership_follows_the_selection_not_the_archive`
and `test_build_shard_set_windows_produce_identical_shards` pin the behaviour.

**Measured build (attempt 2, complete).** 75 shards × 2,000 = 150,000 patches, 24,000,215,850
bytes on disk (160,001.4 B/patch), 0 rejected out of 150,000, over 5 passes of
[452.04, 446.82, 446.42, 498.28, 435.49] s — 2,285.78 s total, **0.0152 s/patch** and 10.50
MB/s of output. The shard-10 speed checkpoint read 0.0224 s/patch against the 0.288 limit
(12.9× margin); minimum free disk was 57.66 GB against the 45 GB floor. Extraction never
touches disk (rasterio `MemoryFile`), so the extraction temp peak is 0 by construction and
`find` over `TMPDIR` reports no leftovers.

**Independent verification** (`results/stage1-loop9-shard-verification.json`) re-hashes all 75
shards against the published manifest and passes all eight checks: hashes, sizes, split purity
against the split CSV, no duplicate IDs, no overlap with the eval shard, all 12 bands, captions
joined (coverage 1.0), row counts. Read back from the written shard files, the prefix table
matches the selection list exactly — 5 shards/10k → min 40 max 212; 25 shards/50k → min 40 max
1,113; 50 shards/100k → min 40 max 2,390; 75 shards/150k → min 40 max 3,845 — which is the
proof that what reached disk *is* the prefix-stratified selection. Per-shard tile counts fall
from 48 tiles toward 32 because tiles with small availability (40, 246, 545, 654 …) exhaust
before the end; that is the proportional gate working, not a bug.

**Upload plan only** (`results/stage1-loop9-upload-plan.json`): 5 parts of 15 shards,
`satquery-stage1-train-shards-part01` … `part05`, 4.83 GB each, 24.00 GB measured total, each
with a `results/stage1-loop9-upload-sha256/<part>.sha256.txt`. Nothing was uploaded and no part
archive was assembled. **Kaggle's size limits were not checked — UNVERIFIED.**

**Plan-from-throughput.** `scripts/run_stage1_full.py` now refuses to start `full` without a
throughput JSON: `load_throughput` demands `measured_samples_per_second`, `shard_manifest_sha256`,
`config_sha256`, `device` and `dtype` and rejects a bad rate or a mismatched manifest/config;
`plan_from_throughput` computes `floor(budget_seconds × samples/s / effective_batch)` and refuses
a zero-step plan; `run_full` writes `plan`, `n_used`, `steps_planned` and `epochs_planned` into the
results JSON before the first optimizer step. `config-full.yaml` carries explicit `steps: null`
and `max_steps_per_session: null` — no hidden placeholder step counts. 16 CPU tests in
`scripts/test_stage1_full.py` cover it with mocked records. No T4 throughput exists yet, so the
guard has never been exercised against a real measurement.

The earlier sentence in this file that `run_stage1_full.py` "records measurements only; there is
no learning gate" is superseded: it now has a throughput gate on *starting*, still no gate on
*learning quality*.

## Loop 10: pixel/ID coupling failure, writer fix, rebuild pending

Pre-registration: `results/stage1-loop10-preregistration.json`; training fix tests:
`results/stage1-loop10-training-test-preregistration.json`. No real shards were
rewritten or deleted, and no model was loaded or trained.

The independent archive probe recovered all arrival IDs for old train shard 0.
Its first ten pixel rows matched those arrival IDs exactly and matched none of
the corresponding sidecar IDs. The loop-9 writer filled pixels in archive
arrival order, then sorted IDs alone by selection rank. Arrival order is
recoverable for this shard from canonical TIFF completion order in the unchanged
archive and window selection; this is evidence of the cause, not a repair.
Loop-8 `build_shards` writes pixels and sidecars in the same arrival order; ten
loop-8 train and ten eval patches passed exact native-array checks. The smoke
package uses a separate extraction path; three patches passed exact source
checks and all package ID/TIFF-name pairs were checked. Evidence:
`results/stage1-loop10-root-cause-attempt2.json` and
`results/stage1-loop10-package-pairing.json`. Probe attempt 1 is preserved.

The writer now holds each ID/caption/tile/source-band set together as one record
and sorts whole records. Each record carries a source-read `native_sha256`:
for each canonical band, SHA256 receives `band\0uint16\0HxW\0`, then little-endian
C-order native uint16 bytes. Resume refuses legacy or checksum-invalid shards.
The regression failed on old code, failed on fix attempt 1 due to a local
variable shadowing error, and passed on fix attempt 2; all outputs are preserved.

The verifier recomputes native checksums for every row. Independently decoded
source TIFFs and BEN.txt caption rows are compared on a fixed-seed sample of
at least max(100, 40 × ceil(shards/10)) patches, with at least forty in each group
of ten shards. Tiny synthetic fixtures check every available row. No pixel,
checksum, or caption mismatch is allowed. On one unchanged old train shard,
all 2,000 source checksums were absent, all 1,200 native-band checks across 100
sampled patches failed, and captions passed. See
`results/stage1-loop10-old-shard-verifier.json`. Existing eval sidecars predate
these checksums: retain the independent eval audit evidence rather than mutate
the frozen eval shard or claim it passes the new train-checksum gate.

Rebuild is **not authorized**. The exact checked selection is
`data/manifests/bigearthnet/stage1-train-rebuild-ids-loop10.v1.json`.
`results/stage1-loop10-rebuild-plan.json` records measured disk sizes/free space,
new-directory projections, phase sizes, and the required approval. Phase 1
uses the first 100,000 IDs of this 150,000-ID list; phase 2 extends the same
selection to 150,000. A fresh n=100,000 selection is not equivalent.
Projected durations from the requested 0.144 s/patch are four and two hours;
these are projections and exclude changes in archive-pass/checksum cost.
The 45 GB guard prevents building while retaining the old shards on this volume.
After an approved rebuild, run full pixel/caption verification and audit A1–A5.

Training-code corrections append caption EOS, retain frozen LM fp16 weights
and fp32 trainable masters, apply CUDA fp16 autocast/GradScaler, save all
trainable weights plus optimizer/scaler/scheduler/RNG/data position, and enforce
the original total-step schedule on resume. The fp32 vision fallback disables
vision autocast; eval checks the wall budget between samples. Shuffle evaluation
now measures all sixteen package images. No initialization, contract, split,
checkpoint, or eval pixels changed. CPU mocked evidence does not establish T4
numerical stability or learning quality.

Open deployment issues: header-derived peak-memory estimate exceeds the 15 GB
T4 audit limit; retry checkpoints can exceed the 20 GB working budget. Keep
pinned weights outside working storage. The full stage requires a throughput
JSON matching its train-shard manifest and config, while the current 16-patch
smoke produces a different record; no matching sharded-throughput producer was
added in this repair loop. Full config budget is 180,000 seconds, not a measured
or authorized thirty-hour run. See `results/stage1-loop10-resource-estimates.json`
and the final loop-10 audit report for checks and limitations.
