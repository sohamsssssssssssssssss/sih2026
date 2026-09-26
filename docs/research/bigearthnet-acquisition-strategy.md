# BigEarthNet-MM bounded acquisition strategy (Day 18)

Status, 2026-09-26: **no BigEarthNet S1/S2 imagery acquired.** The tracked pilot manifests are verified for internal consistency. Each acquisition option is quantified below. The recommended option needs a user decision because it exceeds this track's 5 GB download budget or changes the data's provenance.

## What was verified from tracked data (no imagery)

`python -m data.bigearthnet_pilot_consistency` → `data/manifests/bigearthnet/pilot-consistency.v1.json`, registry `SQ-20260926-409`.

| Check (100 pilot pairs) | Result |
|---|---|
| Manifest SHA-256 `4bd30aef…1541` recomputed | pass |
| S1 name and S2 patch name share tile and x/y index (pair identity) | 100/100 |
| Reference map present for the patch | 100/100 |
| Reference-map CRS is the UTM zone of the S2 tile | 100/100 (EPSG 32629–32635) |
| Reference map is 120 × 120 at 10 m | 100/100 |
| Reference-map timestamp equals the S2 patch sensing time | 100/100 |
| Declared lat/lon lies inside the reference-map footprint | 100/100 |
| Manifest labels equal the 19-class mapping of the reference-map CLC classes | 100/100 |
| S1/S2/reference member lists equal the manifest's expected assets | 3/3 lists |
| \|S1 − S2\| acquisition separation | min 4.7 h, median 17.2 h, max 55.2 h |
| Cloud/shadow or seasonal-snow flags | 0 / 0 |

Label agreement is **consistency, not accuracy**: labels and reference maps both derive from CLC2018 v2020_u1.

**Unverified without imagery:** S1/S2 archive member existence; native S1/S2 CRS, transforms, nodata and units; S1 polarisation content and calibration; pixel-level S1–S2 alignment.

## Facts about the official distribution (measured today)

- Zenodo record 10891137 (BigEarthNet v2.0.0, CDLA-Permissive-1.0) is unchanged since the phase-2 handoff. `BigEarthNet-S1.tar.zst` is 54,439,153,171 B; `BigEarthNet-S2.tar.zst` is 63,251,710,377 B; together **117,690,863,548 B**.
- Each archive is **one zstd frame**. The first frame's declared content size is 65,833,062,400 B for S1 and 102,406,983,680 B for S2, i.e. the whole tar. The last 4 bytes are not the seekable-zstd magic `b1ea928f`. So there is **no frame-level random access**: extracting any member means decompressing the stream from byte 0 up to that member.
- Zenodo honours HTTP byte ranges for GET, so resumable full downloads work, but ranges cannot address individual members.
- The official BIFOLD Hugging Face organisation hosts only `BigEarthNet.txt` (text parquet, revision `72d865f…`, unchanged). Its README directs users to download S1/S2 from bigearth.net (the same Zenodo archives) and convert them with `rico-hdl` to LMDB. **No official per-patch or chunked imagery distribution was found.**
- Bandwidth from this machine to Zenodo, 64 MiB range samples: 5.59 MB/s, then 0.77 MB/s (`SQ-20260926-412`; the network was shared with concurrent agent downloads). A full 117.7 GB transfer takes **~5.8–42 h**.

Probe cost: ~128 MiB transferred in total; no archive bytes were stored.

## Options

| # | Option | Transfer | Peak disk | Provenance | Verdict |
|---|---|---|---|---|---|
| A | Download both archives, verify MD5, extract `s1-members`/`s2-members` (phase-2 handoff plan) | 117.7 GB | ~118 GB + extraction | Official, exact | Correct, but exceeds this machine's free disk (~80 GB shared) |
| B | Stream: `curl … \| zstd -d \| tar -x -T members` per archive, stopping early with `bsdtar -q` (fast-read) once all members are found | up to 117.7 GB (member order unknown; the pilot spans 28 tiles and 2017–2018 dates, so assume most of it) | **< 100 MB** (pilot members only) | Official, exact; MD5 **cannot** be verified on an early-stopped stream, and on a full stream only by `tee` into `md5` | Feasible on this machine; 6–42 h at measured rates; an interruption restarts that archive from byte 0 |
| C | Full download on a remote volume (Kaggle dataset, cloud VM) with ≥ 125 GB, extract the pilot there, copy back only the selected members | 117.7 GB remote | remote ≥ 125 GB; local < 100 MB | Official, exact, MD5-verifiable | Best provenance-to-cost ratio if a ≥ 125 GB volume is available |
| D | Re-acquire the **same acquisitions** from Planetary Computer (the source products of 98/100 pilot pairs are available: S2 100/100, S1 98/100; `SQ-20260926-411`) and attach the official reference maps (282 MB archive) | ~10–30 MB of COG windows + 282 MB | < 400 MB | **Not BigEarthNet pixels**: BigEarthNet S2 carries processing baseline N9999 and the dataset authors preprocessed S1, whereas PC serves ESA/ESRI L2A (baselines 02.12/03.00 here) and γ⁰ RTC | Useful for SatQuery-specific S1/S2 experiments with CLC reference maps; must never be reported as a BigEarthNet result |
| E | Third-party mirrors (TorchGeo HF copy, Kaggle subsets) | varies | varies | Unofficial; the phase-2 audit found v1 or Serbia-only subsets | Rejected: provenance |

## Recommendation

1. **For a BigEarthNet-comparable result, use option C.** Run it on a remote volume with ≥ 125 GB (e.g. a Kaggle notebook with a ≥ 125 GB scratch disk, or a cloud VM). Verify both official MD5s. Extract only the member lists already tracked in `data/manifests/bigearthnet/`, and bring back the < 100 MB pilot. Commands are in `docs/research/phase-2-data-acquisition-handoff.md`.
2. **If no remote volume is available, use option B** on this machine, one archive at a time, overnight. `tee` the stream through `md5` and run it to completion, so the official checksum is still verified; drop `-q` early-stopping in that case. Budget one full transfer per archive.
3. **Option D may be used now** for SatQuery's own sensor-necessity or fusion experiments that need CLC reference maps. Every result must be labelled "BigEarthNet-referenced re-acquisition (Planetary Computer)", not BigEarthNet.

**Needs user approval** (above the 5 GB budget): options A, B and C. Option D's 282 MB reference-map archive is within budget, but it was not downloaded in this pass because no downstream experiment consumed it yet.

## Geographic control for whatever is acquired

Use the official split as-is for test and bench. See `docs/research/geographic-leakage-controls.md`: the official **test** split is ≥ 11.95 km from every train patch, but the official **validation** split touches train. Validation is fit for model selection only and must not be used as a held-out result.
