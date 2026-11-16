# T14 Sen1Floods11 hand-labeled pack

Pack only: no training. Python 3.11 with numpy, rasterio, pandas, pyarrow and pytest.
The existing local `~/ben-work/.venv-torch/bin/python` supplies these dependencies.
No dependency installation or credentials are required for this builder.

Run from the isolated `task/T14-flood-pack` worktree:

```bash
python3.11 -m pytest data/flood/test_build_pack.py -q
# Commit the builder first, so builder_commit identifies its actual code.
python3.11 data/flood/build_pack.py
python3.11 -m pytest data/flood/test_real_pack.py -q -s
```

Defaults: raw files and URL/SHA-256 fetch log in `~/satquery-data/flood/raw/`;
packed arrays in `~/satquery-data/flood/packs/sen1floods11-hand-v1/`.
Only official HandLabeled layers are fetched. HTTP/access errors stop the fetch;
no credential fallback or network retry. Cached files must match their logged
hash. A pre-existing pack is never overwritten. To verify without downloading:

```bash
python3.11 data/flood/build_pack.py --verify-only
```

Each official split (train, valid, test, bolivia) contains five NCHW arrays:
`s1.npy` Float32 VV/VH, `s2.npy` Int16 13-band native scaled DN in the documented
canonical order, `label.npy`/`otsu.npy` Int8, and `jrc.npy` UInt8. Chip rows retain
CSV order. Index and manifest link chip IDs to rows and retain quality flags,
source CRS/transform/encoding, per-array chip hashes and train-only band moments.
No pixel is resampled and no chip is dropped. The build reopens a seed-17 random
5% (rounded up); the separate real-pack test checks every file and every chip.
The manifest's own hash is printed separately, avoiding a self-referential hash.

See DECISIONS.md for the approved 1e-6-pixel transform tolerance and quality
semantics. Pixel-size and origin maxima are normalized against S1's native affine
pixel basis, recorded per chip, and independently recomputed by pack verification.
S2 raw threshold counts have unknown physical meaning, not a saturation claim.

Licence: official STAC source and label collections say `proprietary`; reuse terms
are unknown. **Redistribution: not allowed until licence is confirmed.** Do not
publish or upload the pack. Raw and packed arrays must stay outside Git. Source
README statements (including nominal 10 m, S1 dB, S2 L1C/scale) are distinguished
from measured raster properties; all chip metadata are retained in the manifest.
