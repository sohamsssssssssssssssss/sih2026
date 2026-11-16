# Source of the BigEarthNet land-cover baseline evidence

| Item | Value |
|---|---|
| Run | Kaggle notebook `ben-landcover-v1`, GPU Tesla T4, AMP on; torch `2.11.0+cu130`; finished 16 Nov 2026 (progress log 20:09–20:23, notebook clock) |
| Code | commit `af4b4c4` on `feature/ben-landcover`; zip `satquery-ben-code.zip` sha256 `fdd8ec1a568cdcc36a7241c6ac26783637038ea5b99740d2639519c616aafa53`; `ben/landcover.py` sha256 `faa4977aaed59fd66f5a60bb5e7cb8fb11d53ba9626f9b0f0bf6d57a03e149f6` |
| Pack | `ben-pilot-20k` (BigEarthNet v2 / reBEN pilot, Europe, 10 m); `MANIFEST.json` sha256 `dc9c43adf76103811a579060fae0688ef9b281d610e1d2e27e5e8ee81c1ae876` |
| Split | train 12,000 / validation 2,000 / test 6,000 (official metadata split) |
| Run timings (`runtime_s` in each JSON) | s1 342.5 s, s2 274.0 s, s1s2 334.7 s |

Each run JSON records `git_sha: null` because the Kaggle copy has no `.git`. Its `landcover_py_sha256` equals the sha256 of `ben/landcover.py` at `af4b4c4`, and its `manifest_sha256` equals the pack value above. Both were checked on 16 Nov 2026.

## Evidence files (sha256)

| File | sha256 |
|---|---|
| `ben-landcover-s1.json` | `de53270156bcd63e486242231f6c0794b1e17bdf68a68c9d986b2ca156889b5c` |
| `ben-landcover-s2.json` | `c6f3b8fbda6f48ee575eac0c08b2a641f247c3edcb2f1c8fd0e2089d76a9ba03` |
| `ben-landcover-s1s2.json` | `88c1243d980ea4909b717732747b7a3d165f7a332bfef643f0d694e9ece59848` |
| `ben-landcover-compare-s2-vs-s1.json` | `ef3907301fc7f87b6edf4b0b04034bb3cd0b71185130e5ecde4fcc9a3981d915` |
| `ben-landcover-compare-s1s2-vs-s2.json` | `16528ce08efd535f2c8d78068017d3abf9caf056e493fec6769ea084d286ac35` |
| `ben-landcover-progress.txt` | `799f24b67dfca2a5cd604990c9145a29530443be2f3a1e8eed818b1690c85539` |
| `ben-landcover-s1-test-logits.npy` | `5a504e1c71a5475492ff635972233f79249e91936e5d274bd1a5d8bcb522fa5a` |
| `ben-landcover-s2-test-logits.npy` | `9e06825886a385056a5102b48e8d39647c9f41467ef11a4b9db592aa524f2b5f` |
| `ben-landcover-s1s2-test-logits.npy` | `01a9044703a467c7b1bcbfb18cf983841095b25920d1a7b685135b0945d7fc6b` |
| `ben-landcover-{s1,s2,s1s2}-test-labels.npy` (identical) | `ac3059ad48f83b08a173d03c9e87e1bfab59ddcb0c5d7689d0811c5f2154ebaa` |
| `ben-landcover-{s1,s2,s1s2}-test-idx.npy` (identical) | `d77c2302417e52bef639596e186833555400494d324b277ab2eff2f0a82187a5` |

`analysis.json` is generated from these files by `python3.11 analyze_ben_landcover.py --json analysis.json`.
