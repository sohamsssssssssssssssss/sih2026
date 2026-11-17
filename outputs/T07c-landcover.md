# T07c Part A — land cover

Copied with git checkout feature/ben-landcover -- ben/landcover.py ben/transfer.py.
Source commit: 4707d4c. No checkpoints or datasets copied.

Five classes: trees, shrub_grass, cropland, built, water; wetland excluded. Scores are
sigmoid(max member logit), not area fractions or calibrated confidence. All tools
remain experimental_unvalidated. CPU only, lazy weights, checkpoint paths in
configs/landcover.json; exact BEN TRAIN normalization retained.

Inputs: finite CHW 120x120 array for a single sensor, with Scene bands/units in
exact BEN order. Joint input is a dict with s1/s2 entries, each containing
pixels, bands and units (dB or reflectance_x1e4). This avoids putting mixed units
in Scene.units. A cloudy joint scene selects s1; otherwise it selects s1s2.

| Mode | Status | CPU | Cold ms | Warm ms |
|---|---|---|---:|---:|
| s1 | experimental_unvalidated | yes | 1243 | 3 |
| s2 | experimental_unvalidated | yes | 106 | 3 |
| s1s2 | experimental_unvalidated | yes | 102 | 4 |

One real BEN test patch, pack row 14000; one cold and one warm call per mode.
These latency samples are not a benchmark. Detailed results and checkpoint hashes
are in T07c-landcover.json.

Gate: python3.11 -m pytest core orchestrator -q -p no:cacheprovider —
486 passed, 2 skipped, 7 subtests passed. Synthetic and all three real-checkpoint
tests passed. No real-weight test was skipped. Router route name is now landcover
as requested; the historical T08 report describes the previous placeholder name.
