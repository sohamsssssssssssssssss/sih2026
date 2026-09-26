"""Geographic leakage controls: spatial blocks, block-level splits, and
cross-split neighbour distances.

Method (see docs/research/geographic-leakage-controls.md):
1. Project lon/lat to EPSG:6933 (equal-area metres) and bucket each sample
   into a square block of `block_size_m`.
2. Assign whole blocks to splits: blocks are ordered by a seeded SHA-256
   and filled greedily to the target fractions by sample count, so no block
   straddles two splits.
3. Audit any split (new or official) by the great-circle distance from each
   evaluation sample to its nearest training sample.

Block assignment bounds leakage only at block edges: two samples in adjacent
blocks can still be close. The audit in step 3 measures what remains; choose
`block_size_m` well above the spatial autocorrelation range of the task and
treat the audit, not the block size, as the evidence.
"""

import hashlib
from collections import defaultdict
from typing import Iterable

import numpy as np
from rasterio.warp import transform
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8
EQUAL_AREA_CRS = "EPSG:6933"


def block_ids(lons: Iterable[float], lats: Iterable[float], block_size_m: float) -> list[str]:
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    xs, ys = transform("EPSG:4326", EQUAL_AREA_CRS, list(lons), list(lats))
    return [f"{int(np.floor(x / block_size_m))}_{int(np.floor(y / block_size_m))}" for x, y in zip(xs, ys)]


def assign_blocks(blocks: list[str], fractions: dict[str, float], seed: int) -> list[str]:
    """Split label per sample; every sample in one block gets the same split."""
    if not fractions or any(value < 0 for value in fractions.values()):
        raise ValueError("fractions must be non-negative")
    total = sum(fractions.values())
    targets = {split: len(blocks) * value / total for split, value in fractions.items()}
    members: dict[str, int] = defaultdict(int)
    for block in blocks:
        members[block] += 1
    order = sorted(members, key=lambda block: hashlib.sha256(f"{seed}:{block}".encode()).hexdigest())
    filled = dict.fromkeys(fractions, 0)
    chosen = {}
    for block in order:
        # ponytail: greedy fill by largest remaining deficit; ILP if exact fractions ever matter
        split = max(fractions, key=lambda name: (targets[name] - filled[name], -list(fractions).index(name)))
        chosen[block] = split
        filled[split] += members[block]
    return [chosen[block] for block in blocks]


def _unit_vectors(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lons), np.radians(lats)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def nearest_distances_m(query_lonlat: np.ndarray, reference_lonlat: np.ndarray) -> np.ndarray:
    """Great-circle distance (m) from each query point to its nearest reference point."""
    if len(reference_lonlat) == 0:
        return np.full(len(query_lonlat), np.inf)
    tree = cKDTree(_unit_vectors(reference_lonlat[:, 0], reference_lonlat[:, 1]))
    chord, _ = tree.query(_unit_vectors(query_lonlat[:, 0], query_lonlat[:, 1]))
    return 2 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))


def leakage_audit(
    lons: Iterable[float],
    lats: Iterable[float],
    splits: Iterable[str],
    train_split: str = "train",
    radii_m: tuple[float, ...] = (1_200, 10_000, 50_000),
) -> dict:
    """Per non-train split: nearest-train distance summary and counts within each radius."""
    coordinates = np.column_stack([list(lons), list(lats)]).astype(float)
    labels = np.asarray(list(splits))
    train = coordinates[labels == train_split]
    report = {}
    for split in sorted(set(labels.tolist()) - {train_split}):
        distances = nearest_distances_m(coordinates[labels == split], train)
        report[split] = {
            "samples": int(distances.size),
            "nearest_train_m": {
                "min": float(distances.min()),
                "p05": float(np.percentile(distances, 5)),
                "median": float(np.median(distances)),
            },
            "within_radius": {str(int(radius)): int(np.count_nonzero(distances <= radius)) for radius in radii_m},
        }
    return report
