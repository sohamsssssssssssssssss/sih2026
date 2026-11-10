"""Stage-1 training shards: seeded selection, streaming build, resumable read.

A shard stores native-resolution uint16 pixels for all twelve S2 bands plus one
caption row per patch. Nothing is normalized or resampled here: the frozen
contract in data/bigearthnet_s2.py is applied at load time, so a shard pixel
array and an extracted patch directory produce identical model inputs.

The builder makes exactly one sequential pass over the gzip tar. tar.gz is not
seekable, so the selected patch set must be known before the pass starts and
every archive member is decompressed in stream order regardless of whether it is
kept. Selected members are decoded from memory (rasterio MemoryFile) and never
written to disk, which is what keeps a 63.5 GB archive build inside the local
disk budget.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tarfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from data.bigearthnet_s2 import BANDS, NATIVE_SIZE, SIZE

def native_patch_sha256(bands: dict[str, np.ndarray]) -> str:
    """SHA256 v1: band\0uint16\0HxW\0 followed by little-endian C-order pixels."""
    digest = hashlib.sha256()
    if set(bands) != set(BANDS):
        raise ValueError("Native checksum requires exactly the canonical twelve bands")
    for band in BANDS:
        array = bands[band]
        size = NATIVE_SIZE[band]
        if array.dtype != np.uint16 or array.shape != (size, size):
            raise ValueError(f"Invalid native band {band}: {array.dtype}/{array.shape}")
        digest.update(f"{band}\0uint16\0{size}x{size}\0".encode())
        digest.update(array.astype("<u2", copy=False).tobytes(order="C"))
    return digest.hexdigest()


SELECTION_ALGORITHM = (
    "numpy.default_rng(seed); train tiles in sorted order; patch IDs sorted "
    "within a tile; round-robin equal quotas capped by each tile's available "
    "count; no replacement; per-tile take order is that tile's rng permutation; "
    "shard order is the selection order"
)


# --------------------------------------------------------------------------- #
# split + captions
# --------------------------------------------------------------------------- #
def load_split_rows(manifest: Path) -> dict[str, dict[str, str]]:
    """Return patch_id -> {mgrs_tile, geo_split, country, official_split}."""
    rows: dict[str, dict[str, str]] = {}
    with gzip.open(Path(manifest), "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            patch_id = row["patch_id"]
            if patch_id in rows:
                raise ValueError(f"Duplicate patch in split manifest: {patch_id}")
            rows[patch_id] = {
                "mgrs_tile": row["mgrs_tile"],
                "geo_split": row["geo_split"],
                "country": row["country"],
                "official_split": row["official_split"],
            }
    if not rows:
        raise ValueError("Split manifest is empty")
    return rows


def select_patch_ids(rows: dict[str, dict[str, str]], side: str, n_total: int, seed: int) -> list[str]:
    """Seeded, tile-stratified, deterministic selection for one geo side.

    Quotas are dealt round robin across the side's tiles (one extra patch per
    tile in sorted order until the quota is used) and capped by that tile's
    available count; surplus from an exhausted tile is not redistributed, so a
    capped tile can leave the selection shorter than ``n_total``. The returned
    order is grouped by tile, matching the accepted loop-4 selection.
    """
    if side not in {"train", "eval"}:
        raise ValueError(f"Unknown geo side: {side}")
    if n_total < 1:
        raise ValueError("n_total must be positive")
    by_tile: dict[str, list[str]] = defaultdict(list)
    for patch_id, row in sorted(rows.items()):
        if row["geo_split"] == side:
            by_tile[row["mgrs_tile"]].append(patch_id)
    tiles = sorted(by_tile)
    if not tiles:
        raise ValueError(f"No patches on the {side} side")
    rng = np.random.default_rng(seed)
    quotas = {tile: 0 for tile in tiles}
    dealt = 0
    while dealt < n_total:
        progressed = False
        for tile in tiles:
            if dealt >= n_total:
                break
            if quotas[tile] >= len(by_tile[tile]):
                continue
            quotas[tile] += 1
            dealt += 1
            progressed = True
        if not progressed:
            break
    selected: list[str] = []
    for tile in tiles:
        available = by_tile[tile]
        if quotas[tile] == 0:
            continue
        order = rng.permutation(len(available))[: quotas[tile]]
        selected.extend(available[int(index)] for index in np.sort(order))
    if len(selected) != sum(quotas.values()):
        raise ValueError("Selection length disagrees with the allocated quotas")
    return selected


def assert_split_purity(selected: list[str], rows: dict[str, dict[str, str]], side: str) -> None:
    if len(set(selected)) != len(selected):
        raise ValueError("Selection contains duplicate patch IDs")
    for patch_id in selected:
        row = rows.get(patch_id)
        if row is None:
            raise ValueError(f"Selected patch is absent from the split manifest: {patch_id}")
        if row["geo_split"] != side:
            raise ValueError(f"Selected patch is on the {row['geo_split']} side: {patch_id}")


def load_captions(parquet: Path, patch_ids: list[str]) -> dict[str, str]:
    """Return patch_id -> caption for the requested captioning rows only."""
    import pyarrow.parquet as pq

    wanted = set(patch_ids)
    table = pq.read_table(parquet, columns=["patch_id", "type", "output"], filters=[("type", "=", "captioning")])
    patch_column = table.column("patch_id").to_pylist()
    output_column = table.column("output").to_pylist()
    captions: dict[str, str] = {}
    for patch_id, output in zip(patch_column, output_column):
        if patch_id in wanted:
            if patch_id in captions:
                raise ValueError(f"Multiple captioning rows for patch: {patch_id}")
            captions[patch_id] = output
    missing = wanted - set(captions)
    if missing:
        raise ValueError(f"No captioning row for {len(missing)} patches; first: {min(missing)}")
    return captions


# --------------------------------------------------------------------------- #
# shard layout
# --------------------------------------------------------------------------- #
def native_arrays(stack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Order a patch's native band arrays into the frozen band order."""
    return {band: stack[band] for band in BANDS}


def plan_shards(patch_ids: list[str], patches_per_shard: int) -> list[list[str]]:
    if patches_per_shard < 1:
        raise ValueError("patches_per_shard must be positive")
    return [patch_ids[start:start + patches_per_shard]
            for start in range(0, len(patch_ids), patches_per_shard)]


def shard_bytes(n_patches: int) -> int:
    """Uncompressed pixel bytes for a shard at each band's native resolution."""
    return n_patches * native_patch_bytes()


def native_patch_bytes() -> int:
    """uint16 bytes of one native patch: sum over bands of h * w * 2."""
    return int(sum(NATIVE_SIZE[band] ** 2 for band in BANDS)
               * np.dtype(np.uint16).itemsize)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def deflated_bytes(pixels: np.ndarray, level: int = 6) -> int:
    """Bytes the same array would occupy under a single deflate stream."""
    compressor = zlib.compressobj(level)
    total = 0
    flat = np.ascontiguousarray(pixels).view(np.uint8).reshape(-1)
    for start in range(0, flat.size, 1 << 22):
        total += len(compressor.compress(flat[start:start + (1 << 22)].tobytes()))
    total += len(compressor.flush())
    return total


# --------------------------------------------------------------------------- #
# streaming archive access
# --------------------------------------------------------------------------- #
class _ConcatStream:
    """Sequential read-only view over split archive parts, in order."""

    def __init__(self, paths: list[Path]):
        self._paths = [Path(path) for path in paths]
        for path in self._paths:
            if not path.is_file():
                raise FileNotFoundError(f"Archive part is missing: {path}")
        self._index = 0
        self._handle = self._paths[0].open("rb")
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        while True:
            data = self._handle.read(size)
            if data or size == 0:
                self.bytes_read += len(data)
                return data
            self._handle.close()
            self._index += 1
            if self._index >= len(self._paths):
                return b""
            self._handle = self._paths[self._index].open("rb")

    def close(self) -> None:
        self._handle.close()


def _member_key(name: str) -> tuple[str, str] | None:
    """Return (patch_id, band) for an S2 TIFF member, else None."""
    parts = name.split("/")
    if len(parts) != 4 or parts[0] != "BigEarthNet-S2" or not name.endswith(".tif"):
        return None
    patch_id, band = parts[2], parts[3][: -len(".tif")]
    prefix = f"{patch_id}_"
    if not band.startswith(prefix):
        return None
    return patch_id, band[len(prefix):]


def decode_member(payload: bytes) -> np.ndarray:
    """Decode one band TIFF; shape and dtype are judged by the acceptance rules."""
    with MemoryFile(payload) as memory:
        with memory.open() as source:
            if source.count != 1 or source.dtypes[0] != "uint16":
                raise ValueError("S2 band TIFF must be single-band uint16")
            if source.nodata not in (None, 0):
                raise ValueError(f"Unsupported nodata value: {source.nodata}")
            return source.read(1)


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_shards(
    archive_parts: list[Path],
    candidates: list[str],
    captions: dict[str, str],
    rows: dict[str, dict[str, str]],
    output_dir: Path,
    *,
    git_sha: str,
    seed: int,
    side: str = "train",
    patches_per_shard: int = 2000,
    norm_contract_path: Path | None = None,
    progress_every: int = 5000,
    log=print,
) -> dict:
    """One sequential archive pass; writes shards, sidecars, and the manifest.

    ``candidates`` is the stratified selection computed up front, exactly as the
    loop-4 selection was: only those patches are decoded, and they land in the
    shards in archive order. A candidate rejected by R1/R2/R3 is recorded and
    absent from the shards, so the accepted count can be lower than the
    selection; the shortfall is reported rather than silently refilled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        raise ValueError("Candidate selection is empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate selection repeats a patch")
    pending = set(candidates)
    rejected: dict[str, str] = {}
    inflight: dict[str, dict] = {}
    shard_count = -(-len(candidates) // patches_per_shard)
    buffers = [{band: np.zeros((patches_per_shard, NATIVE_SIZE[band], NATIVE_SIZE[band]),
                                dtype=np.uint16) for band in BANDS}
               for _ in range(shard_count)]
    accepted: list[str] = []
    members_seen = 0
    started = time.perf_counter()
    stream = _ConcatStream(archive_parts)
    stream_complete = True
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                members_seen += 1
                if not member.isfile():
                    continue
                key = _member_key(member.name)
                if key is None or key[0] not in pending:
                    continue
                patch_id, band = key
                if band not in BANDS:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Archive member is unreadable: {member.name}")
                entry = inflight.setdefault(patch_id, {"bands": {}, "shapes": []})
                array = decode_member(source.read())
                if array.dtype != np.uint16:
                    rejected[patch_id] = f"dtype:{array.dtype}"
                    inflight.pop(patch_id)
                    continue
                if array.shape != (NATIVE_SIZE[band], NATIVE_SIZE[band]):
                    entry["shapes"].append(f"{band}:{array.shape[0]}x{array.shape[1]}")
                entry["bands"][band] = array
                if len(entry["bands"]) < len(BANDS) and not entry["shapes"]:
                    continue
                pending.discard(patch_id)
                inflight.pop(patch_id)
                missing = [band for band in BANDS if band not in entry["bands"]]
                if entry["shapes"]:
                    rejected[patch_id] = "shape:" + ",".join(sorted(set(entry["shapes"])))
                elif missing:
                    rejected[patch_id] = "missing_bands:" + ",".join(missing)
                elif any(bool((entry["bands"][band] == 0).any()) for band in BANDS):
                    rejected[patch_id] = "raw_zero_dn"
                else:
                    slot = len(accepted)
                    target = buffers[slot // patches_per_shard]
                    row = slot % patches_per_shard
                    for band in BANDS:
                        target[band][row] = entry["bands"][band]
                    accepted.append(patch_id)
                if len(accepted) >= len(candidates):
                    stream_complete = False
                    break
                if progress_every and len(accepted) and len(accepted) % progress_every == 0:
                    rate = len(accepted) / max(time.perf_counter() - started, 1e-9)
                    log(f"accepted {len(accepted)}/{len(candidates)} patches "
                        f"({len(rejected)} rejected, {members_seen} members, {rate:.1f}/s, "
                        f"{stream.bytes_read / 1e9:.1f} GB streamed)")
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    for patch_id, entry in sorted(inflight.items()):
        missing = [band for band in BANDS if band not in entry["bands"]]
        rejected[patch_id] = "incomplete_in_archive:" + ",".join(missing or ["no_bands"])
    if not accepted:
        reasons: dict[str, int] = {}
        for value in rejected.values():
            reasons[value.split(":")[0]] = reasons.get(value.split(":")[0], 0) + 1
        raise ValueError(f"No selected patch could be built: {len(rejected)} rejected; "
                         f"reasons {reasons}; examples "
                         f"{dict(list(rejected.items())[:3])}")
    accepted_set = set(accepted)
    if len(accepted_set) != len(accepted):
        raise ValueError("Accepted patch list contains duplicates")
    never_seen = sorted(pending - accepted_set - set(rejected))

    plan: list[list[str]] = []
    for index in range(shard_count):
        group = accepted[index * patches_per_shard:(index + 1) * patches_per_shard]
        if group:
            plan.append(group)
    for patch_id in accepted:
        if captions.get(patch_id) is None:
            raise ValueError(f"Accepted patch has no caption: {patch_id}")

    shard_records = []
    # Report-only: the loop-3 "all twelve bands equal 1" pixel metric cannot be
    # computed across native GSDs without resampling, so per-band DN-1 fractions
    # are recorded instead. Not gated.
    per_band_one: list[float] = []
    total_raw = 0
    total_deflated = 0
    for shard_index, patch_ids in enumerate(plan):
        count = len(patch_ids)
        band_arrays = {band: buffers[shard_index][band][:count] for band in BANDS}
        stacked = np.concatenate([band_arrays[band].reshape(count, -1) for band in BANDS], axis=1)
        for band in BANDS:
            per_band_one.append(float((band_arrays[band] == 1).mean()))
        npz_path = output_dir / f"shard-{shard_index:05d}.npz"
        with npz_path.open("wb") as handle:
            np.savez(handle, **band_arrays)
        raw = shard_bytes(count)
        deflated = deflated_bytes(stacked)
        total_raw += raw
        total_deflated += deflated
        shard_records.append({
            "index": shard_index,
            "npz": npz_path.name,
            "npz_bytes": npz_path.stat().st_size,
            "npz_sha256": sha256_file(npz_path),
            "rows": count,
            "raw_pixel_bytes": raw,
            "deflated_pixel_bytes": deflated,
            "sidecar": f"shard-{shard_index:05d}.json",
            "row_sha256": hashlib.sha256(
                "\n".join(f"{pid}\t{tiles}" for pid, tiles in
                          ((patch_id, rows[patch_id]["mgrs_tile"]) for patch_id in patch_ids)
                          ).encode()).hexdigest(),
        })
        sidecar = {
            "shard_index": shard_index,
            "git_sha": git_sha,
            "split_name": f"caption-geo-split.v1/{side}",
            "seed": seed,
            "n": count,
            "bands": list(BANDS),
            "native_size": {band: NATIVE_SIZE[band] for band in BANDS},
            "dtype": "uint16",
            "resampled_in_shard": False,
            "contract_target_size": [SIZE, SIZE],
            "rows": [
                {
                    "patch_id": patch_id,
                    "mgrs_tile": rows[patch_id]["mgrs_tile"],
                    "country": rows[patch_id]["country"],
                    "official_split": rows[patch_id]["official_split"],
                    "caption": captions[patch_id],
                }
                for patch_id in patch_ids
            ],
        }
        (output_dir / f"shard-{shard_index:05d}.json").write_text(
            json.dumps(sidecar, indent=2) + "\n")
        del band_arrays
    buffers.clear()

    manifest = {
        "kind": "stage-1 training shards",
        "git_sha": git_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "split_name": f"caption-geo-split.v1/{side}",
        "seed": seed,
        "selection": SELECTION_ALGORITHM,
        "bands": list(BANDS),
        "dtype": "uint16",
        "native_size": {band: NATIVE_SIZE[band] for band in BANDS},
        "resampled_in_shard": False,
        "normalized_in_shard": False,
        "contract_target_size": [SIZE, SIZE],
        "norm_contract_path": str(norm_contract_path) if norm_contract_path else None,
        "n_patches": sum(record["rows"] for record in shard_records),
        "shard_count": len(shard_records),
        "archive_parts": [{"name": Path(part).name, "bytes": Path(part).stat().st_size}
                          for part in archive_parts],
        "archive_bytes_streamed": stream.bytes_read,
        "archive_stream_stopped_early": not stream_complete,
        "archive_members_seen": members_seen,
        "build_seconds": elapsed,
        "tile_counts": {tile: sum(1 for patch_ids in plan for patch_id in patch_ids
                                  if rows[patch_id]["mgrs_tile"] == tile)
                        for tile in sorted({rows[patch_id]["mgrs_tile"] for group in plan for patch_id in group})},
        "raw_pixel_bytes_total": total_raw,
        "deflated_pixel_bytes_total": total_deflated,
        "deflate_ratio": total_deflated / total_raw if total_raw else None,
        "candidates_considered": len(accepted) + len(rejected),
        "selection_size": len(candidates),
        "shortfall": len(candidates) - len(accepted) - len(never_seen),
        "never_seen_in_archive": len(never_seen),
        "never_seen_examples": never_seen[:20],
        "rejected_count": len(rejected),
        "candidate_rejected_fraction": len(rejected) / (len(accepted) + len(rejected))
                                        if accepted or rejected else None,
        "rejected_by_reason": {
            reason.split(":")[0]: sum(1 for value in rejected.values()
                                      if value.split(":")[0] == reason.split(":")[0])
            for reason in sorted(rejected.values())
        },
        "rejected_patches": dict(sorted(rejected.items())),
        "raw_zero_fraction": 0.0,
        "band_dn1_fraction_mean": {
            band: (sum(value for index, value in enumerate(per_band_one)
                       if index % len(BANDS) == BANDS.index(band)) /
                  max(1, len(per_band_one) // len(BANDS)))
            for band in BANDS
        },
        "all_one_pixel_fraction_note": (
            "not computable across native GSDs without resampling; per-band DN-1 "
            "fractions are reported instead and are not gated"),
        "shards": shard_records,
    }
    manifest_path = output_dir / "shards-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(manifest_path: Path) -> dict:
    """Re-hash every shard file and re-check sidecar/manifest agreement."""
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest_path).parent
    problems: list[str] = []
    total_rows = 0
    seen_ids: set[str] = set()
    tiles: set[str] = set()
    for record in manifest["shards"]:
        npz_path = root / record["npz"]
        if not npz_path.is_file():
            problems.append(f"missing shard file: {record['npz']}")
            continue
        if npz_path.stat().st_size != record["npz_bytes"]:
            problems.append(f"size mismatch: {record['npz']}")
        if sha256_file(npz_path) != record["npz_sha256"]:
            problems.append(f"sha256 mismatch: {record['npz']}")
        sidecar = json.loads((root / record["sidecar"]).read_text())
        if sidecar["n"] != record["rows"] or len(sidecar["rows"]) != record["rows"]:
            problems.append(f"sidecar row count mismatch: {record['sidecar']}")
        with np.load(npz_path) as payload:
            stored = {band: payload[band] for band in BANDS}
        for band, array in stored.items():
            expected = (record["rows"], NATIVE_SIZE[band], NATIVE_SIZE[band])
            if array.shape != expected or array.dtype != np.uint16:
                problems.append(f"native band array mismatch for {band} in {record['npz']}")
        ids = [row["patch_id"] for row in sidecar["rows"]]
        if len(set(ids)) != len(ids):
            problems.append(f"duplicate patch inside shard: {record['npz']}")
        for patch_id in ids:
            if patch_id in seen_ids:
                problems.append(f"duplicate patch across shards: {patch_id}")
            seen_ids.add(patch_id)
        tiles.update(row["mgrs_tile"] for row in sidecar["rows"])
        total_rows += record["rows"]
    if total_rows != manifest["n_patches"]:
        problems.append("manifest n_patches disagrees with the shard row counts")
    return {"manifest": str(manifest_path), "ok": not problems, "problems": problems,
            "n_patches": total_rows, "tiles": sorted(tiles)}


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class ShardDataset:
    """Deterministic, resumable reader over a shard manifest.

    The sample order is a per-epoch permutation derived only from (seed, epoch),
    so a checkpoint position of (epoch, cursor) reproduces the exact same
    sequence on resume without storing the permutation.
    """

    def __init__(self, manifest_path: Path, *, seed: int):
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["dtype"] != "uint16" or tuple(manifest["bands"]) != BANDS:
            raise ValueError("Shard manifest does not describe 12-band uint16 native pixels")
        if manifest.get("resampled_in_shard") is not False:
            raise ValueError("Shard manifest must record that no resampling happened in the shard")
        root = Path(manifest_path).parent
        self.root = root
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = sha256_file(Path(manifest_path))
        self.seed = seed
        self.split_name = manifest["split_name"]
        self.entries: list[tuple[int, int, str]] = []
        self.tiles: set[str] = set()
        self._sidecars: dict[int, dict] = {}
        for record in manifest["shards"]:
            sidecar = json.loads((root / record["sidecar"]).read_text())
            self._sidecars[record["index"]] = sidecar
            for row_index, row in enumerate(sidecar["rows"]):
                self.entries.append((record["index"], row_index, row["patch_id"]))
                self.tiles.add(row["mgrs_tile"])
        if len(self.entries) != manifest["n_patches"]:
            raise ValueError("Shard manifest row count disagrees with its sidecars")
        self._cache_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self.epoch = 0
        self.cursor = 0
        self._order_epoch: int | None = None
        self._order: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(epoch)])
        return rng.permutation(len(self.entries))

    def state(self) -> dict:
        rng = np.random.default_rng([self.seed, int(self.epoch)])
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor,
                "shard_manifest_sha256": self.manifest_sha256,
                "epoch_rng_state": json.loads(json.dumps(rng.bit_generator.state))}

    def load_state(self, state: dict) -> None:
        if state["seed"] != self.seed:
            raise ValueError("Checkpoint seed does not match the dataset seed")
        if state["shard_manifest_sha256"] != self.manifest_sha256:
            raise ValueError("Checkpoint shard manifest SHA256 does not match this manifest")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order_epoch = None

    def shard_index(self, entry: int) -> int:
        return self.entries[entry][0]

    def _pixels_for(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index != shard_index:
            name = f"shard-{shard_index:05d}.npz"
            with np.load(self.root / name) as payload:
                self._cache = {band: payload[band] for band in BANDS}
            self._cache_index = shard_index
        return self._cache

    def row(self, entry: int) -> dict:
        shard_index, row_index, patch_id = self.entries[entry]
        return self._sidecars[shard_index]["rows"][row_index]

    def raw_patch(self, entry: int) -> dict[str, np.ndarray]:
        """One patch at each band's native resolution, in the frozen band order."""
        shard_index, row_index, _patch_id = self.entries[entry]
        cached = self._pixels_for(shard_index)
        return {band: cached[band][row_index] for band in BANDS}

    def _cached_order(self) -> np.ndarray:
        if self._order_epoch != self.epoch or self._order is None:
            self._order = self.epoch_order(self.epoch)
            self._order_epoch = self.epoch
        return self._order

    def next_entry(self) -> tuple[int, int]:
        """Return (position_in_epoch, entry_index) and advance the cursor.

        The epoch rolls over as soon as its last position is consumed, so a
        state of (epoch, 0) after a full pass resumes at the start of the next
        epoch's permutation.
        """
        order = self._cached_order()
        if self.cursor >= len(order):
            self.epoch += 1
            self.cursor = 0
            order = self._cached_order()
        position = self.cursor
        entry = int(order[position])
        self.cursor = position + 1
        if self.cursor >= len(order):
            self.epoch += 1
            self.cursor = 0
        return position, entry

    def entries_from(self, start: int = 0):
        """Yield (position_in_epoch, entry_index, patch_id) for the epoch order.

        The caller advances the position with ``advance``; the epoch rolls over
        there, so the sequence after a resume at (epoch, cursor) is identical.
        """
        order = self.epoch_order(self.epoch)
        for position in range(start, len(order)):
            entry = int(order[position])
            yield position, entry, self.entries[entry][2]

    def advance(self, position: int) -> None:
        """Record the consumed cursor, rolling the epoch when it completes."""
        if position >= len(self.entries):
            self.epoch += 1
            self.cursor = 0
        else:
            self.cursor = position

# --------------------------------------------------------------------------- #
# prefix-stratified selection (loop 9)
# --------------------------------------------------------------------------- #
INTERLEAVED_SELECTION = (
    "numpy.random.default_rng([seed, tile_index]) per tile; patch IDs sorted within a "
    "tile; round-robin equal quotas dealt one at a time across the side's tiles in "
    "sorted order and capped by each tile's available count; the selection is then "
    "emitted round-robin across tiles, so every prefix is tile-balanced"
)


def select_patch_ids_interleaved(rows: dict[str, dict[str, str]], side: str,
                                 n_total: int, seed: int) -> list[str]:
    """Seeded, tile-stratified selection whose every prefix stays tile-balanced.

    ``select_patch_ids`` emits tiles in blocks, so a prefix of a long selection
    covers only the first few tiles; that is fine for a 2,000-patch shard set but
    not for a build that can be stopped at any shard boundary. Here the per-tile
    queues are dealt round-robin, which makes the max-minus-min per-tile count of
    any prefix at most one while quotas are unexhausted.
    """
    if side not in {"train", "eval"}:
        raise ValueError(f"Unknown geo side: {side}")
    if n_total < 1:
        raise ValueError("n_total must be positive")
    by_tile: dict[str, list[str]] = defaultdict(list)
    for patch_id, row in sorted(rows.items()):
        if row["geo_split"] == side:
            by_tile[row["mgrs_tile"]].append(patch_id)
    tiles = sorted(by_tile)
    if not tiles:
        raise ValueError(f"No patches on the {side} side")
    quotas = {tile: 0 for tile in tiles}
    dealt = 0
    while dealt < n_total:
        progressed = False
        for tile in tiles:
            if dealt >= n_total:
                break
            if quotas[tile] >= len(by_tile[tile]):
                continue
            quotas[tile] += 1
            dealt += 1
            progressed = True
        if not progressed:
            break
    queues: dict[str, list[str]] = {}
    for index, tile in enumerate(tiles):
        available = by_tile[tile]
        # Keep the permutation order: np.sort() here would always return
        # [0, 1, ..., quota-1] and silently turn the seed into a no-op that
        # selects the lowest patch IDs per tile.
        order = np.random.default_rng([int(seed), index]).permutation(len(available))
        queues[tile] = [available[int(position)] for position in order[:quotas[tile]]]
    selected: list[str] = []
    cursors = {tile: 0 for tile in tiles}
    remaining = sum(quotas.values())
    while remaining:
        for tile in tiles:
            if cursors[tile] >= quotas[tile]:
                continue
            selected.append(queues[tile][cursors[tile]])
            cursors[tile] += 1
            remaining -= 1
    if len(selected) != sum(quotas.values()):
        raise ValueError("Interleaved selection length disagrees with the allocated quotas")
    return selected


def prefix_tile_balance(rows: dict[str, dict[str, str]], selection: list[str],
                        prefixes: list[int]) -> dict[int, dict[str, float]]:
    """Per-tile counts over prefixes of a selection; no pixels are read.

    Train-tile availability spans 40 to 31,770 patches, so an equal count per
    tile is unattainable. The reported ``max_deviation_pp`` is therefore the
    largest difference, in percentage points, between a tile's share of a prefix
    and its share of the whole selection; that is the gate in
    results/stage1-loop9-preregistration.json amendments[0].
    """
    tiles = [rows[patch_id]["mgrs_tile"] for patch_id in selection]
    total_counts: dict[str, int] = defaultdict(int)
    for tile in tiles:
        total_counts[tile] += 1
    total = len(selection)
    shares = {tile: count / total for tile, count in total_counts.items()}
    report: dict[int, dict[str, float]] = {}
    for prefix in prefixes:
        counts: dict[str, int] = defaultdict(int)
        for tile in tiles[:prefix]:
            counts[tile] += 1
        values = list(counts.values())
        deviation = max(abs(counts.get(tile, 0) / prefix - share)
                        for tile, share in shares.items()) * 100
        report[prefix] = {
            "tiles_present": len(counts),
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
            "max_deviation_pp": deviation,
        }
    return report


# --------------------------------------------------------------------------- #
# resumable multi-shard build (loop 9)
# --------------------------------------------------------------------------- #
class BuildStop(RuntimeError):
    """Raised by a guard; whatever exists on disk is a valid prefix."""

    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def default_shard_window(patches_per_shard: int = 2000, *,
                         ram_fraction: float = 0.30) -> int:
    """How many shards may be buffered in RAM at once, from physical memory.

    A prefix-stratified selection cannot be served by a single archive pass with one
    shard buffer, because the archive is tile-major and shard ranks arrive out of
    order. Buffering a window of shards trades RAM for archive passes, so the window
    is sized to a fixed fraction of physical memory.
    """
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        total = 8 * 1024 ** 3
    return max(1, int(ram_fraction * total) // max(shard_bytes(patches_per_shard), 1))


def verify_shard_files(root: Path, records: list[dict]) -> tuple[list[int], list[str]]:
    """Return (verified shard indices, problems) for existing shard files."""
    verified: list[int] = []
    problems: list[str] = []
    for record in records:
        path = root / record["npz"]
        sidecar = root / record["sidecar"]
        if not path.is_file() or not sidecar.is_file():
            problems.append(f"missing files for shard {record['index']}")
            continue
        if path.stat().st_size != record["npz_bytes"]:
            problems.append(f"size mismatch for shard {record['index']}")
            continue
        if sha256_file(path) != record["npz_sha256"]:
            problems.append(f"sha256 mismatch for shard {record['index']}")
            continue
        verified.append(record["index"])
    return verified, problems


def build_shard_set(
    archive_parts: list[Path],
    candidates: list[str],
    captions: dict[str, str],
    rows: dict[str, dict[str, str]],
    output_dir: Path,
    *,
    git_sha: str,
    seed: int,
    side: str = "train",
    patches_per_shard: int = 2000,
    published_manifest: Path | None = None,
    excluded_ids: frozenset[str] = frozenset(),
    excluded_tile_counts: dict[str, int] | None = None,
    min_free_bytes: int | None = None,
    speed_guard_after_shards: int | None = None,
    speed_guard_max_seconds_per_patch: float | None = None,
    max_rejection_fraction: float | None = None,
    shard_window: int | None = None,
    log=print,
) -> dict:
    """One archive pass writing every shard in order, resumably and incrementally.

    Existing shards whose manifest entry still matches the file (size and
    SHA256) are verified and skipped, their patch IDs are excluded from the
    candidate set, and the pass continues with the first unverified shard. The
    manifest is rewritten after each shard so an interrupted build is a valid
    prefix.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        raise ValueError("Candidate selection is empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate selection repeats a patch")
    shard_count = -(-len(candidates) // patches_per_shard)

    previous: dict = {}
    if published_manifest is not None and Path(published_manifest).is_file():
        previous = json.loads(Path(published_manifest).read_text())
    existing_records = previous.get("shards", [])
    verified, problems = verify_shard_files(output_dir, existing_records)
    if problems:
        raise ValueError(f"Existing shards are unverifiable: {problems[:3]}")

    verified_set = set(verified)
    shard_ids: dict[int, list[str]] = {}
    seen_ids: set[str] = set(excluded_ids)
    for record in existing_records:
        if record["index"] not in verified_set:
            continue
        sidecar = json.loads((output_dir / record["sidecar"]).read_text())
        if sidecar.get("native_checksum") != "sha256-native-v1":
            raise ValueError("Legacy shards lack source pixel checksums; rebuild in a new directory")
        with np.load(output_dir / record["npz"]) as pixels:
            for i, row in enumerate(sidecar["rows"]):
                if native_patch_sha256({band: pixels[band][i] for band in BANDS}) != row.get("native_sha256"):
                    raise ValueError(f"Existing shard pixel checksum mismatch: {row['patch_id']}")
        ids = [row["patch_id"] for row in sidecar["rows"]]
        shard_ids[record["index"]] = ids
        overlap = seen_ids & set(ids)
        if overlap:
            raise ValueError(f"Existing shard {record['index']} repeats patch IDs: {sorted(overlap)[:3]}")
        seen_ids.update(ids)
    pending = {patch_id for patch_id in candidates
               if patch_id not in seen_ids and patch_id not in shard_ids}
    skipped_upfront = len(candidates) - len(pending)

    records: list[dict] = [record for record in existing_records if record["index"] in verified_set]
    # The selection is prefix-stratified, so a patch's rank decides its shard. The archive
    # is tile-major, so those ranks arrive out of order and no shard can be written until
    # the pass reaches its last patch. A window of shards is buffered at a time -- window
    # x 320 MB of RAM -- and each window costs one more pass over the 63.5 GB stream.
    # Shard membership therefore follows the selection order, not the archive order.
    target_shard = {patch_id: index // patches_per_shard
                    for index, patch_id in enumerate(candidates)}
    selection_rank = {patch_id: index for index, patch_id in enumerate(candidates)}
    window = max(1, min(int(shard_window or default_shard_window()), shard_count))
    rejected: dict[str, str] = {}
    passes: list[dict] = []
    streamed_total = 0
    members_seen = 0
    accepted_this_run = 0
    rejected_this_run = 0
    free_min = shutil.disk_usage(output_dir).free
    started = time.perf_counter()
    shard_started = started
    progress: list[dict] = []
    stopped = None

    def write_shard(index: int, patch_records: list[dict]) -> dict:
        patch_records = sorted(patch_records, key=lambda item: selection_rank[item["row"]["patch_id"]])
        ids = [item["row"]["patch_id"] for item in patch_records]
        free_now = shutil.disk_usage(output_dir).free
        if min_free_bytes is not None and free_now < min_free_bytes:
            raise BuildStop("disk_guard", {"shard": index, "free_bytes": free_now,
                                           "required_bytes": min_free_bytes})
        arrays = {band: np.stack([item["bands"][band] for item in patch_records]) for band in BANDS}
        npz_path = output_dir / f"shard-{index:05d}.npz"
        with npz_path.open("wb") as handle:
            np.savez(handle, **arrays)
        sidecar_rows = [item["row"] for item in patch_records]
        (output_dir / f"shard-{index:05d}.json").write_text(
            json.dumps({"shard_index": index, "git_sha": git_sha,
                        "split_name": f"caption-geo-split.v1/{side}", "seed": seed,
                        "n": len(ids), "bands": list(BANDS), "dtype": "uint16",
                        "native_size": {band: NATIVE_SIZE[band] for band in BANDS},
                        "resampled_in_shard": False, "native_checksum": "sha256-native-v1", "rows": sidecar_rows}, indent=2) + "\n")
        for band in BANDS:
            expected = (len(ids), NATIVE_SIZE[band], NATIVE_SIZE[band])
            if arrays[band].shape != expected or arrays[band].dtype != np.uint16:
                raise ValueError(f"Shard {index} band {band} has shape/dtype {arrays[band].shape}")
        tile_counts: dict[str, int] = defaultdict(int)
        for patch_id in ids:
            tile_counts[rows[patch_id]["mgrs_tile"]] += 1
        return {"index": index, "npz": npz_path.name, "npz_bytes": npz_path.stat().st_size,
                "npz_sha256": sha256_file(npz_path), "sidecar": f"shard-{index:05d}.json",
                "rows": len(ids), "raw_pixel_bytes": shard_bytes(len(ids)),
                "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "seconds": time.perf_counter() - shard_started,
                "tile_counts": dict(sorted(tile_counts.items())),
                "tile_min": min(tile_counts.values()), "tile_max": max(tile_counts.values())}

    def manifest_payload(status: str, stop_reason: str | None = None) -> dict:
        elapsed = time.perf_counter() - started
        considered = accepted_this_run + rejected_this_run
        return {
            "kind": "stage-1 training shard set",
            "git_sha": git_sha, "seed": seed, "side": side,
            "split_name": f"caption-geo-split.v1/{side}",
            "status": status, "stopped_reason": stop_reason,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "selection": (SELECTION_ALGORITHM if not previous else previous.get("selection")),
            "selection_algorithm": INTERLEAVED_SELECTION,
            "bands": list(BANDS), "dtype": "uint16",
            "native_size": {band: NATIVE_SIZE[band] for band in BANDS},
            "resampled_in_shard": False, "normalized_in_shard": False,
            "patches_per_shard": patches_per_shard,
            "selection_size": len(candidates),
            "shard_count": shard_count,
            "shards_written": len(records),
            "n_patches": sum(record["rows"] for record in records),
            "total_bytes": sum(record["npz_bytes"] for record in records),
            "verified_on_resume": sorted(verified_set),
            "skipped_upfront_patches": skipped_upfront,
            "accepted_this_run": accepted_this_run,
            "rejected_this_run": rejected_this_run,
            "rejected_count_total": len(rejected),
            "rejection_fraction_this_run": (rejected_this_run / considered) if considered else 0.0,
            "rejected_by_reason": {
                reason.split(":")[0]: sum(1 for value in rejected.values()
                                          if value.split(":")[0] == reason.split(":")[0])
                for reason in sorted(rejected.values())},
            "rejected_examples": dict(list(sorted(rejected.items()))[:20]),
            "archive_parts": [{"name": Path(part).name, "bytes": Path(part).stat().st_size}
                              for part in archive_parts],
            "shard_membership": "selection rank // patches_per_shard; shards are filled and "
                                "written in selection order, not archive order",
            "shard_window": window,
            "archive_passes": passes,
            "archive_bytes_streamed": streamed_total,
            "seconds_elapsed": elapsed,
            "seconds_per_patch_this_run": (elapsed / accepted_this_run) if accepted_this_run else None,
            "free_bytes_min": free_min,
            "progress": progress,
            "shards": sorted(records, key=lambda record: record["index"]),
        }

    def publish(status: str, stop_reason: str | None = None) -> dict:
        payload = manifest_payload(status, stop_reason)
        (output_dir / "shards-manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        if published_manifest is not None:
            Path(published_manifest).write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    def run_window(first: int, last: int) -> None:
        """One archive pass that writes shards first..last-1 in selection order."""
        nonlocal streamed_total, members_seen, shard_started
        nonlocal accepted_this_run, rejected_this_run, free_min
        shard_rows: list[list[dict]] = [[] for _ in range(last - first)]
        window_pending = {patch_id for patch_id in pending
                          if first <= target_shard[patch_id] < last}
        inflight: dict[str, dict] = {}
        pass_started = time.perf_counter()
        stream = _ConcatStream(archive_parts)
        try:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    members_seen += 1
                    if not member.isfile():
                        continue
                    key = _member_key(member.name)
                    if key is None or key[0] not in window_pending:
                        continue
                    patch_id, band = key
                    if band not in BANDS:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Archive member is unreadable: {member.name}")
                    entry = inflight.setdefault(patch_id, {"bands": {}, "shapes": []})
                    array = decode_member(source.read())
                    if array.dtype != np.uint16:
                        rejected[patch_id] = f"dtype:{array.dtype}"
                        rejected_this_run += 1
                        inflight.pop(patch_id)
                        pending.discard(patch_id)
                        window_pending.discard(patch_id)
                        continue
                    if array.shape != (NATIVE_SIZE[band], NATIVE_SIZE[band]):
                        entry["shapes"].append(f"{band}:{array.shape[0]}x{array.shape[1]}")
                    entry["bands"][band] = array
                    if len(entry["bands"]) < len(BANDS) and not entry["shapes"]:
                        continue
                    pending.discard(patch_id)
                    window_pending.discard(patch_id)
                    inflight.pop(patch_id)
                    missing = [name for name in BANDS if name not in entry["bands"]]
                    if entry["shapes"]:
                        rejected[patch_id] = "shape:" + ",".join(sorted(set(entry["shapes"])))
                    elif missing:
                        rejected[patch_id] = "missing_bands:" + ",".join(missing)
                    elif any(bool((entry["bands"][name] == 0).any()) for name in BANDS):
                        rejected[patch_id] = "raw_zero_dn"
                    else:
                        if patch_id in seen_ids:
                            raise ValueError(f"Duplicate patch ID reached the shard writer: {patch_id}")
                        local = target_shard[patch_id] - first
                        metadata = rows[patch_id]
                        caption = captions.get(patch_id)
                        if not caption or metadata["geo_split"] != side or metadata["mgrs_tile"] in (excluded_tile_counts or {}):
                            raise ValueError(f"Invalid accepted caption/split: {patch_id}")
                        shard_rows[local].append({
                            "row": {"patch_id": patch_id, "caption": caption,
                                    "mgrs_tile": metadata["mgrs_tile"], "country": metadata["country"],
                                    "official_split": metadata["official_split"],
                                    "native_sha256": native_patch_sha256(entry["bands"])},
                            "bands": entry["bands"],
                        })
                        seen_ids.add(patch_id)
                        accepted_this_run += 1
                        if len(shard_rows[local]) == patches_per_shard:
                            index = first + local
                            record = write_shard(index, shard_rows[local])
                            records.append(record)
                            progress.append({"shard": index, "rows": record["rows"],
                                             "bytes": record["npz_bytes"],
                                             "seconds": round(record["seconds"], 2)})
                            free_now = shutil.disk_usage(output_dir).free
                            free_min = min(free_min, free_now)
                            log(f"shard {index:05d} rows={record['rows']} bytes={record['npz_bytes']} "
                                f"s={record['seconds']:.1f} tiles={record['tile_min']}-{record['tile_max']} "
                                f"free={free_now / 1e9:.1f}GB total={sum(r['npz_bytes'] for r in records) / 1e9:.2f}GB")
                            publish("in_progress")
                            shard_started = time.perf_counter()
                            shard_rows[local] = []
                    considered = accepted_this_run + rejected_this_run
                    if (max_rejection_fraction is not None and considered
                            and rejected_this_run / considered > max_rejection_fraction):
                        raise BuildStop("rejection_fraction_guard",
                                        {"fraction": rejected_this_run / considered,
                                         "limit": max_rejection_fraction,
                                         "rejected_by_reason": manifest_payload("stopped")["rejected_by_reason"]})
                    if (speed_guard_max_seconds_per_patch is not None
                            and speed_guard_after_shards is not None
                            and len(progress) >= speed_guard_after_shards):
                        rate = (time.perf_counter() - started) / accepted_this_run
                        if rate > speed_guard_max_seconds_per_patch:
                            raise BuildStop("speed_guard",
                                            {"seconds_per_patch": rate,
                                             "limit": speed_guard_max_seconds_per_patch,
                                             "shards_completed": len(progress)})
        finally:
            streamed_total += stream.bytes_read
            stream.close()
        for patch_id, entry in sorted(inflight.items()):
            rejected[patch_id] = "incomplete_in_archive:" + ",".join(
                sorted(name for name in BANDS if name not in entry["bands"]) or ["no_bands"])
            rejected_this_run += 1
        partial = 0
        for local, patch_records in enumerate(shard_rows):
            if patch_records:
                index = first + local
                record = write_shard(index, patch_records)
                records.append(record)
                progress.append({"shard": index, "rows": record["rows"],
                                 "bytes": record["npz_bytes"],
                                 "seconds": round(record["seconds"], 2)})
                log(f"shard {index:05d} (partial) rows={record['rows']}")
                partial += 1
        passes.append({"first_shard": first, "last_shard": last,
                       "archive_bytes": stream.bytes_read,
                       "partial_shards": partial,
                       "seconds": round(time.perf_counter() - pass_started, 2)})
        log(f"window {first:05d}-{last - 1:05d} pass done in "
            f"{time.perf_counter() - pass_started:.1f}s (pass {len(passes)})")

    for first in range(0, shard_count, window):
        last = min(first + window, shard_count)
        done = {record["index"] for record in records}
        if all(index in done for index in range(first, last)):
            log(f"shards {first:05d}-{last - 1:05d} already verified; skipping pass")
            continue
        log(f"pass {len(passes) + 1}/{ -(-shard_count // window)}: shards {first:05d}-{last - 1:05d}, "
            f"{window} shards ({window * shard_bytes(patches_per_shard) / 1e9:.1f} GB) in RAM")
        try:
            run_window(first, last)
        except BuildStop as stop:
            stopped = {"reason": stop.reason, "detail": stop.detail,
                       "shards_written": len(records),
                       "last_shard": records[-1]["index"] if records else None}
            log(f"STOP {stop.reason}: {json.dumps(stop.detail)}")
            return publish("stopped", stop.reason)

    payload = publish("complete")
    payload["seconds_per_patch_overall"] = (
        (time.perf_counter() - started) / accepted_this_run) if accepted_this_run else None
    log(f"complete: {payload['shards_written']} shards, {payload['n_patches']} patches, "
        f"{payload['total_bytes']} bytes")
    return payload
