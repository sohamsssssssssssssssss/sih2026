"""CPU tests for the Stage-1 shard selection, builder, and resumable reader.

The archive is a synthetic tar.gz written here (twelve 120x120 uint16 GeoTIFFs
per patch), so the streaming path, band handling, and manifest hashing are
exercised end to end without touching the 63.5 GB real archive.
"""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.io import MemoryFile

from data.bigearthnet_s2 import (
    BANDS, NATIVE_SIZE, SIZE, apply_contract, load_norm_contract, resample_bands,
)
from data.bigearthnet_shards import (
    ShardDataset,
    assert_split_purity,
    build_shard_set,
    build_shards,
    default_shard_window,
    deflated_bytes,
    load_captions,
    load_split_rows,
    native_patch_bytes,
    prefix_tile_balance,
    select_patch_ids,
    select_patch_ids_interleaved,
    shard_bytes,
    verify_manifest,
    verify_shard_files,
)

TRAIN_TILES = ("T29SNB", "T30TVM")
EVAL_TILE = "T33UWP"


def _tif_bytes(seed: int, band: str) -> bytes:
    """A deterministic single-band uint16 GeoTIFF at the band's native GSD."""
    rng = np.random.default_rng(seed)
    size = NATIVE_SIZE[band]
    array = rng.integers(1, 9000, size=(size, size), dtype=np.uint16)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=size, width=size, count=1,
                         dtype="uint16", nodata=0) as dataset:
            dataset.write(array, 1)
            dataset.set_band_description(1, band)
        return memory.read()


def _patch_id(tile: str, index: int) -> str:
    return f"S2A_MSIL2A_20180413T095031_N9999_R079_{tile}_06_0{index}"


def _split_rows(patch_ids):
    return {
        patch_id: {"mgrs_tile": patch_id.split("_")[-3], "geo_split": geo,
                   "country": "Finland", "official_split": "train"}
        for patch_id, geo in patch_ids
    }


def _make_archive(tmp_path: Path, patch_ids, *, zero_band_for=()):
    """Write BigEarthNet-S2.tar.gzaa/ab holding one TIFF per band per patch."""
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for index, patch_id in enumerate(patch_ids):
            tile = patch_id.split("_")[-3]
            archive.addfile(tarfile.TarInfo(f"BigEarthNet-S2/{tile}/"), None)
            for band in BANDS:
                name = f"BigEarthNet-S2/{tile}/{patch_id}/{patch_id}_{band}.tif"
                info = tarfile.TarInfo(name)
                data = _tif_bytes(index * 13 + BANDS.index(band), band)
                if (patch_id, band) in zero_band_for:
                    array = np.zeros((SIZE, SIZE), dtype=np.uint16)
                    with MemoryFile() as memory:
                        with memory.open(driver="GTiff", height=SIZE, width=SIZE, count=1,
                                         dtype="uint16", nodata=0) as dataset:
                            dataset.write(array, 1)
                        data = memory.read()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    raw = payload.getvalue()
    half = len(raw) // 2
    first = tmp_path / "BigEarthNet-S2.tar.gzaa"
    second = tmp_path / "BigEarthNet-S2.tar.gzab"
    first.write_bytes(raw[:half])
    second.write_bytes(raw[half:])
    return [first, second]


def _fixture(tmp_path, *, per_tile=4, eval_patches=2):
    train_ids = [_patch_id(tile, index)
                 for tile in TRAIN_TILES for index in range(per_tile)]
    eval_ids = [_patch_id(EVAL_TILE, index) for index in range(eval_patches)]
    rows = _split_rows([(pid, "train") for pid in train_ids] + [(pid, "eval") for pid in eval_ids])
    parts = _make_archive(tmp_path, train_ids + eval_ids)
    captions = {pid: f"caption for {pid}" for pid in train_ids + eval_ids}
    return train_ids, eval_ids, rows, parts, captions


# --------------------------------------------------------------------------- #
def test_selection_is_deterministic_and_stratified():
    rows = _fixture_selection_rows()
    first = select_patch_ids(rows, "train", 8, seed=5)
    second = select_patch_ids(rows, "train", 8, seed=5)
    assert first == second
    assert len(set(first)) == 8
    tiles = {rows[patch_id]["mgrs_tile"] for patch_id in first}
    assert tiles == {"T29SNB", "T30TVM"}
    # T29SNB holds only 3 patches, so its quota caps there and the selection is
    # 3 + 5: surplus is not redistributed.
    assert [rows[p]["mgrs_tile"] for p in first] == ["T29SNB"] * 3 + ["T30TVM"] * 5
    assert select_patch_ids(rows, "train", 8, seed=6) != first


def _fixture_selection_rows():
    rows = {}
    for tile, available in (("T29SNB", 3), ("T30TVM", 10)):
        for index in range(available):
            patch_id = _patch_id(tile, index)
            rows[patch_id] = {"mgrs_tile": tile, "geo_split": "train",
                              "country": "Finland", "official_split": "train"}
    return rows


def test_selection_quota_cap_does_not_redistribute():
    rows = _fixture_selection_rows()
    selected = select_patch_ids(rows, "train", 13, seed=1)
    counts = {}
    for patch_id in selected:
        counts[rows[patch_id]["mgrs_tile"]] = counts.get(rows[patch_id]["mgrs_tile"], 0) + 1
    assert counts == {"T29SNB": 3, "T30TVM": 10}


def test_split_purity_rejects_eval_and_duplicates(tmp_path):
    train_ids, eval_ids, rows, _parts, _captions = _fixture(tmp_path, per_tile=2, eval_patches=1)
    assert_split_purity(train_ids, rows, "train")
    with pytest.raises(ValueError, match="eval side"):
        assert_split_purity([eval_ids[0]], rows, "train")
    with pytest.raises(ValueError, match="duplicate"):
        assert_split_purity([train_ids[0], train_ids[0]], rows, "train")
    with pytest.raises(ValueError, match="absent from the split manifest"):
        assert_split_purity(["S2A_x_T31AAA_01_01"], rows, "train")


def test_shard_plan_is_candidate_order_with_rejections(tmp_path, monkeypatch):
    import data.bigearthnet_shards as shards_module

    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    # The first selected patch is off-size in the archive: it is rejected with a
    # reason and is absent from the shards, and the shortfall is recorded.
    original = shards_module.decode_member
    state = {"seen": 0}

    def decode_off_size_first(payload):
        state["seen"] += 1
        return np.zeros((20, 20), dtype=np.uint16) if state["seen"] == 1 else original(payload)

    monkeypatch.setattr(shards_module, "decode_member", decode_off_size_first)
    manifest = build_shards(parts, train_ids, captions, rows,
                            tmp_path / "shards", git_sha="test-sha", seed=3,
                            patches_per_shard=2)
    assert manifest["n_patches"] == len(train_ids) - 1
    assert manifest["selection_size"] == len(train_ids)
    assert manifest["shortfall"] == 1
    assert manifest["rejected_count"] == 1
    assert list(manifest["rejected_patches"]) == [train_ids[0]]
    accepted_order = [row["patch_id"] for shard in manifest["shards"]
                      for row in json.loads(
                          (tmp_path / "shards" / shard["sidecar"]).read_text())["rows"]]
    assert accepted_order == train_ids[1:]


def test_build_rejects_off_size_and_raw_zero_candidates(tmp_path, monkeypatch):
    import data.bigearthnet_shards as shards_module

    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    original = shards_module.decode_member
    calls = {"n": 0}

    def decode(payload):
        calls["n"] += 1
        # The second member is a 10 m band; handing it a 20 m array is off-size.
        return np.zeros((20, 20), dtype=np.uint16) if calls["n"] == 2 else original(payload)

    monkeypatch.setattr(shards_module, "decode_member", decode)
    manifest = build_shards(parts, train_ids, captions, rows,
                            tmp_path / "shards", git_sha="test-sha", seed=3,
                            patches_per_shard=2)
    assert "shape:" in manifest["rejected_patches"][train_ids[0]]
    assert "shape" in manifest["rejected_by_reason"]


# --------------------------------------------------------------------------- #
def test_build_shards_streams_and_records_manifest(tmp_path):
    train_ids, eval_ids, rows, parts, captions = _fixture(tmp_path)
    out = tmp_path / "shards"
    selected = select_patch_ids(rows, "train", len(train_ids), seed=3)
    assert_split_purity(selected, rows, "train")
    manifest = build_shards(parts, selected, captions, rows, out,
                            git_sha="test-sha", seed=3, patches_per_shard=3)
    assert manifest["n_patches"] == len(train_ids)
    assert manifest["resampled_in_shard"] is False
    assert manifest["native_size"]["B02"] == 120 and manifest["native_size"]["B05"] == 60 \
        and manifest["native_size"]["B01"] == 20
    assert manifest["shard_count"] == math_ceil(len(train_ids), 3)
    assert manifest["normalized_in_shard"] is False
    assert manifest["raw_zero_fraction"] == 0.0
    assert manifest["rejected_count"] == 0
    assert manifest["candidate_rejected_fraction"] == 0.0
    assert manifest["archive_bytes_streamed"] > 0
    # The pass stops as soon as the requested rows are accepted, so it reads
    # fewer members than the archive holds.
    assert manifest["archive_stream_stopped_early"] is True
    assert len(train_ids) * len(BANDS) <= manifest["archive_members_seen"] <= \
        (len(train_ids) + len(eval_ids)) * len(BANDS)
    assert manifest["shortfall"] == 0
    assert manifest["tile_counts"] == {"T29SNB": 4, "T30TVM": 4}
    check = verify_manifest(out / "shards-manifest.json")
    assert check["ok"], check["problems"]
    assert check["n_patches"] == len(train_ids)
    assert not (set(selected) & set(eval_ids))


def math_ceil(total: int, size: int) -> int:
    return -(-total // size)


def test_build_shards_is_byte_deterministic(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path)
    out = tmp_path / "shards"
    candidates = select_patch_ids(rows, "train", len(train_ids), seed=3)
    first = build_shards(parts, candidates, captions, rows, out,
                         git_sha="test-sha", seed=3, patches_per_shard=3)
    second = build_shards(parts, candidates, captions, rows, out / "again",
                          git_sha="test-sha", seed=3, patches_per_shard=3)
    assert [record["npz_sha256"] for record in first["shards"]] == \
           [record["npz_sha256"] for record in second["shards"]]


def test_build_shards_rejects_raw_zero_candidates(tmp_path, monkeypatch):
    """A patch carrying raw DN 0 is rejected with a recorded reason, never dropped."""
    import data.bigearthnet_shards as shards_module

    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    original = shards_module.decode_member
    state = {"seen": 0}
    zero_native = NATIVE_SIZE["B01"]

    def decode_with_one_zero(payload):
        state["seen"] += 1
        return np.zeros((zero_native, zero_native), dtype=np.uint16) if state["seen"] == 1 \
            else original(payload)

    monkeypatch.setattr(shards_module, "decode_member", decode_with_one_zero)
    manifest = shards_module.build_shards(parts, train_ids, captions, rows,
                                          tmp_path / "shards", git_sha="test-sha", seed=3,
                                          patches_per_shard=2)
    assert manifest["raw_zero_fraction"] == 0.0
    assert manifest["rejected_by_reason"]["raw_zero_dn"] == 1
    assert train_ids[0] in manifest["rejected_patches"]
    assert manifest["n_patches"] == len(train_ids) - 1


def test_build_shards_fails_when_nothing_can_be_built(tmp_path, monkeypatch):
    import data.bigearthnet_shards as shards_module

    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    monkeypatch.setattr(shards_module, "decode_member",
                        lambda payload: np.zeros((7, 7), dtype=np.uint16))
    with pytest.raises(ValueError, match="No selected patch could be built"):
        build_shards(parts, train_ids, captions, rows, tmp_path / "shards",
                     git_sha="test-sha", seed=3, patches_per_shard=2)


def test_build_shards_records_never_seen_candidates(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    absent = _patch_id("T30TVM", 99)
    rows[absent] = {"mgrs_tile": "T30TVM", "geo_split": "train",
                    "country": "Finland", "official_split": "train"}
    captions[absent] = "not in the archive"
    manifest = build_shards(parts, train_ids + [absent], captions, rows,
                            tmp_path / "shards", git_sha="test-sha", seed=3, patches_per_shard=2)
    assert manifest["never_seen_in_archive"] == 1
    assert manifest["never_seen_examples"] == [absent]
    assert manifest["shortfall"] == 0


def test_verify_manifest_detects_tampering(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    out = tmp_path / "shards"
    build_shards(parts, train_ids, captions, rows, out,
                 git_sha="test-sha", seed=3, patches_per_shard=2)
    target = out / "shard-00000.npz"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    check = verify_manifest(out / "shards-manifest.json")
    assert not check["ok"]
    assert any("sha256 mismatch" in problem for problem in check["problems"])


def test_native_sizes_match_the_sentinel2_band_specification():
    """10 m bands are 120, 20 m bands are 60, 60 m bands are 20 (archive-measured)."""
    assert {band for band, size in NATIVE_SIZE.items() if size == 120} == \
        {"B02", "B03", "B04", "B08"}
    assert {band for band, size in NATIVE_SIZE.items() if size == 60} == \
        {"B05", "B06", "B07", "B8A", "B11", "B12"}
    assert {band for band, size in NATIVE_SIZE.items() if size == 20} == \
        {"B01", "B09"}


def test_shard_bytes_and_deflate_are_measured_shapes():
    assert native_patch_bytes() == 4 * 120 * 120 * 2 + 6 * 60 * 60 * 2 + 2 * 20 * 20 * 2
    raw = shard_bytes(2)
    assert raw == 2 * native_patch_bytes()
    pixels = np.full((2, len(BANDS), SIZE * SIZE), 7, dtype=np.uint16)
    compressed = deflated_bytes(pixels)
    assert 0 < compressed < raw


def test_dataset_epoch_order_and_resume_are_exact(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path)
    out = tmp_path / "shards"
    build_shards(parts, train_ids, captions, rows, out,
                 git_sha="test-sha", seed=3, patches_per_shard=2)
    dataset = ShardDataset(out / "shards-manifest.json", seed=11)
    assert len(dataset) == len(train_ids)

    def consume(limit):
        seen = []
        for position, entry, patch_id in dataset.entries_from(dataset.cursor):
            seen.append(patch_id)
            dataset.advance(position + 1)
            if len(seen) >= limit:
                break
        return seen

    full = []
    while len(full) < len(train_ids):
        full.extend(consume(len(train_ids)))
    assert len(set(full)) == len(train_ids)

    dataset2 = ShardDataset(out / "shards-manifest.json", seed=11)
    first_half = consume_from(dataset2, 3)
    state = json.loads(json.dumps(dataset2.state()))
    second_half = consume_from(dataset2, 5)
    uninterrupted = full[:8]

    dataset3 = ShardDataset(out / "shards-manifest.json", seed=11)
    resumed_first = consume_from(dataset3, 3)
    dataset3.load_state(state)
    resumed_second = consume_from(dataset3, 5)
    assert resumed_first == uninterrupted[:3]
    assert resumed_second == uninterrupted[3:8]
    assert first_half + second_half == uninterrupted


def consume_from(dataset, limit):
    seen = []
    for position, entry, patch_id in dataset.entries_from(dataset.cursor):
        seen.append(patch_id)
        dataset.advance(position + 1)
        if len(seen) >= limit:
            break
    return seen


def test_dataset_epoch_rollover_is_a_legal_position(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path)
    out = tmp_path / "shards"
    build_shards(parts, train_ids, captions, rows, out,
                 git_sha="test-sha", seed=3, patches_per_shard=4)
    dataset = ShardDataset(out / "shards-manifest.json", seed=2)
    consume_from(dataset, len(train_ids))
    assert dataset.epoch == 1 and dataset.cursor == 0
    state = dataset.state()
    order_next = [patch_id for _position, _entry, patch_id in dataset.entries_from(0)]
    dataset2 = ShardDataset(out / "shards-manifest.json", seed=2)
    dataset2.load_state(state)
    assert [patch_id for _position, _entry, patch_id in dataset2.entries_from(0)] == order_next


def test_dataset_rejects_foreign_manifest_state(tmp_path):
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    out = tmp_path / "shards"
    build_shards(parts, train_ids, captions, rows, out,
                 git_sha="test-sha", seed=3, patches_per_shard=2)
    dataset = ShardDataset(out / "shards-manifest.json", seed=2)
    with pytest.raises(ValueError, match="seed"):
        dataset.load_state({"seed": 999, "epoch": 0, "cursor": 0,
                            "shard_manifest_sha256": dataset.manifest_sha256})
    with pytest.raises(ValueError, match="manifest SHA256"):
        dataset.load_state({"seed": 2, "epoch": 0, "cursor": 0,
                            "shard_manifest_sha256": "0" * 64})


def test_shard_pixels_match_the_extracted_directory_loader(tmp_path, monkeypatch):
    """A shard row and the extracted TIFF directory give identical pixels."""
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    out = tmp_path / "shards"
    build_shards(parts, train_ids, captions, rows, out,
                 git_sha="test-sha", seed=3, patches_per_shard=4)
    patch_id = train_ids[0]
    patch_dir = tmp_path / "extracted" / patch_id.split("_")[-3] / patch_id
    patch_dir.mkdir(parents=True)
    for index, band in enumerate(BANDS):
        (patch_dir / f"{patch_id}_{band}.tif").write_bytes(_tif_bytes(index, band))
    from data.bigearthnet_s2 import load_s2_patch

    contract = load_norm_contract(Path("data/manifests/bigearthnet/norm_contract.json"))
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))

    from_directory = load_s2_patch(patch_dir, norm_contract=contract_path)
    dataset = ShardDataset(out / "shards-manifest.json", seed=1)
    entry = dataset.entries.index(next(
        item for item in dataset.entries if item[2] == patch_id))
    native = dataset.raw_patch(entry)
    assert {band: array.shape for band, array in native.items()} == \
        {band: (NATIVE_SIZE[band], NATIVE_SIZE[band]) for band in BANDS}
    from_shard = apply_contract(
        torch.from_numpy(np.stack(resample_bands(native))), contract)
    assert torch_allclose(from_directory, from_shard)


def torch_allclose(left, right) -> bool:
    return bool(torch.allclose(left, right, atol=1e-6))


def test_captions_come_from_the_captioning_rows(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    parquet = tmp_path / "BEN.txt.parquet"
    pq.write_table(pa.table({
        "patch_id": ["a", "a", "b"],
        "type": ["captioning", "classification", "captioning"],
        "output": ["first", "ignored", "second"],
    }), parquet)
    assert load_captions(parquet, ["a", "b"]) == {"a": "first", "b": "second"}
    with pytest.raises(ValueError, match="No captioning row"):
        load_captions(parquet, ["c"])

# --------------------------------------------------------------------------- #
# prefix-stratified selection and the resumable shard set (loop 9)
# --------------------------------------------------------------------------- #
def test_loop8_selection_is_not_prefix_stratified_and_stays_untouched():
    """The loop-8 ordering is tile-blocked; its function and artifacts are unchanged."""
    rows = _fixture_selection_rows()
    blocked = select_patch_ids(rows, "train", 8, seed=5)
    assert [rows[p]["mgrs_tile"] for p in blocked] == ["T29SNB"] * 3 + ["T30TVM"] * 5
    report = prefix_tile_balance(rows, blocked, [8])
    assert report[8]["tiles_present"] == 2


def _two_tiles(available=20):
    rows = {}
    for tile in ("T29SNB", "T30TVM"):
        for index in range(available):
            patch_id = _patch_id(tile, index)
            rows[patch_id] = {"mgrs_tile": tile, "geo_split": "train",
                              "country": "Finland", "official_split": "train"}
    return rows


def test_interleaved_selection_is_prefix_stratified_proportionally():
    rows = _two_tiles(20)
    selection = select_patch_ids_interleaved(rows, "train", 20, seed=5)
    assert len(selection) == 20 and len(set(selection)) == 20
    assert selection == select_patch_ids_interleaved(rows, "train", 20, seed=5)
    # With more candidates than the quota, the seed changes which patches are taken.
    assert selection != select_patch_ids_interleaved(rows, "train", 20, seed=6)
    report = prefix_tile_balance(rows, selection, [10, 20])
    for values in report.values():
        assert values["tiles_present"] == 2
        assert values["min"] == values["max"]
        assert values["max_deviation_pp"] == pytest.approx(0.0, abs=1e-9)
    # The blocked ordering still fails the same measurement.
    blocked = select_patch_ids(rows, "train", 20, seed=5)
    assert prefix_tile_balance(rows, blocked, [10])[10]["tiles_present"] == 1


def test_interleaved_selection_rejects_bad_input():
    rows = _fixture_selection_rows()
    with pytest.raises(ValueError, match="n_total"):
        select_patch_ids_interleaved(rows, "train", 0, seed=1)
    with pytest.raises(ValueError, match="geo side"):
        select_patch_ids_interleaved(rows, "holdout", 4, seed=1)


def _shard_set_fixture(tmp_path, *, per_tile=3):
    train_ids, eval_ids, rows, parts, captions = _fixture(tmp_path, per_tile=per_tile,
                                                          eval_patches=2)
    selection = select_patch_ids_interleaved(rows, "train", len(train_ids), seed=3)
    assert_split_purity(selection, rows, "train")
    return train_ids, eval_ids, rows, parts, captions, selection


def test_build_shard_set_writes_manifest_incrementally_and_resumes(tmp_path):
    train_ids, eval_ids, rows, parts, captions, selection = _shard_set_fixture(tmp_path)
    out = tmp_path / "set"
    published = tmp_path / "published.json"
    result = build_shard_set(parts, selection, captions, rows, out, git_sha="test-sha",
                             seed=3, patches_per_shard=4, published_manifest=published)
    assert result["status"] == "complete"
    assert result["n_patches"] == len(train_ids)
    assert result["shards_written"] == math_ceil(len(train_ids), 4)
    assert published.is_file(), "the manifest must exist for the published path"
    # The same content in two places, and the per-shard records carry timings.
    assert all("seconds" in record and "tile_counts" in record for record in result["shards"])
    assert [record["index"] for record in result["shards"]] == \
           list(range(result["shards_written"]))

    # A rerun verifies every shard and skips it, adding nothing.
    again = build_shard_set(parts, selection, captions, rows, out, git_sha="test-sha",
                            seed=3, patches_per_shard=4, published_manifest=published)
    assert again["verified_on_resume"] == [record["index"] for record in result["shards"]]
    assert again["accepted_this_run"] == 0
    assert again["n_patches"] == result["n_patches"]
    assert [record["npz_sha256"] for record in again["shards"]] == \
           [record["npz_sha256"] for record in result["shards"]]


def test_build_shard_set_refuses_eval_and_duplicate_ids(tmp_path):
    train_ids, eval_ids, rows, parts, captions, selection = _shard_set_fixture(tmp_path)
    out = tmp_path / "set"
    with pytest.raises(ValueError, match="repeats a patch"):
        build_shard_set(parts, selection + [selection[0]], captions, rows, out,
                        git_sha="test-sha", seed=3, patches_per_shard=4)
    excluded = frozenset(eval_ids)
    result = build_shard_set(parts, list(selection) + list(excluded), captions, rows, out,
                             git_sha="test-sha", seed=3, patches_per_shard=4,
                             excluded_ids=excluded)
    written = [row["patch_id"] for record in result["shards"]
               for row in json.loads((out / record["sidecar"]).read_text())["rows"]]
    assert not set(written) & excluded, "excluded IDs must never reach a shard"
    assert result["skipped_upfront_patches"] == len(excluded)
    assert result["n_patches"] == len(train_ids)


def test_build_shard_set_disk_guard_stops_with_a_valid_prefix(tmp_path, monkeypatch):
    train_ids, eval_ids, rows, parts, captions, selection = _shard_set_fixture(tmp_path)
    out = tmp_path / "set"
    published = tmp_path / "published.json"
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type(
        "_U", (), {"free": 44_000_000_000, "total": 0, "used": 0})())
    result = build_shard_set(parts, selection, captions, rows, out, git_sha="test-sha",
                             seed=3, patches_per_shard=4, published_manifest=published,
                             min_free_bytes=45_000_000_000)
    assert result["status"] == "stopped"
    assert result["stopped_reason"] == "disk_guard"
    assert result["shards_written"] == 0, "nothing may be written below the disk floor"
    assert json.loads(published.read_text())["stopped_reason"] == "disk_guard"


def test_build_shard_set_speed_guard_stops_after_its_checkpoint(tmp_path):
    train_ids, eval_ids, rows, parts, captions, selection = _shard_set_fixture(tmp_path)
    out = tmp_path / "set"
    result = build_shard_set(parts, selection, captions, rows, out, git_sha="test-sha",
                             seed=3, patches_per_shard=2,
                             speed_guard_after_shards=1, speed_guard_max_seconds_per_patch=0.0)
    assert result["status"] == "stopped"
    assert result["stopped_reason"] == "speed_guard"
    assert result["shards_written"] >= 1
    verified, problems = verify_shard_files(out, result["shards"])
    assert len(verified) == result["shards_written"] and problems == []


def test_build_shard_set_matches_the_loop8_single_shard_bytes(tmp_path):
    """The multi-shard writer must not drift from the accepted single-shard format."""
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=2)
    single_dir = tmp_path / "single"
    single = build_shards(parts, train_ids, captions, rows, single_dir,
                          git_sha="test-sha", seed=3, patches_per_shard=len(train_ids))
    multi_dir = tmp_path / "multi"
    multi = build_shard_set(parts, train_ids, captions, rows, multi_dir, git_sha="test-sha",
                            seed=3, patches_per_shard=len(train_ids))
    for record_a, record_b in zip(single["shards"], multi["shards"]):
        assert record_a["npz_sha256"] == record_b["npz_sha256"]
        assert record_a["rows"] == record_b["rows"]


def _written_ids(out, manifest):
    return [row["patch_id"] for record in manifest["shards"]
            for row in json.loads((out / record["sidecar"]).read_text())["rows"]]


def test_build_shard_set_membership_follows_the_selection_not_the_archive(tmp_path):
    """Regression: shards were once filled in archive order, which blocked every tile.

    The archive is tile-major, so an archive-order writer puts a whole tile in shard 0
    and the prefix stratification of the selection never reaches disk. Shard k must hold
    selection[k * patches_per_shard : (k + 1) * patches_per_shard].
    """
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=4)
    selection = select_patch_ids_interleaved(rows, "train", len(train_ids), seed=3)
    assert len({rows[p]["mgrs_tile"] for p in selection[:4]}) == 2, "fixture must interleave"
    out = tmp_path / "ordered"
    manifest = build_shard_set(parts, selection, captions, rows, out,
                               git_sha="test-sha", seed=3, patches_per_shard=4)
    assert _written_ids(out, manifest) == selection
    # Bind every pixel row to the independently generated source for its ID.
    for record in manifest["shards"]:
        sidecar = json.loads((out / record["sidecar"]).read_text())
        with np.load(out / record["npz"]) as pixels:
            for i, row in enumerate(sidecar["rows"]):
                source_index = train_ids.index(row["patch_id"])
                for band in BANDS:
                    with MemoryFile(_tif_bytes(source_index * 13 + BANDS.index(band), band)) as memory:
                        with memory.open() as source:
                            np.testing.assert_array_equal(pixels[band][i], source.read(1))
    for record in manifest["shards"]:
        block = selection[record["index"] * 4:record["index"] * 4 + record["rows"]]
        assert block == [row["patch_id"] for row in
                         json.loads((out / record["sidecar"]).read_text())["rows"]]
    # The first shard is tile-balanced, which an archive-order writer could not do.
    first = json.loads((out / manifest["shards"][0]["sidecar"]).read_text())
    assert manifest["shards"][0]["tile_min"] >= 1 and manifest["shards"][0]["tile_min"] < 4
    assert len({row["mgrs_tile"] for row in first["rows"]}) == 2


def test_build_shard_set_windows_produce_identical_shards(tmp_path):
    """A window of shards per archive pass must not change a single output byte."""
    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=4)
    selection = select_patch_ids_interleaved(rows, "train", len(train_ids), seed=3)
    digests = []
    for window in (1, 2, 99):
        out = tmp_path / f"w{window}"
        manifest = build_shard_set(parts, selection, captions, rows, out,
                                   git_sha="test-sha", seed=3, patches_per_shard=3,
                                   shard_window=window)
        assert manifest["shard_window"] == min(window, manifest["shard_count"])
        assert len(manifest["archive_passes"]) == \
            math_ceil(len(train_ids), 3 * min(window, manifest["shard_count"]))
        digests.append([record["npz_sha256"] for record in manifest["shards"]])
    assert digests[0] == digests[1] == digests[2]
    assert default_shard_window(2000) >= 1
    assert default_shard_window(2000, ram_fraction=0.0001) >= 1


def test_verifier_checks_source_pixels_captions_and_all_row_checksums(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.verify_stage1_train_shards import main, verify

    train_ids, _eval, rows, parts, captions = _fixture(tmp_path, per_tile=4)
    selection = select_patch_ids_interleaved(rows, "train", len(train_ids), seed=3)
    out = tmp_path / "verified"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_shard_set(parts, selection, captions, rows, out,
                               git_sha="test-sha", seed=3, patches_per_shard=4,
                               published_manifest=manifest_path)
    source = tmp_path / "captions.parquet"
    pq.write_table(pa.table({"patch_id": train_ids, "type": ["captioning"] * len(train_ids),
                             "output": [captions[pid] for pid in train_ids]}), source)
    report = verify(out, manifest_path, excluded_dirs=[], captions_parquet=source,
                    archive_parts=parts)
    assert report["all_passed"], report["problems"]
    assert report["pixel_audit"]["checksum_rows_checked"] == len(train_ids)
    # Reproduce the legacy bug: pixels move independently of their sidecars.
    record = manifest["shards"][0]
    with np.load(out / record["npz"]) as pixels:
        scrambled = {band: pixels[band][::-1] for band in BANDS}
    np.savez(out / record["npz"], **scrambled)
    record["npz_sha256"] = hashlib.sha256((out / record["npz"]).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    sidecar_path = out / record["sidecar"]
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["rows"][0]["caption"] = "wrong caption"
    sidecar_path.write_text(json.dumps(sidecar))
    report_path = tmp_path / "report.json"
    argv = ["--shards", str(out), "--manifest", str(manifest_path),
            "--captions", str(source), "--report", str(report_path)]
    for part in parts:
        argv += ["--archive-part", str(part)]
    assert main(argv) == 1
    report = json.loads(report_path.read_text())
    for check in ("native_checksums_match", "raw_pixels_match", "source_captions_match"):
        assert not report["checks"][check]


def test_rebuild_phases_use_one_checked_selection_prefix(tmp_path):
    from scripts.build_stage1_train_shards import read_selected_ids
    ids = ["a", "b", "c", "d"]
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"seed": 26168, "patch_ids": ids,
                               "id_list_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()}))
    assert read_selected_ids(path, 2, 26168) == read_selected_ids(path, 4, 26168)[:2]
    with pytest.raises(ValueError, match="seed"):
        read_selected_ids(path, 4, 0)
    record = json.loads(path.read_text())
    record["patch_ids"][0] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="SHA256"):
        read_selected_ids(path, 4, 26168)
