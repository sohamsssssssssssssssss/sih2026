"""Synthetic regression tests for the offline Sentinel-1 change workflow."""

import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import scripts.s1_offline_change as change


# Constants from the workflow
PINNED_BEFORE = "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE"
PINNED_AFTER = "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE"
SAR_BANDS = ("VV", "VH", "dataMask")
EXPECTED_GRID = {
    "width": 749,
    "height": 1649,
    "epsg": 32645,
    "resolution_m": 10,
    "bounds": (432970.0, 2867120.0, 440460.0, 2883610.0),
}
NODATA_VALUE = -9999.0


def make_geotiff(width: int, height: int, bounds, epsg: int,
                 vv_val: float = 0.1, vh_val: float = 0.02, mask_val: float = 1.0) -> bytes:
    """Create a 3-band float32 GeoTIFF (VV, VH, dataMask)."""
    vv = np.full((height, width), vv_val, dtype="float32")
    vh = np.full((height, width), vh_val, dtype="float32")
    mask = np.full((height, width), mask_val, dtype="float32")
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=width, height=height, count=3, dtype="float32",
            crs=f"EPSG:{epsg}", transform=from_bounds(*bounds, width, height),
        ) as dst:
            dst.write(np.stack([vv, vh, mask]).astype("float32"))
        return mem.read()


def make_userdata(product_id: str) -> bytes:
    """Create valid userdata.json with exactly one orbit and matching product ID."""
    return json.dumps({
        "orbits": [{
            "tiles": [{"sentinel1ProductId": product_id}]
        }]
    }).encode("utf-8")


def make_tar(tiff_bytes: bytes, userdata_bytes: bytes) -> bytes:
    """Create a TAR archive with default.tif and userdata.json."""
    members = [("default.tif", tiff_bytes), ("userdata.json", userdata_bytes)]
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return bio.getvalue()


def make_tar_single(tiff_bytes: bytes) -> bytes:
    """Create a TAR archive with only default.tif (missing userdata.json)."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:") as archive:
        info = tarfile.TarInfo(name="default.tif")
        info.size = len(tiff_bytes)
        archive.addfile(info, io.BytesIO(tiff_bytes))
    return bio.getvalue()


def test_known_log_ratio_values():
    """Test VV/VH change computation with known values."""
    vv_before = np.array([[0.1, 0.2], [0.05, 0.15]], dtype="float32")
    vh_before = np.array([[0.02, 0.04], [0.01, 0.03]], dtype="float32")
    vv_after = np.array([[0.01, 0.2], [0.05, 0.075]], dtype="float32")
    vh_after = np.array([[0.002, 0.04], [0.01, 0.015]], dtype="float32")
    mask_before = np.ones_like(vv_before)
    mask_after = np.ones_like(vv_before)

    vv_change, vh_change, valid = change.compute_change(
        vv_before, vh_before, vv_after, vh_after, mask_before, mask_after
    )

    assert valid.all()
    # VV: 0.1 -> 0.01 = -10 dB, 0.2 -> 0.2 = 0 dB, 0.05 -> 0.05 = 0 dB, 0.15 -> 0.075 = -3.01 dB
    expected_vv = np.array([[-10.0, 0.0], [0.0, -3.0103]], dtype="float32")
    np.testing.assert_allclose(vv_change, expected_vv, rtol=1e-3)
    # VH: 0.02 -> 0.002 = -10 dB, 0.04 -> 0.04 = 0 dB, 0.01 -> 0.01 = 0 dB, 0.03 -> 0.015 = -3.01 dB
    expected_vh = np.array([[-10.0, 0.0], [0.0, -3.0103]], dtype="float32")
    np.testing.assert_allclose(vh_change, expected_vh, rtol=1e-3)


def test_invalid_and_nonfinite_inputs():
    """Test that invalid/nonfinite inputs are masked out."""
    vv_before = np.array([[0.1, np.nan], [np.inf, 0.1]], dtype="float32")
    vh_before = np.array([[0.02, 0.02], [0.02, 0.02]], dtype="float32")
    vv_after = np.array([[0.1, 0.1], [0.1, 0.1]], dtype="float32")
    vh_after = np.array([[0.02, 0.02], [0.02, 0.02]], dtype="float32")
    mask_before = np.array([[1, 1], [1, 1]], dtype="float32")
    mask_after = np.array([[1, 1], [1, 1]], dtype="float32")

    vv_change, vh_change, valid = change.compute_change(
        vv_before, vh_before, vv_after, vh_after, mask_before, mask_after
    )

    assert valid[0, 0] == True
    assert valid[0, 1] == False  # NaN in before
    assert valid[1, 0] == False  # Inf in before
    assert valid[1, 1] == True
    assert vv_change[0, 0] == 0.0
    assert vh_change[0, 0] == 0.0


def test_zero_negative_power():
    """Test that zero or negative power values are masked out."""
    vv_before = np.array([[0.1, 0.0, -0.1]], dtype="float32")
    vh_before = np.array([[0.02, 0.02, 0.02]], dtype="float32")
    vv_after = np.array([[0.1, 0.1, 0.1]], dtype="float32")
    vh_after = np.array([[0.02, 0.02, 0.02]], dtype="float32")
    mask_before = np.ones((1, 3), dtype="float32")
    mask_after = np.ones((1, 3), dtype="float32")

    vv_change, vh_change, valid = change.compute_change(
        vv_before, vh_before, vv_after, vh_after, mask_before, mask_after
    )

    assert valid[0, 0] == True
    assert valid[0, 1] == False  # zero
    assert valid[0, 2] == False  # negative


def test_mismatched_grids():
    """Test that on_grid rejects mismatched grid parameters."""
    tiff = make_geotiff(100, 100, (0, 0, 1000, 1000), 32645)
    # Wrong CRS
    with pytest.raises(change.ChangeError, match="requested grid"):
        change.on_grid(tiff, {**EXPECTED_GRID, "epsg": 4326})
    # Wrong dimensions
    with pytest.raises(change.ChangeError, match="requested grid"):
        change.on_grid(tiff, {**EXPECTED_GRID, "width": 100, "height": 100})
    # Wrong bounds
    with pytest.raises(change.ChangeError, match="requested grid"):
        change.on_grid(tiff, {**EXPECTED_GRID, "bounds": (0, 0, 1000, 1000)})


def test_wrong_checksums():
    """Test that verify_checksum rejects wrong checksums."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        tmp = Path(f.name)
    try:
        with pytest.raises(change.ChangeError, match="checksum mismatch"):
            change.verify_checksum(tmp, "0" * 64, "TEST")
    finally:
        tmp.unlink()


def test_wrong_acquisition_ids():
    """Test that parse_userdata rejects wrong product IDs."""
    userdata = make_userdata("S1A_WRONG_PRODUCT")
    with pytest.raises(change.ChangeError, match="source product ID mismatch"):
        change.parse_userdata(userdata, PINNED_BEFORE)

    # Test multiple source IDs
    userdata_multi = json.dumps({
        "orbits": [{"tiles": [
            {"sentinel1ProductId": PINNED_BEFORE},
            {"sentinel1ProductId": "S1A_OTHER"}
        ]}]
    }).encode("utf-8")
    with pytest.raises(change.ChangeError, match="source product ID mismatch"):
        change.parse_userdata(userdata_multi, PINNED_BEFORE)


def test_parse_tar_response_valid():
    """Test valid TAR parsing."""
    tiff = make_geotiff(100, 100, (0, 0, 1000, 1000), 32645)
    userdata = make_userdata(PINNED_BEFORE)
    tar_bytes = make_tar(tiff, userdata)
    tiff_out, userdata_out = change.parse_tar_response(tar_bytes)
    assert tiff_out == tiff
    assert userdata_out == userdata


def test_parse_tar_response_missing_member():
    """Test TAR missing a required member."""
    tiff = make_geotiff(100, 100, (0, 0, 1000, 1000), 32645)
    tar_bytes = make_tar_single(tiff)  # Missing userdata.json
    with pytest.raises(change.ChangeError, match="missing default.tif or userdata.json"):
        change.parse_tar_response(tar_bytes)


def test_parse_tar_response_malformed():
    """Test malformed TAR."""
    with pytest.raises(change.ChangeError, match="malformed TAR"):
        change.parse_tar_response(b"not a tar")


def test_full_workflow_synthetic():
    """End-to-end test with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        before_tar = tmpdir / "before.tar"
        after_tar = tmpdir / "after.tar"
        output_dir = tmpdir / "output"

        # Create synthetic TARs
        tiff_before = make_geotiff(
            EXPECTED_GRID["width"], EXPECTED_GRID["height"],
            EXPECTED_GRID["bounds"], EXPECTED_GRID["epsg"],
            vv_val=0.1, vh_val=0.02, mask_val=1.0
        )
        tiff_after = make_geotiff(
            EXPECTED_GRID["width"], EXPECTED_GRID["height"],
            EXPECTED_GRID["bounds"], EXPECTED_GRID["epsg"],
            vv_val=0.01, vh_val=0.002, mask_val=1.0
        )

        before_bytes = make_tar(tiff_before, make_userdata(PINNED_BEFORE))
        after_bytes = make_tar(tiff_after, make_userdata(PINNED_AFTER))

        before_tar.write_bytes(before_bytes)
        after_tar.write_bytes(after_bytes)

        before_sha = change.sha256_file(before_tar)
        after_sha = change.sha256_file(after_tar)

        # Run workflow
        import sys
        sys.argv = ["s1_offline_change.py",
                    "--before", str(before_tar),
                    "--after", str(after_tar),
                    "--before-sha256", before_sha,
                    "--after-sha256", after_sha,
                    "--output-dir", str(output_dir)]
        change.main()

        # Verify outputs
        assert (output_dir / "sar_change.tif").exists()
        assert (output_dir / "sar_change.json").exists()

        report = json.loads((output_dir / "sar_change.json").read_text())
        assert report["validity_criteria"]["valid_fraction"] == 1.0
        # VV change: 0.1 -> 0.01 = -10 dB
        assert abs(report["statistics"]["VV_change_dB"]["mean"] - (-10.0)) < 0.01
        # VH change: 0.02 -> 0.002 = -10 dB
        assert abs(report["statistics"]["VH_change_dB"]["mean"] - (-10.0)) < 0.01


def test_failure_cleanup():
    """Test that failed runs don't leave partial outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        before_tar = tmpdir / "before.tar"
        after_tar = tmpdir / "after.tar"
        output_dir = tmpdir / "output"

        # Valid before, invalid after (wrong product ID)
        tiff_before = make_geotiff(
            EXPECTED_GRID["width"], EXPECTED_GRID["height"],
            EXPECTED_GRID["bounds"], EXPECTED_GRID["epsg"],
            vv_val=0.1, vh_val=0.02, mask_val=1.0
        )
        tiff_after = make_geotiff(
            EXPECTED_GRID["width"], EXPECTED_GRID["height"],
            EXPECTED_GRID["bounds"], EXPECTED_GRID["epsg"],
            vv_val=0.01, vh_val=0.002, mask_val=1.0
        )

        before_bytes = make_tar(tiff_before, make_userdata(PINNED_BEFORE))
        after_bytes = make_tar(tiff_after, make_userdata("S1A_WRONG"))

        before_tar.write_bytes(before_bytes)
        after_tar.write_bytes(after_bytes)

        before_sha = change.sha256_file(before_tar)
        after_sha = change.sha256_file(after_tar)

        import sys
        sys.argv = ["s1_offline_change.py",
                    "--before", str(before_tar),
                    "--after", str(after_tar),
                    "--before-sha256", before_sha,
                    "--after-sha256", after_sha,
                    "--output-dir", str(output_dir)]

        with pytest.raises(change.ChangeError, match="source product ID mismatch"):
            change.main()

        # Output directory should not exist or be empty
        if output_dir.exists():
            assert list(output_dir.iterdir()) == []


def test_descriptive_stats():
    """Test descriptive statistics computation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    valid = np.array([True, True, False, True, True])
    stats = change.descriptive_stats(arr, valid)
    assert stats["count"] == 4
    assert stats["mean"] == 3.0
    assert abs(stats["std"] - np.std([1, 2, 4, 5])) < 1e-6
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0

    # All invalid
    stats = change.descriptive_stats(arr, np.zeros_like(valid, dtype=bool))
    assert stats["count"] == 0
    assert stats["mean"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_shared_validator_rejects_duplicate_userdata_keys():
    with pytest.raises(change.ChangeError, match="duplicate object keys"):
        change.parse_userdata(b'{"orbits": [], "orbits": []}', PINNED_BEFORE)


def test_shared_tar_validator_enforces_response_limit(monkeypatch):
    content = make_tar(b"tiff", make_userdata(PINNED_BEFORE))
    monkeypatch.setattr(change.s1, "MAX_PROCESS_BYTES", len(content) - 1)
    with pytest.raises(change.ChangeError, match="response exceeds byte limit"):
        change.parse_tar_response(content)


def test_shared_tar_validator_accepts_only_complete_missing_eof():
    content = make_tar(b"tiff", make_userdata(PINNED_BEFORE))
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        list(archive)
        content = content[:archive.offset]
    assert change.parse_tar_response(content) == (b"tiff", make_userdata(PINNED_BEFORE))
    with pytest.raises(change.ChangeError):
        change.parse_tar_response(content[:-1])
