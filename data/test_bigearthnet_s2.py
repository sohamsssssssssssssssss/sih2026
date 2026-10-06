import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from data.bigearthnet_s2 import BANDS, load_norm_contract, load_s2_patch


def write_band(patch: Path, band: str, value: int, shape=(2, 3), dtype="uint16", suffix=".tif"):
    path = patch / f"{patch.name}_{band}{suffix}"
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0],
                       count=1, dtype=dtype, transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.full(shape, value, dtype=dtype), 1)
    return path


@pytest.fixture
def patch(tmp_path):
    directory = tmp_path / "S2A_patch"
    directory.mkdir()
    for index, band in enumerate(BANDS, 1):
        write_band(directory, band, index * 100, shape=(2 + index % 3, 3 + index % 2))
    return directory


def test_band_order_dtype_shape_and_deterministic_resampling(patch):
    assert BANDS == ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    first = load_s2_patch(patch)
    assert first.shape == (12, 120, 120)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, load_s2_patch(patch), rtol=0, atol=0)
    for index in range(12):
        assert torch.all(first[index] == (index + 1) * 100)


def test_bilinear_resampling(patch):
    path = patch / f"{patch.name}_B01.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint16", transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.array([[0, 0], [0, 100]], dtype="uint16"), 1)
    result = load_s2_patch(patch)[0]
    assert result[0, 0] == 0
    assert result[-1, -1] == 100
    assert 0 < result[60, 60] < 100


def test_missing_duplicate_and_unexpected_bands_are_rejected(patch):
    (patch / f"{patch.name}_B12.tif").unlink()
    with pytest.raises(ValueError, match="Missing S2 bands: B12"):
        load_s2_patch(patch)
    write_band(patch, "B12", 1200)
    write_band(patch, "B01", 100, suffix=".tiff")
    with pytest.raises(ValueError, match="Duplicate S2 band mapping: B01"):
        load_s2_patch(patch)
    (patch / f"{patch.name}_B01.tiff").unlink()
    write_band(patch, "B10", 1000)
    with pytest.raises(ValueError, match="Unexpected S2 band mapping: B10"):
        load_s2_patch(patch)
    (patch / f"{patch.name}_B10.tif").unlink()
    write_band(patch, "B01", 1, suffix=".tiff")
    (patch / f"{patch.name}_B01.tiff").rename(patch / "unrelated.tiff")
    with pytest.raises(ValueError, match="Unexpected TIFF"):
        load_s2_patch(patch)


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_malformed_band_dtype_is_rejected(patch, dtype):
    write_band(patch, "B02", 5, dtype=dtype)
    with pytest.raises(ValueError, match="B02 must be a single-band uint16 TIFF"):
        load_s2_patch(patch)


def test_multiband_and_nonzero_nodata_are_rejected(patch):
    path = patch / f"{patch.name}_B03.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=2,
                       dtype="uint16", transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.ones((2, 2, 2), dtype="uint16"))
    with pytest.raises(ValueError, match="B03 must be a single-band"):
        load_s2_patch(patch)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint16", nodata=0, transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.ones((2, 2), dtype="uint16"), 1)
    assert load_s2_patch(patch)[2, 0, 0] == 1
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint16", nodata=65535, transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.ones((2, 2), dtype="uint16"), 1)
    with pytest.raises(ValueError, match="B03 declares unsupported nodata"):
        load_s2_patch(patch)


def test_frozen_contract_loads_and_normalizes_without_image_statistics(patch, tmp_path):
    contract = {
        "version": "1.0", "bands": list(BANDS), "input_dtype": "uint16", "output_dtype": "float32",
        "target_size": [120, 120], "resampling": "rasterio bilinear",
        "split_name": "caption-geo-split.v1/train", "divisor": 10000.0,
        "clip_min": 0.0, "clip_max": 0.5,
        "mean": {band: 0.1 for band in BANDS}, "std": {band: 0.2 for band in BANDS},
        "nodata": {"source_value": 0, "action": "reject patch with any raw zero when normalization is requested"},
        "raw_one_handling": "set pixels where all 12 resampled bands equal 1 to zero after standardization; retain band-specific ones",
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    write_band(patch, "B12", 6000)
    assert load_norm_contract(path)["bands"] == list(BANDS)
    tensor = load_s2_patch(patch, norm_contract=path)
    assert tensor.shape == (12, 120, 120) and torch.isfinite(tensor).all()
    torch.testing.assert_close(tensor[0], torch.full((120, 120), -0.45))
    torch.testing.assert_close(tensor[-1], torch.full((120, 120), 2.0))
    contract["bands"] = list(reversed(BANDS))
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Invalid Stage-1 normalization contract"):
        load_s2_patch(patch, norm_contract=path)
    contract["bands"] = list(BANDS)
    path.write_text(json.dumps(contract))
    write_band(patch, "B01", 0)
    with pytest.raises(ValueError, match="raw nodata zero"):
        load_s2_patch(patch, norm_contract=path)
    for band in BANDS:
        write_band(patch, band, 1)
    assert torch.all(load_s2_patch(patch, norm_contract=path) == 0)
    write_band(patch, "B04", 100)
    assert torch.all(load_s2_patch(patch, norm_contract=path)[BANDS.index("B09")] != 0)
