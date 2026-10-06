"""Prepare one accessible BigEarthNet S2 patch for a 12-channel Conv2d.

Input is single-band uint16 TIFF digital numbers. Output is float32 CHW,
resampled to 120x120 with rasterio bilinear interpolation; no normalization.

``apply_contract`` is the single normalization path: the TIFF loader below and
the Stage-1 shard loader (data/bigearthnet_shards.py) both call it, so a shard
and an extracted directory produce identical pixels for identical input.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import torch


BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
SIZE = 120
# BigEarthNet V2 S2 stores every band at its own ground sampling distance:
# 10 m bands (B02, B03, B04, B08) are 120x120, 20 m bands (B05, B06, B07, B8A,
# B11, B12) are 60x60, and 60 m bands (B01, B09) are 20x20. Measured from the
# archive, see results/stage1-loop8-preregistration.json amendments[2]. The
# frozen contract upsamples all twelve to 120x120 with rasterio bilinear.
NATIVE_SIZE = {
    "B02": 120, "B03": 120, "B04": 120, "B08": 120,
    "B05": 60, "B06": 60, "B07": 60, "B8A": 60, "B11": 60, "B12": 60,
    "B01": 20, "B09": 20,
}


def load_norm_contract(path: Path) -> dict:
    contract = json.loads(Path(path).read_text())
    if (contract.get("version") != "1.0" or tuple(contract.get("bands", ())) != BANDS
            or contract.get("input_dtype") != "uint16" or contract.get("output_dtype") != "float32"
            or contract.get("target_size") != [SIZE, SIZE] or contract.get("resampling") != "rasterio bilinear"
            or contract.get("split_name") != "caption-geo-split.v1/train"
            or contract.get("nodata") != {"source_value": 0, "action": "reject patch with any raw zero when normalization is requested"}
            or contract.get("raw_one_handling") != "set pixels where all 12 resampled bands equal 1 to zero after standardization; retain band-specific ones"
            or set(contract.get("mean", {})) != set(BANDS) or set(contract.get("std", {})) != set(BANDS)
            or not all(np.isfinite(contract["mean"][band]) and np.isfinite(contract["std"][band])
                       and contract["std"][band] > 0 for band in BANDS)
            or contract.get("divisor") != 10000.0 or contract.get("clip_min") != 0.0
            or not isinstance(contract.get("clip_max"), (int, float)) or not np.isfinite(contract["clip_max"])
            or contract["clip_max"] <= 0):
        raise ValueError("Invalid Stage-1 normalization contract")
    return contract


def resample_band(array: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Bilinearly resample one native uint16 band to ``size`` x ``size`` float32.

    The pixels are produced by rasterio through an in-memory GeoTIFF, so a
    shard band and a file band go through the same rasterio code path as the
    frozen contract requires. Already-120x120 bands are returned unchanged.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("A band must be a 2-D array")
    if array.shape == (size, size):
        return array.astype(np.float32, copy=False)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=array.shape[0], width=array.shape[1],
                         count=1, dtype=array.dtype) as dataset:
            dataset.write(array, 1)
        with memory.open() as source:
            return source.read(1, out_shape=(size, size), out_dtype="float32",
                               resampling=Resampling.bilinear)


def resample_bands(bands: dict[str, np.ndarray], size: int = SIZE) -> list[np.ndarray]:
    """Resample a whole patch's bands in the frozen band order."""
    return [resample_band(bands[band], size) for band in BANDS]


def apply_contract(array, contract: dict) -> torch.Tensor:
    """Return the contract-normalized float32 CHW tensor for uint16 pixels.

    Same order of operations as before this function existed (divide, clamp,
    standardize, then zero the all-band-one pixels); it is only factored out so
    the shard loader cannot drift from the extracted-directory loader.
    """
    tensor = array if torch.is_tensor(array) else torch.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor.to(torch.float32)
    all_one = (tensor == 1).all(dim=0)
    means = torch.tensor([contract["mean"][band] for band in BANDS], dtype=torch.float32)[:, None, None]
    stds = torch.tensor([contract["std"][band] for band in BANDS], dtype=torch.float32)[:, None, None]
    tensor = tensor.div(contract["divisor"]).clamp(contract["clip_min"], contract["clip_max"])
    tensor = tensor.sub(means).div(stds)
    tensor[:, all_one] = 0
    return tensor


def load_s2_patch(patch_dir: Path, *, norm_contract: Path | None = None) -> torch.Tensor:
    """Return float32 CHW; raw DN unless a frozen train-fit contract is supplied."""
    patch_dir = Path(patch_dir)
    contract = load_norm_contract(norm_contract) if norm_contract is not None else None
    if not patch_dir.is_dir():
        raise ValueError(f"Patch directory does not exist: {patch_dir}")
    paths = {}
    for path in patch_dir.iterdir():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        prefix = f"{patch_dir.name}_"
        if not path.name.startswith(prefix):
            raise ValueError(f"Unexpected TIFF in patch directory: {path.name}")
        band = path.stem[len(prefix):]
        if band not in BANDS:
            raise ValueError(f"Unexpected S2 band mapping: {band}")
        if band in paths:
            raise ValueError(f"Duplicate S2 band mapping: {band}")
        paths[band] = path
    missing = set(BANDS) - paths.keys()
    if missing:
        raise ValueError(f"Missing S2 bands: {', '.join(sorted(missing))}")

    arrays = []
    for band in BANDS:
        with rasterio.open(paths[band]) as source:
            if source.count != 1 or source.dtypes[0] != "uint16":
                raise ValueError(f"{band} must be a single-band uint16 TIFF")
            # BigEarthNet V2 declares zero as nodata. Keep zeros in raw DN scale;
            # rasterio bilinear resampling includes them, without mask filling.
            if source.nodata not in (None, 0):
                raise ValueError(f"{band} declares unsupported nodata: {source.nodata}")
            native = source.read(1)
        if contract is not None and np.any(native == 0):
            raise ValueError(f"{band} contains raw nodata zero")
        arrays.append(resample_band(native))
    tensor = torch.stack([torch.from_numpy(array) for array in arrays])
    if contract is not None:
        tensor = apply_contract(tensor, contract)
    return tensor
