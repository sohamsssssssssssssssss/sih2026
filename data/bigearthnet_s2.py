"""Prepare one accessible BigEarthNet S2 patch for a 12-channel Conv2d.

Input is single-band uint16 TIFF digital numbers. Output is float32 CHW,
resampled to 120x120 with rasterio bilinear interpolation; no normalization.
"""

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
import torch


BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
SIZE = 120


def load_s2_patch(patch_dir: Path) -> torch.Tensor:
    """Return float32 [12, 120, 120] in ``BANDS`` order from one patch directory."""
    patch_dir = Path(patch_dir)
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
            if source.nodata is not None:
                raise ValueError(f"{band} declares nodata; masked pixels need an explicit policy")
            arrays.append(source.read(1, out_shape=(SIZE, SIZE), out_dtype="float32", resampling=Resampling.bilinear))
    return torch.stack([torch.from_numpy(array) for array in arrays])
