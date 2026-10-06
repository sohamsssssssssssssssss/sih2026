import io
import tarfile

import pytest

from scripts.extract_selected_s2 import extract


def test_extract_only_requested_member(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("BigEarthNet-S2/tile/patch/patch_B01.tif", "BigEarthNet-S2/tile/patch/patch_B10.tif"):
            content = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    selected = {"BigEarthNet-S2/tile/patch/patch_B01.tif"}
    assert extract(io.BytesIO(buffer.getvalue()), selected, tmp_path) == 1
    assert (tmp_path / next(iter(selected))).read_bytes() == next(iter(selected)).encode()
    assert not (tmp_path / "BigEarthNet-S2/tile/patch/patch_B10.tif").exists()
    with pytest.raises(ValueError, match="missing"):
        extract(io.BytesIO(buffer.getvalue()), {"BigEarthNet-S2/missing.tif"}, tmp_path)
