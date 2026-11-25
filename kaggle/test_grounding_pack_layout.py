from hashlib import sha256
import json
import zipfile

import pytest

from kaggle.run_grounding_baselines import (
    PACK_NAME, discover_pack, open_pack, verify_manifest,
)


@pytest.fixture
def payload():
    data = b"fixture"
    manifest = json.dumps([{"file": "images/sample", "bytes": len(data),
                            "sha256": sha256(data).hexdigest()}]).encode()
    return {"MANIFEST.json": manifest, "images/sample": data}


def write_extracted(root, payload):
    for name, data in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_zip_layout(tmp_path, payload):
    path = tmp_path / (PACK_NAME + ".zip")
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with open_pack(discover_pack(tmp_path), sha256(path.read_bytes()).hexdigest()) as pack:
        verify_manifest(pack)
        assert pack.read("images/sample") == b"fixture"


@pytest.mark.parametrize(("change", "message"), [
    (None, None), ("corrupt", "images/sample"), ("missing", "missing="),
    ("extra", "extra="), ("manifest", "MANIFEST.json SHA-256 mismatch"),
])
def test_extracted_layout(tmp_path, payload, change, message):
    root = tmp_path / "datasets" / "owner" / PACK_NAME
    write_extracted(root, payload)
    if change == "corrupt":
        (root / "images/sample").write_bytes(b"fixturE")
    elif change == "missing":
        (root / "images/sample").unlink()
    elif change == "extra":
        (root / "extra").touch()
    elif change == "manifest":
        (root / "MANIFEST.json").write_bytes(b"[]")
    def check():
        with open_pack(discover_pack(tmp_path), "", sha256(payload["MANIFEST.json"]).hexdigest()) as pack:
            verify_manifest(pack)
            assert pack.read("images/sample") == b"fixture"

    if message:
        with pytest.raises(ValueError, match=f"HARD STOP: .*{message}"):
            check()
    else:
        check()


def test_two_packs(tmp_path, payload):
    write_extracted(tmp_path / PACK_NAME, payload)
    (tmp_path / (PACK_NAME + ".zip")).touch()
    with pytest.raises(ValueError, match="found 1 ZIPs and 1 extracted"):
        discover_pack(tmp_path)


def test_zero_packs(tmp_path):
    with pytest.raises(ValueError, match="found 0 ZIPs and 0 extracted"):
        discover_pack(tmp_path)
