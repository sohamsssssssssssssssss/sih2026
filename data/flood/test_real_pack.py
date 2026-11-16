"""Explicit real-pack gate; skips when local data is unavailable. Never downloads."""
import os
from pathlib import Path

import pytest
from data.flood.build_pack import verify_pack, sha256


def test_real_pack_all_hashes_and_every_chip():
    path=Path(os.environ.get('FLOOD_PACK',Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    if not (path/'MANIFEST.json').exists():
        pytest.skip('real pack not built; run this test explicitly for the T14 gate')
    manifest=verify_pack(path)
    assert sum(manifest['counts']['per_split'].values())==len(manifest['chips'])
    assert manifest['checks']['alignment_tolerance_pixels']==1e-6
    for chip in manifest['chips'].values():
        assert all(value<=1e-6 for value in chip['alignment'].values())
    print('REAL PACK: every file hash and all chip/layer hashes PASS')
    print('MANIFEST sha256:',sha256(path/'MANIFEST.json'))
