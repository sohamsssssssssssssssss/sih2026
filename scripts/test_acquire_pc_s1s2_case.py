from affine import Affine

from scripts.acquire_pc_s1s2_case import grids_aligned, unsigned


def test_grids_aligned_requires_same_spacing_and_whole_pixel_offset():
    s2 = Affine(10, 0, 199980, 0, -10, 2000040)

    assert grids_aligned(Affine(10, 0, 210840, 0, -10, 1931400), s2)
    assert not grids_aligned(Affine(10, 0, 210845, 0, -10, 1931400), s2)  # half-pixel shift
    assert not grids_aligned(Affine(20, 0, 199980, 0, -20, 2000040), s2)  # different spacing


def test_unsigned_strips_sas_token():
    href = "https://x.blob.core.windows.net/c/a.tif?st=2026&se=2026&sig=secret"

    assert unsigned(href) == "https://x.blob.core.windows.net/c/a.tif"
