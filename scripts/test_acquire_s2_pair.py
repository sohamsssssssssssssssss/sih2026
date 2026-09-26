import numpy as np
import pytest
from rasterio.windows import Window

from scripts.acquire_s2_pair import even_window, reflectance, upsample_scl


TILE = (10980, 10980)


def test_even_window_snaps_outward_to_even_offsets_and_sizes():
    window = even_window(Window(1401.3, 3057.6, 398.9, 450.2), TILE)

    assert (window.col_off, window.row_off) == (1400, 3056)
    assert window.width % 2 == 0 and window.height % 2 == 0
    assert window.col_off + window.width >= 1401.3 + 398.9
    assert window.row_off + window.height >= 3057.6 + 450.2


def test_even_window_refuses_oversized_aoi():
    with pytest.raises(ValueError, match="exceeds"):
        even_window(Window(0, 0, 900, 10), TILE)


@pytest.mark.parametrize(
    "window",
    [Window(10570, 4168, 654, 558), Window(-4, 10, 100, 100), Window(10, 10900, 100, 100)],
)
def test_even_window_refuses_aoi_outside_tile(window):
    # first case: the exact 44PLV window that was silently clipped and stretched
    with pytest.raises(ValueError, match="not inside"):
        even_window(window, TILE)


def test_reflectance_applies_declared_scale_offset_and_masks_dn_zero():
    values, valid = reflectance(np.array([[0, 1000, 2500]], dtype=np.uint16), 0.0001, -0.1)

    assert valid.tolist() == [[False, True, True]]
    assert values[0, 1:] == pytest.approx([0.0, 0.15])
    assert values[0, 0] == 0.0


def test_scl_upsample_nests_each_20m_cell_into_2x2():
    scl = np.array([[4, 9], [3, 6]], dtype=np.uint8)

    assert upsample_scl(scl)[:, :2].tolist() == [[4, 4], [4, 4], [3, 3], [3, 3]]
