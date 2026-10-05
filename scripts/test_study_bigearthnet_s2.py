import numpy as np
from PIL import Image

from scripts.study_bigearthnet_s2 import quicklook, stats


def test_stats_and_quicklook(tmp_path):
    result = stats(np.array([0, 1, 1, 5, 10000, 65535], dtype=np.float32))
    assert result["min"] == 0 and result["max"] == 65535
    assert result["frac_eq_1"] == 2 / 6
    assert result["frac_ge_dtype_max"] == 1 / 6
    assert result["frac_le_5"] == 4 / 6
    image = np.ones((12, 120, 120), dtype=np.float32)
    quicklook(image, tmp_path / "quicklook.png")
    assert Image.open(tmp_path / "quicklook.png").size == (360, 120)
