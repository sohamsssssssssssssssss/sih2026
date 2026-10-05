from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from models.change import ChangeModel
from models.change.model import CHANGE_THRESHOLD


def write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    transform=None,
    nodata=None,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform or from_origin(10, 20, 0.1, 0.1),
        nodata=nodata,
    ) as dataset:
        dataset.write(data.astype("float32"))
    return path


@pytest.fixture
def identical_pair(tmp_path):
    image = np.arange(48, dtype="float32").reshape(3, 4, 4)
    return write_raster(tmp_path / "t1.tif", image), write_raster(
        tmp_path / "t2.tif", image
    )


def test_identical_pair_has_zero_change_and_is_deterministic(identical_pair) -> None:
    model = ChangeModel()
    first = model.infer([str(path) for path in identical_pair], "What changed?")
    second = model.infer([str(path) for path in identical_pair], "What changed?")

    assert first == second
    statistics = first["evidence"][1]
    assert statistics["change_magnitude"]["max"] == 0
    assert statistics["changed_pixels"] == 0
    assert statistics["changed_fraction"] == 0
    assert statistics["threshold"] == CHANGE_THRESHOLD
    assert statistics["threshold_rule"].endswith("> 0.1")
    assert first["confidence"] is None
    assert not any(item["type"] == "change_extent" for item in first["evidence"])


def test_threshold_is_strictly_greater_than_point_one(tmp_path) -> None:
    before = np.ones((5, 2, 2), dtype="float32")
    after = before.copy()
    after[3] = 1.5
    result = ChangeModel().infer(
        [
            str(write_raster(tmp_path / "t1.tif", before)),
            str(write_raster(tmp_path / "t2.tif", after)),
        ],
        "What changed?",
    )

    statistics = result["evidence"][1]
    assert statistics["change_magnitude"]["mean"] == pytest.approx(0.1)
    assert statistics["changed_pixels"] == 0


def test_known_rgb_region_produces_spatial_extent(tmp_path) -> None:
    before = np.zeros((3, 4, 4), dtype="float32")
    after = before.copy()
    after[:, 1:3, 2:4] = 100
    result = ChangeModel().infer(
        [
            str(write_raster(tmp_path / "before.tif", before)),
            str(write_raster(tmp_path / "after.tif", after)),
        ],
        "Did built-up area increase?",
    )

    statistics = result["evidence"][1]
    assert statistics["method"] == "normalized_rgb_difference_heuristic"
    assert statistics["semantic_interpretation"] is False
    assert statistics["changed_pixels"] == 4
    assert result["evidence"][3]["coordinates"] == [0.5, 0.25, 1.0, 0.75]
    assert "does not infer semantic change classes" in result["answer"]
    assert "built-up" not in result["answer"]


def test_both_temporal_inputs_are_recorded(identical_pair) -> None:
    inputs = ChangeModel().infer(
        [str(path) for path in identical_pair], "What changed?"
    )["evidence"][0]

    # Paths outside the repository are reduced to their file name.
    assert inputs["t1"]["path"] == identical_pair[0].name == "t1.tif"
    assert inputs["t2"]["path"] == identical_pair[1].name == "t2.tif"
    assert inputs["t1"]["sha256"] != ""


def test_temporal_inputs_use_repo_relative_paths(identical_pair, monkeypatch) -> None:
    import models.paths

    monkeypatch.setattr(models.paths, "REPO_ROOT", identical_pair[0].parent.parent)
    inputs = ChangeModel().infer(
        [str(path) for path in identical_pair], "What changed?"
    )["evidence"][0]

    folder = identical_pair[0].parent.name
    assert inputs["t1"]["path"] == f"{folder}/t1.tif"
    assert inputs["t2"]["path"] == f"{folder}/t2.tif"
    assert str(identical_pair[0].parent) not in str(inputs)
    assert inputs["temporal_order"] == "input_1_is_t1_input_2_is_t2"


@pytest.mark.parametrize(
    ("shape", "crs", "transform"),
    [
        ((3, 3, 4), "EPSG:4326", None),
        ((3, 4, 4), "EPSG:3857", None),
        ((3, 4, 4), "EPSG:4326", from_origin(11, 20, 0.1, 0.1)),
    ],
)
def test_incompatible_dimensions_crs_or_grid_rejected(
    identical_pair, tmp_path, shape, crs, transform
) -> None:
    incompatible = write_raster(
        tmp_path / f"incompatible-{crs.split(':')[-1]}-{shape[1]}.tif",
        np.ones(shape),
        crs=crs,
        transform=transform,
    )

    with pytest.raises(ValueError, match="identical bands, CRS, and pixel grid"):
        ChangeModel().infer([str(identical_pair[0]), str(incompatible)], "Change?")


def test_no_covalid_pixels_rejected(tmp_path) -> None:
    before = np.zeros((3, 2, 2), dtype="float32")
    after = np.ones((3, 2, 2), dtype="float32")
    with pytest.raises(ValueError, match="no co-valid RGB pixels"):
        ChangeModel().infer(
            [
                str(write_raster(tmp_path / "t1.tif", before, nodata=0)),
                str(write_raster(tmp_path / "t2.tif", after)),
            ],
            "Change?",
        )


def test_nonfinite_and_nodata_pixels_are_excluded(tmp_path) -> None:
    before = np.ones((3, 2, 2), dtype="float32")
    after = before.copy()
    before[:, 0, 0] = np.nan
    after[:, 0, 1] = np.inf
    before[:, 1, 0] = -9999
    result = ChangeModel().infer(
        [
            str(write_raster(tmp_path / "t1.tif", before, nodata=-9999)),
            str(write_raster(tmp_path / "t2.tif", after, nodata=-9999)),
        ],
        "Change?",
    )

    coverage = result["evidence"][2]
    assert coverage["valid_pixels"] == 1
    assert coverage["valid_fraction"] == 0.25
    assert result["evidence"][1]["change_magnitude"]["max"] == 0


def test_multispectral_index_deltas_are_reported(tmp_path) -> None:
    before = np.ones((5, 2, 2), dtype="float32")
    after = before.copy()
    before[4] = after[4] = 1
    after[3, 0, 0] = 3
    result = ChangeModel().infer(
        [
            str(write_raster(tmp_path / "t1.tif", before)),
            str(write_raster(tmp_path / "t2.tif", after)),
        ],
        "What changed?",
    )

    statistics = result["evidence"][1]
    assert statistics["method"] == "spectral_index_delta"
    assert statistics["delta_ndvi"]["max"] > 0
    assert statistics["delta_ndwi"]["mean"] < 0


def test_single_image_is_rejected(identical_pair) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        ChangeModel().infer([str(identical_pair[0])], "Change?")
