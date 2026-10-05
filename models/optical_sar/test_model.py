from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from models.optical_sar import OpticalSARModel


def write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    transform=None,
    descriptions: tuple[str, ...] | None = None,
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
    ) as dataset:
        dataset.write(data.astype("float32"))
        if descriptions:
            for index, description in enumerate(descriptions, 1):
                dataset.set_band_description(index, description)
    return path


@pytest.fixture
def pair(tmp_path):
    optical = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.3, 0.4], [0.5, 0.6]],
            [[0.2, 0.2], [0.4, 0.4]],
            [[0.6, 0.4], [0.8, 0.2]],
            [[1, 1], [1, 1]],
        ],
        dtype="float32",
    )
    sar = np.array(
        [
            [[1.0, 2.0], [4.0, 8.0]],
            [[0.5, 1.0], [1.0, 2.0]],
            [[1, 1], [1, 1]],
        ],
        dtype="float32",
    )
    return write_raster(tmp_path / "optical.tif", optical), write_raster(
        tmp_path / "sar.tif", sar
    )


def test_pair_uses_both_modalities_and_is_deterministic(pair) -> None:
    model = OpticalSARModel()
    first = model.infer([str(pair[0]), str(pair[1])], "Analyze together")
    second = model.infer([str(pair[0]), str(pair[1])], "Analyze together")

    assert first == second
    assert first["confidence"] is None
    assert [item["type"] for item in first["evidence"]] == [
        "optical_statistics",
        "sar_statistics",
        "joint_valid_coverage",
    ]
    assert first["evidence"][0]["indices"]["ndvi"]["valid_pixels"] == 4
    assert first["evidence"][1]["vv_db"]["valid_pixels"] == 4
    assert first["evidence"][2]["modalities"] == ["optical", "sar"]
    assert first["evidence"][2]["valid_fraction"] == 1.0


def test_reversed_input_order_is_supported(pair) -> None:
    evidence = OpticalSARModel().infer(
        [str(pair[1]), str(pair[0])], "Analyze together"
    )["evidence"]

    assert evidence[0]["source"]["path"] == pair[0].name == "optical.tif"
    assert evidence[1]["source"]["path"] == pair[1].name == "sar.tif"
    assert str(pair[0].parent) not in str(evidence)


@pytest.mark.parametrize(
    ("shape", "crs", "message"),
    [((3, 3, 3), "EPSG:4326", "same CRS and pixel grid"), ((3, 2, 2), "EPSG:3857", "same CRS and pixel grid")],
)
def test_incompatible_grid_or_crs_is_rejected(pair, tmp_path, shape, crs, message) -> None:
    sar = np.ones(shape, dtype="float32")
    incompatible = write_raster(tmp_path / f"bad-{crs.split(':')[-1]}.tif", sar, crs=crs)

    with pytest.raises(ValueError, match=message):
        OpticalSARModel().infer([str(pair[0]), str(incompatible)], "Analyze")


def test_missing_required_band_is_rejected(pair, tmp_path) -> None:
    incomplete = write_raster(tmp_path / "four-band.tif", np.ones((4, 2, 2)))

    with pytest.raises(ValueError, match="Expected one 5-band"):
        OpticalSARModel().infer([str(incomplete), str(pair[1])], "Analyze")


def test_declared_wrong_band_order_is_rejected(pair, tmp_path) -> None:
    wrong = write_raster(
        tmp_path / "wrong-optical.tif",
        np.ones((5, 2, 2)),
        descriptions=("B02", "B03", "B08", "B04", "dataMask"),
    )

    with pytest.raises(ValueError, match="band descriptions"):
        OpticalSARModel().infer([str(wrong), str(pair[1])], "Analyze")


def test_nonfinite_and_masked_pixels_are_excluded(tmp_path) -> None:
    optical = np.ones((5, 2, 2), dtype="float32")
    optical[2, 0, 0] = np.nan
    optical[4, 0, 1] = 0
    sar = np.ones((3, 2, 2), dtype="float32")
    sar[0, 1, 0] = np.inf
    result = OpticalSARModel().infer(
        [
            str(write_raster(tmp_path / "optical.tif", optical)),
            str(write_raster(tmp_path / "sar.tif", sar)),
        ],
        "Analyze",
    )

    optical_evidence, sar_evidence, joint = result["evidence"]
    assert optical_evidence["valid_pixels"] == 2
    assert sar_evidence["valid_pixels"] == 3
    assert joint["valid_pixels"] == 1
    assert all(
        np.isfinite(value)
        for evidence in (optical_evidence["indices"]["ndvi"], sar_evidence["vv_db"])
        for value in evidence.values()
        if isinstance(value, float)
    )


def test_zero_optical_denominators_are_rejected(pair, tmp_path) -> None:
    optical = np.ones((5, 2, 2), dtype="float32")
    optical[1] = -1
    optical[2] = -1
    with pytest.raises(ValueError, match="no valid NDVI/NDWI denominators"):
        OpticalSARModel().infer(
            [str(write_raster(tmp_path / "zero-denominator.tif", optical)), str(pair[1])],
            "Analyze",
        )


def test_nonpositive_sar_samples_are_rejected(pair, tmp_path) -> None:
    sar = np.ones((3, 2, 2), dtype="float32")
    sar[:2] = 0
    with pytest.raises(ValueError, match="no valid positive VV/VH samples"):
        OpticalSARModel().infer(
            [str(pair[0]), str(write_raster(tmp_path / "nonpositive-sar.tif", sar))],
            "Analyze",
        )


def test_pair_without_covalid_pixels_is_rejected(tmp_path) -> None:
    optical = np.ones((5, 2, 2), dtype="float32")
    optical[4, :, 1] = 0
    sar = np.ones((3, 2, 2), dtype="float32")
    sar[2, :, 0] = 0
    with pytest.raises(ValueError, match="no co-valid finite pixels"):
        OpticalSARModel().infer(
            [
                str(write_raster(tmp_path / "optical.tif", optical)),
                str(write_raster(tmp_path / "sar.tif", sar)),
            ],
            "Analyze",
        )


def test_one_modality_is_rejected(pair) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        OpticalSARModel().infer([str(pair[0])], "Analyze")
