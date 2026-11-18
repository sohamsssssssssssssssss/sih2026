import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from core.schema import Query, Scene, ToolResult
from core.tools import TOOLS

SCENE = Scene(
    scene_id="0" * 64,
    sensor="S2",
    bands=("B02", "B03", "B04", "B08"),
    units="reflectance_x1e4",
    gsd_m=10.0,
    crs="EPSG:32643",
    bounds=(0.0, 0.0, 80.0, 80.0),
    nodata_pct=0.0,
    source="synthetic",
)
TRANSFORM = from_origin(500000.0, 2000000.0, 10.0, 10.0)


def tool(name: str):
    return next(item for item in TOOLS if item.name == name)


def skip_unless_ready(name: str) -> None:
    from orchestrator.registry import get

    readiness = get(name).readiness()
    if not readiness.available:
        pytest.skip(
            f"{name}: {readiness.reason_code or 'UNAVAILABLE'}: "
            f"{readiness.detail or 'no local weights/runtime'}"
        )


def write_raster(path: Path, arrays: list[np.ndarray], descriptions: tuple[str, ...]) -> None:
    height, width = arrays[0].shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=len(arrays),
        dtype="float32",
        crs="EPSG:32643",
        transform=TRANSFORM,
    ) as dataset:
        for index, array in enumerate(arrays, 1):
            dataset.write(array.astype("float32"), index)
            dataset.set_band_description(index, descriptions[index - 1])


def assert_contract(result: ToolResult) -> None:
    assert isinstance(result, ToolResult)
    assert result.latency_ms >= 0
    assert result.tool_version
    json.loads(result.model_dump_json())


def test_qwen_adapter_contract(tmp_path: Path) -> None:
    skip_unless_ready("qwen2.5vl-3b")
    image = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    result = tool("qwen2.5vl-3b").run(
        SCENE, Query(text="What is visible?", qtype="landcover"), (str(image),)
    )

    assert_contract(result)
    assert "answer" in result.value or "error" in result.value


def test_grounding_adapter_contract(tmp_path: Path) -> None:
    skip_unless_ready("grounding-dino-swint")
    image = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    result = tool("grounding-dino-swint").run(
        SCENE, Query(text="building", qtype="locate"), (str(image),)
    )

    assert_contract(result)
    assert isinstance(result.value.get("boxes"), list) or "error" in result.value


def test_optical_sar_adapter_contract(tmp_path: Path) -> None:
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"
    ones = np.ones((8, 8), dtype="float32")
    write_raster(
        optical,
        [
            np.full((8, 8), 1000.0),
            np.full((8, 8), 1200.0),
            np.full((8, 8), 800.0),
            np.full((8, 8), 3000.0),
            ones,
        ],
        ("B02", "B03", "B04", "B08", "dataMask"),
    )
    write_raster(
        sar,
        [np.full((8, 8), 0.1), np.full((8, 8), 0.02), ones],
        ("VV", "VH", "dataMask"),
    )

    result = tool("optical-sar-deterministic").run(
        SCENE, Query(text="Compare optical and SAR.", qtype="context"),
        (str(optical), str(sar)),
    )

    assert_contract(result)
    assert result.value.get("error") is None, result.value
    assert result.value["answer"]


def test_change_adapter_contract(tmp_path: Path) -> None:
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    base = np.arange(64, dtype="float32").reshape(8, 8)
    write_raster(
        before,
        [base, base + 100.0, base + 200.0],
        ("red", "green", "blue"),
    )
    write_raster(
        after,
        [base + 400.0, base + 100.0, base + 200.0],
        ("red", "green", "blue"),
    )

    result = tool("change-deterministic").run(
        SCENE, Query(text="Did the scene change?", qtype="change"),
        (str(before), str(after)),
    )

    assert_contract(result)
    assert result.value.get("error") is None, result.value
    assert result.value["answer"]


def test_sar_flood_adapter_contract(tmp_path: Path) -> None:
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    mask = np.ones((40, 40), dtype="float32")

    def sar(vv: np.ndarray) -> list[np.ndarray]:
        return [vv, vv / 5.0, mask]

    land = np.full((40, 40), 0.1, dtype="float32")
    water_before = land.copy()
    water_before[:5] = 0.005
    water_after = land.copy()
    water_after[:15] = 0.005
    write_raster(before, sar(water_before), ("VV", "VH", "dataMask"))
    write_raster(after, sar(water_after), ("VV", "VH", "dataMask"))

    result = tool("change-deterministic").run(
        SCENE, Query(text="Did flooding expand?", qtype="flood"),
        (str(before), str(after)),
    )

    assert_contract(result)
    assert "answer" in result.value or "error" in result.value
