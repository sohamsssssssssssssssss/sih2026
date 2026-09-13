from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.scene_pack import resolution_asset, scene_asset, scene_catalog


def test_scene_pack_serves_verified_assets() -> None:
    scenes = {scene["id"]: scene for scene in scene_catalog()["scenes"]}

    assert scenes["loveda-golden-vqa"]["available"] is True
    assert scene_asset("loveda-golden-vqa").is_file()
    assert scenes["dior-rsvg-07272"]["available"] is True
    assert scene_asset("dior-rsvg-07272").is_file()

    response = TestClient(app).get("/api/scenes")
    assert response.status_code == 200
    assert response.json()["scenes"][1]["unavailable_reason"] is None


def test_resolution_rung_images_are_verified_before_serving() -> None:
    client = TestClient(app)

    available = client.get("/api/resolution/rungs/0.3/image")
    assert available.status_code == 200
    with Image.open(BytesIO(available.content)) as image:
        assert image.size == (1024, 1024)

    expected_sizes = {1.0: (307, 307), 2.0: (154, 154), 5.0: (61, 61), 10.0: (31, 31)}
    for gsd, expected_size in expected_sizes.items():
        assert resolution_asset(gsd).is_file()
        response = client.get(f"/api/resolution/rungs/{gsd}/image")
        assert response.status_code == 200
        with Image.open(BytesIO(response.content)) as image:
            assert image.size == expected_size
