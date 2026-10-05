from pathlib import Path

import models.paths
from models.paths import REPO_ROOT, public_path


def test_path_under_repo_root_is_repo_relative() -> None:
    path = REPO_ROOT / "data" / "runtime" / "rasters" / "scene_x.tif"

    assert public_path(path) == "data/runtime/rasters/scene_x.tif"
    assert public_path(str(path)) == "data/runtime/rasters/scene_x.tif"


def test_path_outside_repo_root_is_reduced_to_file_name(tmp_path: Path) -> None:
    assert public_path(tmp_path / "secret" / "scene.tif") == "scene.tif"
    assert public_path("/kaggle/working/run/data/tile.png") == "tile.png"
    assert public_path(r"C:\Users\someone\tile.png") == "tile.png"


def test_traversal_out_of_repo_root_is_not_treated_as_relative() -> None:
    escaped = f"{REPO_ROOT}/data/../../outside/scene.tif"

    assert public_path(escaped) == "scene.tif"


def test_symlinked_root_still_maps_to_relative_path(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real"
    (real / "data").mkdir(parents=True)
    target = real / "data" / "scene.tif"
    target.write_bytes(b"")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(models.paths, "REPO_ROOT", link)

    assert public_path(target) == "data/scene.tif"
    assert public_path(link / "data" / "scene.tif") == "data/scene.tif"


def test_relative_paths_are_kept_only_when_they_stay_inside_the_repo() -> None:
    assert public_path("data/runtime/rasters/scene_x.tif") == "data/runtime/rasters/scene_x.tif"
    assert public_path("./data/./scene.tif") == "data/scene.tif"
    assert public_path("../outside/scene.tif") == "scene.tif"
    assert public_path("data/../../outside/scene.tif") == "scene.tif"
