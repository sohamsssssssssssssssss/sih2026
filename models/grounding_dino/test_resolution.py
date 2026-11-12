import json
import pytest
from models.grounding_dino.resolution import filename_gsd, metadata_gsd, resolution_gate, refusal


@pytest.mark.parametrize("gsd,status", [(10, "not_resolvable"), (0.3, "resolvable"), (1, "resolvable"), (None, "gsd_unknown")])
def test_building_resolution_gate(gsd, status):
    gate = resolution_gate("small buildings", gsd)
    assert gate["count_status"] == status
    assert gate["object_class"] == "building" and gate["required_gsd_m"] == 1
    if status == "not_resolvable":
        result = refusal(gate)
        assert "cannot be determined at 10 m/px" in result["answer"]
        assert "1 m/px or finer" in result["answer"]
        assert "count" not in result and result["answer_source"] == "resolution_gate"
    if gsd is None:
        assert gate["unverified_gsd"] is True


@pytest.mark.parametrize("text,expected", [("scene_gsd0.3", 0.3), ("upload_gsd10.tif", 10),
    ("x_gsd0.3.png", 0.3), ("unknown", None), ("x_gsd0", None), ("x_gsd10cm", None)])
def test_scene_id_gsd(text, expected):
    assert filename_gsd(text) == expected


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, "degrees", "-2 m/px"])
def test_invalid_metadata_gsd_is_unknown(value):
    assert metadata_gsd(value) is None


def test_class_aliases_and_config_validation(tmp_path):
    assert resolution_gate("cars", 0.3)["object_class"] == "vehicle"
    assert resolution_gate("storage tanks", 0.3)["object_class"] == "storage tank"
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"resolution": {"min_pixels": "8", "object_size_m": {"default": 3}}}))
    with pytest.raises(ValueError):
        resolution_gate("buildings", 10, config=path)
