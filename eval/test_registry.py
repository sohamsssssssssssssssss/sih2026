import json

import pytest

from eval.registry import REQUIRED_FIELDS, write_record


def _record(**overrides):
    record = {field: None for field in REQUIRED_FIELDS}
    record.update(experiment_id="SQ-20260926-001", title="t", status="PASSED", metrics={})
    record.update(overrides)
    return record


def test_writes_record_with_captured_repository_and_environment(tmp_path):
    path = write_record(_record(), registry_dir=tmp_path)

    written = json.loads(path.read_text())
    assert written["experiment_id"] == "SQ-20260926-001"
    assert written["repository"]["url"].endswith("/sih2026.git")
    assert len(written["repository"]["sha"]) == 40
    assert "python" in written["environment"]


def test_refuses_to_overwrite_existing_record(tmp_path):
    write_record(_record(), registry_dir=tmp_path)

    with pytest.raises(FileExistsError):
        write_record(_record(status="FAILED"), registry_dir=tmp_path)
    assert json.loads((tmp_path / "SQ-20260926-001.json").read_text())["status"] == "PASSED"


@pytest.mark.parametrize(
    "overrides",
    [{"status": "GREAT"}, {"experiment_id": "SQ-2026-1"}],
)
def test_rejects_invalid_status_or_id(tmp_path, overrides):
    with pytest.raises(ValueError):
        write_record(_record(**overrides), registry_dir=tmp_path)


def test_rejects_missing_required_field(tmp_path):
    record = _record()
    del record["seed"]

    with pytest.raises(ValueError, match="seed"):
        write_record(record, registry_dir=tmp_path)


def test_caller_supplied_environment_is_preserved(tmp_path):
    remote = {"python": "3.11.13", "gpu": "Tesla T4"}

    path = write_record(_record(environment=remote), registry_dir=tmp_path)

    assert json.loads(path.read_text())["environment"] == remote
