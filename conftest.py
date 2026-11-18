"""Isolate project tests from the application's persisted audit chain."""

import pytest

from models.base import ModelReadiness
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel
from orchestrator import trace


@pytest.fixture(autouse=True)
def isolated_execution_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(trace, "_TRACE", [])
    monkeypatch.setattr(trace, "_LOADED_PATH", None)


@pytest.fixture
def ready_providers(monkeypatch):
    """Tests that exercise routing provide explicit ready model doubles."""
    monkeypatch.setattr(QwenVLModel, "readiness", lambda _: ModelReadiness(True))
    monkeypatch.setattr(GroundingDINOModel, "readiness", lambda _: ModelReadiness(True))
    monkeypatch.setattr(OpticalSARModel, "readiness", lambda _: ModelReadiness(True))
    monkeypatch.setattr(ChangeModel, "readiness", lambda _: ModelReadiness(True))


# These optional package assets are not tracked; synthetic-only tests still run.
_PACKAGE = "packages/stage1-t4-smoke"
_FIXTURE_INPUTS = {
    ("scripts/test_stage1_resume_variant.py", "tiny_run"): ("config-last4-vision-fp32.yaml",),
    ("scripts/test_stage1_revision_gate.py", "proof"): ("config.yaml",),
    ("scripts/test_stage1_t4_package.py", "package_dir"): (
        "config.yaml", "config-vision-fp32.yaml", "config-last4.yaml",
        "config-last4-vision-fp32.yaml", "captions.json", "revision.json", "norm_contract.json",
    ),
}
_TEST_INPUTS = {
    ("scripts/test_stage1_full.py", "test_config_full_yaml_has_no_placeholder_steps"): ("config-full.yaml",),
    ("scripts/test_stage1_smoke.py", "test_t4_smoke_package"): (
        "config.yaml", "config-vision-fp32.yaml", "captions.json", "norm_contract.json", "patches.tar.gz",
    ),
    ("scripts/test_stage1_smoke.py", "test_t4_package_entry_and_stages"): (
        "revision.json", "config.yaml", "config-vision-fp32.yaml", "run.py",
        "src/scripts/run_stage1_t4.py", "README.md",
    ),
    ("scripts/test_stage1_t4_package.py", "test_package_manifest_integrity"): ("package-manifest.json",),
    ("scripts/test_stage1_t4_package.py", "test_updated_package_runner_matches_source"): (
        "src/scripts/run_stage1_smoke.py", "src/scripts/run_stage1_t4.py",
    ),
}


def pytest_collection_modifyitems(config, items):
    from importlib.util import find_spec

    for item in items:
        module = item.path.relative_to(config.rootpath).as_posix()
        name = item.name.split("[", 1)[0]
        required = list(_TEST_INPUTS.get((module, name), ()))
        for fixture in getattr(item, "fixturenames", ()):
            required.extend(_FIXTURE_INPUTS.get((module, fixture), ()))
        for filename in required:
            relative = f"{_PACKAGE}/{filename}"
            if not (config.rootpath / relative).is_file():
                item.add_marker(pytest.mark.skip(reason=f"fixture not in repo: {relative}"))
                break
        if (module == "core/tools/test_adapters.py" and name == "test_grounding_adapter_contract"
                and find_spec("groundingdino") is None):
            item.add_marker(pytest.mark.skip(reason="dependency not installed: groundingdino"))
