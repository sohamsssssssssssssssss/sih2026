"""Isolate project tests from the application's persisted audit chain."""

import pytest

from models.base import ModelReadiness
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel
from orchestrator import trace

# Frozen Kaggle bundles carry their own tests and sys.path root; run them from inside the package.
collect_ignore_glob = ["packages/*"]


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
