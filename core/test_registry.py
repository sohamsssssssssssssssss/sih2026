import json
from pathlib import Path

import pytest

from core.registry import (
    CAPABILITY_TOOLS,
    InvalidToolStatus,
    ToolNeeds,
    ToolRegistrationError,
    ToolRegistry,
    UnknownTool,
    capability_state,
    status_for,
)
from core.schema import Query, Scene, ToolResult, ToolStatus


class FakeTool:
    def __init__(self, name: str = "fake", version: str = "1.0") -> None:
        self.name = name
        self.version = version
        self.needs = ToolNeeds(sensors=("S2",))
        self.status = None

    def run(self, scene: Scene, query: Query, pixels: object) -> ToolResult:
        return ToolResult(
            tool=self.name, tool_version=self.version, value={"ok": True}, latency_ms=0
        )


def write_status(path: Path, tools: dict) -> None:
    path.write_text(json.dumps({"schema_version": 1, "tools": tools}), encoding="utf-8")


def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(
        status_path=tmp_path / "tool_status.json", evidence_root=tmp_path
    )


def test_register_get_list_and_names(tmp_path: Path) -> None:
    reg = registry(tmp_path)
    tool = reg.register(FakeTool("b-tool"))
    reg.register(FakeTool("a-tool"))

    assert reg.get("b-tool") is tool
    assert [item.name for item in reg.list()] == ["a-tool", "b-tool"]
    assert reg.names() == ("a-tool", "b-tool")
    assert isinstance(tool.status, ToolStatus)
    assert tool.status.status == "experimental_unvalidated"
    assert tool.status.evidence_refs == ()


def test_unknown_tool_is_an_error(tmp_path: Path) -> None:
    reg = registry(tmp_path)
    with pytest.raises(UnknownTool, match="Registered tools: none"):
        reg.get("missing")


def test_duplicate_name_is_rejected(tmp_path: Path) -> None:
    reg = registry(tmp_path)
    reg.register(FakeTool("dup"))
    with pytest.raises(ToolRegistrationError, match="already registered"):
        reg.register(FakeTool("dup"))


def test_register_rejects_invalid_tools(tmp_path: Path) -> None:
    reg = registry(tmp_path)

    class NoRun:
        name = "no-run"
        version = "1"
        needs = ToolNeeds()

    class BadNeeds:
        name = "bad-needs"
        version = "1"
        needs = {}

        def run(self, scene, query, pixels):
            raise AssertionError

    class EmptyName(NoRun):
        name = ""

    for tool, message in (
        (NoRun(), "run"),
        (BadNeeds(), "needs"),
        (EmptyName(), "name"),
    ):
        with pytest.raises(ToolRegistrationError, match=message):
            reg.register(tool)  # type: ignore[arg-type]


def test_a_tool_cannot_forge_validated_status(tmp_path: Path) -> None:
    reg = registry(tmp_path)
    tool = FakeTool()
    tool.status = ToolStatus(name="fake", version="1.0", status="validated")

    reg.register(tool)

    assert tool.status.status == "experimental_unvalidated"


def test_validated_without_evidence_downgrades(tmp_path: Path) -> None:
    write_status(
        tmp_path / "tool_status.json",
        {"fake": {"status": "validated", "evidence_refs": []}},
    )

    assert reg_status(tmp_path) == "experimental_unvalidated"


def test_validated_with_missing_results_file_downgrades(tmp_path: Path) -> None:
    write_status(
        tmp_path / "tool_status.json",
        {
            "fake": {
                "status": "validated",
                "evidence_refs": [{"commit": "a" * 40, "results": "results/nope.json"}],
            }
        },
    )

    assert reg_status(tmp_path) == "experimental_unvalidated"


def test_validated_with_evidence_is_honoured(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text("{}", encoding="utf-8")
    write_status(
        tmp_path / "tool_status.json",
        {
            "fake": {
                "status": "validated",
                "evidence_refs": [{"commit": "b" * 40, "results": "results/r.json"}],
            }
        },
    )

    reg = registry(tmp_path)
    tool = reg.register(FakeTool())
    assert tool.status.status == "validated"
    assert tool.status.evidence_refs == (f"{'b' * 40} results/r.json",)


def test_disabled_is_honoured(tmp_path: Path) -> None:
    write_status(
        tmp_path / "tool_status.json",
        {"fake": {"status": "disabled", "evidence_refs": []}},
    )

    assert reg_status(tmp_path) == "disabled"


def reg_status(tmp_path: Path) -> str:
    reg = registry(tmp_path)
    return reg.register(FakeTool()).status.status


def test_missing_status_file_defaults_to_experimental(tmp_path: Path) -> None:
    status = status_for("fake", "1", path=tmp_path / "absent.json", evidence_root=tmp_path)
    assert status.status == "experimental_unvalidated"
    assert status.evidence_refs == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": {}},
        {"schema_version": 2, "tools": {}},
        {"schema_version": 1, "tools": []},
        {"schema_version": 1, "tools": {"fake": {"status": "trusted"}}},
        {"schema_version": 1, "tools": {"fake": {"status": "validated", "evidence_refs": [{"commit": "short", "results": "r"}]}}},
        {"schema_version": 1, "tools": {"fake": {"status": "validated", "evidence_refs": ["results/r.json"]}}},
    ],
)
def test_malformed_status_files_fail_closed(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "tool_status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidToolStatus):
        status_for("fake", "1", path=path, evidence_root=tmp_path)


def test_capability_mapping_validated_and_experimental(tmp_path: Path) -> None:
    write_status(
        tmp_path / "tool_status.json",
        {"qwen2.5vl-3b": {"status": "experimental_unvalidated", "evidence_refs": []}},
    )
    reg = registry(tmp_path)
    reg.register(FakeTool("qwen2.5vl-3b"))
    state = capability_state("single_image_vqa", registry=reg)

    assert state["state"] == "AVAILABLE"
    assert state["available"] is True
    assert state["label"] == "experimental"


def test_capability_mapping_disabled_and_unregistered(tmp_path: Path) -> None:
    write_status(
        tmp_path / "tool_status.json",
        {"qwen2.5vl-3b": {"status": "disabled", "evidence_refs": []}},
    )
    reg = registry(tmp_path)
    reg.register(FakeTool("qwen2.5vl-3b"))

    disabled = capability_state("single_image_vqa", registry=reg)
    assert disabled == {
        "capability": "single_image_vqa",
        "tool": "qwen2.5vl-3b",
        "registered": True,
        "state": "UNAVAILABLE",
        "available": False,
        "label": None,
    }
    assert capability_state("grounding", registry=reg)["state"] == "NOT_IMPLEMENTED"
    assert capability_state("not_a_capability", registry=reg)["state"] == "NOT_IMPLEMENTED"


def test_default_tools_have_expected_status() -> None:
    import core.registry as module

    module.reset_registry()
    assert set(module.names()) == set(CAPABILITY_TOOLS.values())
    assert {t.name: t.status.status for t in module.list_tools()} == {
        name: "disabled" if name == "change-deterministic" else "experimental_unvalidated"
        for name in CAPABILITY_TOOLS.values()
    }
