"""One registry for every runnable tool, speaking the ``core.schema`` contracts.

Tool identity is ``name`` + ``version``; capability maturity comes from exactly one
file, ``core/tool_status.json``. Registration never invents ``validated``: a
``validated`` entry without a commit + existing results file is downgraded to
``experimental_unvalidated``, and a tool absent from the file is
``experimental_unvalidated`` as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.router import CHANGE_UNSUPPORTED, VQA_MLX
from core.schema import Query, Scene, ToolResult, ToolStatus

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = Path(__file__).resolve().with_name("tool_status.json")
STATUS_SCHEMA_VERSION = 1
STATUSES = ("validated", "experimental_unvalidated", "disabled")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# Capability vocabulary -> the tool that serves it. Kept here so the backend has
# one mapping function instead of its own table.
CAPABILITY_TOOLS: dict[str, str] = {
    "landcover": "landcover",
    "flood": "flood-otsu",
    "single_image_vqa": "qwen2.5vl-3b",
    "grounding": "grounding-dino-swint",
    "change_vqa": "change-deterministic",
    "optical_sar": "optical-sar-deterministic",
}


class UnknownTool(KeyError):
    """The requested tool has not been registered."""


class ToolRegistrationError(ValueError):
    """A tool does not satisfy the protocol or collides with a registered name."""


class InvalidToolStatus(ValueError):
    """``core/tool_status.json`` is unreadable or malformed."""


@dataclass(frozen=True)
class ToolNeeds:
    """Declared input needs; callers use this to route, not to validate pixels."""

    sensors: tuple[str, ...] = ()
    bands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("sensors", "bands"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"ToolNeeds.{field} must be a tuple of non-empty strings")


@runtime_checkable
class Tool(Protocol):
    """The only interface orchestration code needs from a runnable tool."""

    name: str
    version: str
    needs: ToolNeeds
    status: ToolStatus

    def run(self, scene: Scene, query: Query, pixels: Any) -> ToolResult:
        """Execute on already-loaded pixels; the pixels are never part of the schema."""
        ...


def _read_entries(status_path: Path) -> dict[str, dict[str, Any]]:
    try:
        text = status_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise InvalidToolStatus(f"Tool status file could not be read: {status_path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidToolStatus("Tool status file is not valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise InvalidToolStatus(
            f"Tool status file must be an object with schema_version {STATUS_SCHEMA_VERSION}"
        )
    tools = data.get("tools", {})
    if not isinstance(tools, dict):
        raise InvalidToolStatus("Tool status file 'tools' must be an object")
    entries: dict[str, dict[str, Any]] = {}
    for name, entry in tools.items():
        if not isinstance(name, str) or not name:
            raise InvalidToolStatus("Tool status keys must be non-empty strings")
        if not isinstance(entry, dict):
            raise InvalidToolStatus(f"Tool status entry for {name!r} must be an object")
        status = entry.get("status", "experimental_unvalidated")
        if status not in STATUSES:
            raise InvalidToolStatus(f"Unknown status {status!r} for tool {name!r}")
        refs = entry.get("evidence_refs", [])
        if not isinstance(refs, list):
            raise InvalidToolStatus(f"evidence_refs for {name!r} must be a list")
        clean_refs: list[dict[str, str]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                raise InvalidToolStatus(f"Evidence refs for {name!r} must be objects")
            commit, results = ref.get("commit"), ref.get("results")
            if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
                raise InvalidToolStatus(f"Evidence commit for {name!r} must be a 40-char hex SHA")
            if not isinstance(results, str) or not results:
                raise InvalidToolStatus(f"Evidence results path for {name!r} must be text")
            clean_refs.append({"commit": commit, "results": results})
        entries[name] = {"status": status, "refs": clean_refs}
    return entries


def status_for(
    name: str,
    version: str,
    *,
    path: str | Path | None = None,
    evidence_root: str | Path | None = None,
) -> ToolStatus:
    """Resolve one tool's maturity from the single status file.

    A ``validated`` claim is honoured only when at least one evidence ref has a
    commit SHA and a results file that exists under ``evidence_root``; otherwise it
    is downgraded. Missing file or missing entry means ``experimental_unvalidated``.
    """
    status_path = Path(path) if path is not None else STATUS_PATH
    root = Path(evidence_root) if evidence_root is not None else ROOT
    entry = _read_entries(status_path).get(name)
    if entry is None:
        return ToolStatus(
            name=name, version=version, status="experimental_unvalidated", evidence_refs=()
        )
    status = str(entry["status"])
    refs: tuple[str, ...] = ()
    evidence_ok = False
    for ref in entry["refs"]:
        results = Path(ref["results"])
        if results.is_absolute() or ".." in results.parts:
            continue
        if not (root / results).is_file():
            continue
        refs += (f"{ref['commit']} {ref['results']}",)
        evidence_ok = True
    if status == "validated" and not evidence_ok:
        status = "experimental_unvalidated"
    return ToolStatus(
        name=name, version=version, status=status, evidence_refs=refs  # type: ignore[arg-type]
    )


class ToolRegistry:
    """A name-keyed, duplicate-rejecting registry with file-backed statuses."""

    def __init__(
        self,
        status_path: str | Path | None = None,
        evidence_root: str | Path | None = None,
    ) -> None:
        self._status_path = Path(status_path) if status_path is not None else STATUS_PATH
        self._evidence_root = Path(evidence_root) if evidence_root is not None else ROOT
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        name = getattr(tool, "name", None)
        version = getattr(tool, "version", None)
        needs = getattr(tool, "needs", None)
        if not isinstance(name, str) or not name:
            raise ToolRegistrationError("Tool.name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ToolRegistrationError("Tool.version must be a non-empty string")
        if not isinstance(needs, ToolNeeds):
            raise ToolRegistrationError("Tool.needs must be a ToolNeeds")
        if not callable(getattr(tool, "run", None)):
            raise ToolRegistrationError("Tool.run must be callable")
        if name in self._tools:
            raise ToolRegistrationError(f"Tool already registered: {name}")
        resolved = status_for(
            name, version, path=self._status_path, evidence_root=self._evidence_root
        )
        try:
            tool.status = resolved  # status is owned by the file, never by the tool
        except (AttributeError, TypeError) as exc:
            raise ToolRegistrationError("Tool.status must be assignable") from exc
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "none"
            raise UnknownTool(
                f"Unknown tool '{name}'. Registered tools: {available}"
            ) from None

    def list(self) -> tuple[Tool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def reset(self) -> None:
        self._tools.clear()


_DEFAULT = ToolRegistry()
_DEFAULTS_LOADED = False


def register_default_tools() -> None:
    """Register the production adapters once; safe to call repeatedly."""
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    from core.tools import TOOLS

    for tool in TOOLS:
        if tool.name not in _DEFAULT.names():
            _DEFAULT.register(tool)
    _DEFAULTS_LOADED = True


def _ensure_defaults() -> None:
    if not _DEFAULTS_LOADED:
        register_default_tools()


def register(tool: Tool) -> Tool:
    return _DEFAULT.register(tool)


def get(name: str) -> Tool:
    _ensure_defaults()
    return _DEFAULT.get(name)


def list_tools() -> tuple[Tool, ...]:
    _ensure_defaults()
    return _DEFAULT.list()


def names() -> tuple[str, ...]:
    _ensure_defaults()
    return _DEFAULT.names()


def reset_registry() -> None:
    """Test hook: forget every registration and the defaults-loaded flag."""
    global _DEFAULTS_LOADED
    _DEFAULT.reset()
    _DEFAULTS_LOADED = False


def capability_state(capability: str, *, registry: ToolRegistry | None = None) -> dict[str, Any]:
    """Map a capability to tool maturity using the one required mapping.

    validated -> AVAILABLE; experimental_unvalidated -> AVAILABLE with label
    "experimental"; disabled -> UNAVAILABLE; not registered -> NOT_IMPLEMENTED.
    """
    active = registry
    if active is None:
        _ensure_defaults()
        active = _DEFAULT
    tool_name = CAPABILITY_TOOLS.get(capability)
    if capability == "single_image_vqa" and VQA_MLX in active.names():
        tool_name = VQA_MLX
    if capability == "change_vqa":
        return {
            "capability": capability, "tool": tool_name,
            "registered": tool_name in active.names(), "state": "UNAVAILABLE",
            "available": False, "label": None, "reason": CHANGE_UNSUPPORTED,
        }
    if tool_name is None:
        return {
            "capability": capability,
            "tool": None,
            "registered": False,
            "state": "NOT_IMPLEMENTED",
            "available": False,
            "label": None,
        }
    try:
        tool = active.get(tool_name)
    except UnknownTool:
        return {
            "capability": capability,
            "tool": tool_name,
            "registered": False,
            "state": "NOT_IMPLEMENTED",
            "available": False,
            "label": None,
        }
    maturity = tool.status.status
    if tool_name == VQA_MLX and maturity != "disabled" and (reason := tool.unavailable_reason):
        return {
            "capability": capability, "tool": VQA_MLX, "registered": True,
            "state": "UNAVAILABLE", "available": False, "label": "experimental", "reason": reason,
        }
    if maturity == "validated":
        state, label, available = "AVAILABLE", None, True
    elif maturity == "experimental_unvalidated":
        state, label, available = "AVAILABLE", "experimental", True
    else:
        state, label, available = "UNAVAILABLE", None, False
    return {
        "capability": capability,
        "tool": tool_name,
        "registered": True,
        "state": state,
        "available": available,
        "label": label,
    }


def capability_table(*, registry: ToolRegistry | None = None) -> list[dict[str, Any]]:
    """Every known capability with its maturity mapping, in stable order."""
    return [capability_state(name, registry=registry) for name in CAPABILITY_TOOLS]
