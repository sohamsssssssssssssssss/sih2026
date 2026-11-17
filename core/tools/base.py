"""Shared adapter machinery. Wrappers stay thin; model code is never rewritten."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from core.registry import ToolNeeds, status_for
from core.schema import Evidence, ToolResult

Pixels = (
    bytes
    | bytearray
    | str
    | Path
    | Sequence[str | Path | bytes | bytearray]
)


def json_safe(value: Any) -> Any:
    """Convert numpy/array output to plain JSON values, or fail loudly."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tool output contains a non-finite float")
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"{type(value).__name__} cannot be represented as JSON")


@contextmanager
def materialize_paths(
    pixels: Pixels, *, count: int | None = None, suffix: str = ".png"
) -> Iterator[list[str]]:
    """Yield local paths for pixels; bytes are written to a temporary directory.

    Tools receive pixels separately from the contract, so bytes are the only
    transport-neutral form. Existing paths are passed through untouched.
    """
    if isinstance(pixels, (str, Path, bytes, bytearray)):
        items: list[Any] = [pixels]
    else:
        items = list(pixels)
    if count is not None and len(items) != count:
        raise ValueError(f"expected {count} pixel inputs, got {len(items)}")
    directory: str | None = None
    paths: list[str] = []
    try:
        for index, item in enumerate(items):
            if isinstance(item, (bytes, bytearray)):
                if directory is None:
                    directory = tempfile.mkdtemp(prefix="satquery-tool-")
                path = Path(directory) / f"pixels_{index}{suffix}"
                path.write_bytes(bytes(item))
                paths.append(str(path))
            else:
                paths.append(str(Path(item)))
        yield paths
    finally:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


class BaseTool:
    """Common adapter behavior: status from the file, latency, JSON-safe results."""

    name = "base"
    version = "0.0.0"
    needs = ToolNeeds()

    def __init__(self) -> None:
        self.status = status_for(self.name, self.version)

    def _result(
        self,
        started: float,
        value: Any,
        *,
        evidence: tuple[Evidence, ...] = (),
        raw_confidence: float | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool=self.name,
            tool_version=self.version,
            value=json_safe(value),
            raw_confidence=raw_confidence,
            evidence=evidence,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000.0)),
        )

    def error_result(self, error: BaseException, started: float) -> ToolResult:
        """Missing weights or failed inference become data, never an import crash."""
        message = " ".join(str(error).split())[:300] or type(error).__name__
        code = (
            "weights_missing"
            if isinstance(error, FileNotFoundError)
            else "runtime_unavailable"
            if isinstance(error, (RuntimeError, OSError))
            else "inference_failed"
        )
        return self._result(
            started,
            {"error": message, "error_code": code, "error_type": type(error).__name__},
            evidence=(Evidence(kind="text", summary={"error": message, "error_code": code, "error_type": type(error).__name__}),),
        )
