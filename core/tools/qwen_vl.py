"""Qwen2.5-VL-3B adapter: one answer, no invented confidence."""

from __future__ import annotations

import time
from typing import Any

from core.schema import Query, Scene, ToolResult
from core.tools.base import BaseTool, materialize_paths
from core.registry import ToolNeeds


class QwenVLTool(BaseTool):
    name = "qwen2.5vl-3b"
    version = "Qwen/Qwen2.5-VL-3B-Instruct"
    needs = ToolNeeds(sensors=("S2", "S1", "RGB_HR", "S1S2", "UNKNOWN"))

    def run(self, scene: Scene, query: Query, pixels: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            with materialize_paths(pixels, suffix=".png") as paths:
                from orchestrator.registry import get

                result = get(self.name).infer(paths, query.text)
            answer = str(result.get("answer", ""))
            confidence = result.get("confidence")
            raw = (
                float(confidence)
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                else None
            )
            return self._result(started, {"answer": answer}, raw_confidence=raw)
        except Exception as error:  # noqa: BLE001 - adapters report failures as data
            return self.error_result(error, started)
