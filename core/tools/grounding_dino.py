"""Grounding DINO adapter: normalized boxes become JSON evidence."""

from __future__ import annotations

import time
from typing import Any

from core.schema import Evidence, Query, Scene, ToolResult
from core.tools.base import BaseTool, json_safe, materialize_paths
from core.registry import ToolNeeds


class GroundingDINOTool(BaseTool):
    name = "grounding-dino-swint"
    version = "ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth"
    needs = ToolNeeds(sensors=("S2", "S1", "RGB_HR", "S1S2", "UNKNOWN"))

    def run(self, scene: Scene, query: Query, pixels: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            with materialize_paths(pixels, count=1, suffix=".png") as paths:
                from orchestrator.registry import get

                result = get(self.name).infer(paths, query.text)
            boxes = [
                item
                for item in json_safe(result.get("evidence", []))
                if isinstance(item, dict) and item.get("type") == "bounding_box"
            ]
            return self._result(
                started,
                {"answer": str(result.get("answer", "")), "boxes": boxes, "count": len(boxes)},
                evidence=(
                    Evidence(kind="boxes", summary={"boxes": boxes, "count": len(boxes)}),
                ),
            )
        except Exception as error:  # noqa: BLE001
            return self.error_result(error, started)
