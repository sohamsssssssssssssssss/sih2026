"""Deterministic optical-SAR baseline adapter (no weights, CPU)."""

from __future__ import annotations

import time
from typing import Any

from core.schema import Evidence, Query, Scene, ToolResult
from core.tools.base import BaseTool, json_safe, materialize_paths
from core.registry import ToolNeeds


class OpticalSARTool(BaseTool):
    name = "optical-sar-deterministic"
    version = "sentinel2-indices__sentinel1-backscatter-v1"
    needs = ToolNeeds(
        sensors=("S2", "S1"),
        bands=("B02", "B03", "B04", "B08", "dataMask", "VV", "VH"),
    )

    def run(self, scene: Scene, query: Query, pixels: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            with materialize_paths(pixels, count=2, suffix=".tif") as paths:
                from orchestrator.registry import get

                result = get(self.name).infer(paths, query.text)
            evidence = json_safe(result.get("evidence", []))
            types = [item.get("type") for item in evidence if isinstance(item, dict)]
            return self._result(
                started,
                {"answer": str(result.get("answer", "")), "evidence": evidence},
                evidence=(Evidence(kind="text", summary={"types": types}),),
            )
        except Exception as error:  # noqa: BLE001
            return self.error_result(error, started)
