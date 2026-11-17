"""Deterministic bi-temporal change adapter, including the SAR-water branch."""

from __future__ import annotations

import time
from typing import Any

from core.schema import Evidence, Query, Scene, ToolResult
from core.tools.base import BaseTool, json_safe, materialize_paths
from core.registry import ToolNeeds


class ChangeTool(BaseTool):
    name = "change-deterministic"
    version = "bitemporal-difference-v2"
    needs = ToolNeeds(
        sensors=("S2", "S1"),
        bands=("B02", "B03", "B04", "B08", "VV", "VH", "dataMask"),
    )

    def run(self, scene: Scene, query: Query, pixels: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            with materialize_paths(pixels, count=2, suffix=".tif") as paths:
                from orchestrator.registry import get

                result = get(self.name).infer(paths, query.text)
            evidence = json_safe(result.get("evidence", []))
            types = [item.get("type") for item in evidence if isinstance(item, dict)]
            kind = "diff" if any("change" in str(item) for item in types) else "text"
            return self._result(
                started,
                {"answer": str(result.get("answer", "")), "evidence": evidence},
                evidence=(Evidence(kind=kind, summary={"types": types}),),
            )
        except Exception as error:  # noqa: BLE001
            return self.error_result(error, started)
