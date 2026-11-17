"""Pure preflight decisions. Trace writing and tool execution live in answer.py."""
from dataclasses import dataclass
import math
from typing import Mapping

from core.registry import ToolRegistry, UnknownTool
from core.router import NON_IMAGERY, Route
from core.schema import Evidence, Query, RefusalReason, Scene

# PROVISIONAL guards, not calibrated accuracy cutoffs; ROUTER_SPEC.md records why.
CLOUD_PCT = 50.0
NODATA_PCT = 50.0
OOD_STD = 2.0
INDIA_BOUNDS = (68.0, 6.0, 98.0, 38.0)
DETERMINISTIC = {"flood-otsu", "change-deterministic", "optical-sar-deterministic"}


@dataclass(frozen=True)
class GateDecision:
    tool: str | None
    mode: str
    refusal_reason: RefusalReason | None
    evidence: tuple[Evidence, ...]

    @property
    def passed(self) -> bool:
        return self.refusal_reason is None

    @property
    def clean(self) -> bool:
        return self.passed and not any(e.summary.get("warning") for e in self.evidence)


def sensors(scene: Scene) -> tuple[str, ...]:
    return ("S1", "S2") if scene.sensor == "S1S2" else (scene.sensor,)


def evaluate(query: Query, scene: Scene, choice: Route, registry: ToolRegistry, *,
             second_scene: Scene | None = None,
             band_means: tuple[Mapping[str, float], ...] = (),
             training_stats: Mapping[str, tuple[float, float]] | None = None) -> GateDecision:
    """Inspect metadata only. Observed/training means must share units and band order."""
    events: list[Evidence] = []
    selected = choice.tools[0] if choice.tools else None
    mode = "S2" if scene.sensor == "S1S2" and selected == "change-deterministic" else scene.sensor

    def event(rule, outcome, *, needs=(), **details):
        events.append(Evidence(kind="text", needs_sensor=needs,
                               summary={"rule": rule, "outcome": outcome, **details}))

    def refuse(reason: RefusalReason, *, needs=(), **details):
        event(reason, "refused", needs=needs, **details)
        return GateDecision(selected, mode, reason, tuple(events))

    if NON_IMAGERY.search(query.text) or choice.intent == "not_answerable":
        return refuse("not_answerable")
    event("answerability", "pass")
    temporal = choice.intent == "change" or query.qtype == "change"
    if temporal and second_scene is None:
        return refuse("second_scene_missing")
    scenes = (scene, second_scene) if temporal else (scene,)
    if temporal and (scene.crs, scene.bounds, scene.bands, scene.gsd_m) != (
            second_scene.crs, second_scene.bounds, second_scene.bands, second_scene.gsd_m):
        return refuse("checks_disagree", detail="Temporal scenes need identical CRS, bounds, bands and GSD")
    event("second_date", "pass" if temporal else "not_applicable")
    for index, item in enumerate(scenes):
        event("region_threshold", "used", scene=index, threshold=list(INDIA_BOUNDS), provenance="PROVISIONAL")
        try:
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(item.crs, "EPSG:4326", *item.bounds)
            if not all(math.isfinite(v) for v in bounds):
                raise ValueError("Non-finite transformed bounds")
            outside = not (INDIA_BOUNDS[0] <= bounds[0] < bounds[2] <= INDIA_BOUNDS[2]
                           and INDIA_BOUNDS[1] <= bounds[1] < bounds[3] <= INDIA_BOUNDS[3])
            event("region", "out_of_region" if outside else "pass", scene=index,
                  warning="out_of_region" if outside else None)
        except (ValueError, RuntimeError):
            event("region", "unknown", scene=index, warning="region_unknown")
        event("nodata_threshold", "used", scene=index, threshold=NODATA_PCT,
              observed=item.nodata_pct, provenance="PROVISIONAL")
        if item.nodata_pct > NODATA_PCT:
            return refuse("nodata", scene=index)
    optical = scene.sensor != "S1" and selected != "flood-otsu"
    if optical:
        cloudy = False
        for index, item in enumerate(scenes):
            event("cloud_threshold", "used", scene=index, threshold=CLOUD_PCT,
                  observed=item.cloud_pct, provenance="PROVISIONAL")
            if item.cloud_pct is None:
                event("cloud", "unknown", scene=index, warning="cloud_unknown")
            elif item.cloud_pct > CLOUD_PCT:
                cloudy = True
        if cloudy:
            radar = {"landcover": "landcover", "flood": "flood-otsu",
                     "change": "change-deterministic"}.get(choice.intent)
            try:
                radar_tool = registry.get(radar) if radar else None
            except UnknownTool:
                radar_tool = None
            if (not all("S1" in sensors(item) for item in scenes)
                    or radar_tool is None or "S1" not in radar_tool.needs.sensors
                    or radar_tool.status.status == "disabled"):
                return refuse("cloud", needs=("S1",))
            selected, mode = radar, "S1"
            event("cloud", "reroute", warning="monsoon_mode", tool=selected, mode=mode)
    event("cloud", "pass" if optical else "not_applicable")
    try:
        tool = registry.get(selected)
    except UnknownTool:
        return refuse("tool_missing", tool=selected)
    if tool.status.status == "disabled":
        return refuse("tool_disabled", tool=selected)
    event("tool_available", "pass", tool=selected, status=tool.status.status)
    if selected == "landcover":
        required = (("VV", "VH", "B02", "B03", "B04", "B08") if mode == "S1S2" else
                    ("VV", "VH") if mode == "S1" else ("B02", "B03", "B04", "B08"))
        required_sensors = ("S1", "S2") if mode == "S1S2" else ("S1",) if mode == "S1" else ("S2",)
    elif selected == "change-deterministic":
        required = (("VV", "VH", "dataMask") if mode == "S1" else
                    ("red", "green", "blue") if mode == "RGB_HR" else
                    ("B02", "B03", "B04", "B08", "dataMask"))
        required_sensors = ("S1",) if mode == "S1" else ("RGB_HR",) if mode == "RGB_HR" else ("S2",)
    elif selected == "flood-otsu":
        required, required_sensors = ("VH",), ("S1",)
        mode = "S1"
    elif selected == "optical-sar-deterministic":
        required, required_sensors = tool.needs.bands, ("S1", "S2")
    else:
        required = tool.needs.bands
        # T07 VLM/DINO sensor lists describe alternatives, not joint inputs.
        required_sensors = () if scene.sensor in tool.needs.sensors or not tool.needs.sensors else tool.needs.sensors
    required_sensors = tuple(dict.fromkeys((*required_sensors, *query.needs_sensor)))
    for index, item in enumerate(scenes):
        missing_sensors = tuple(s for s in required_sensors if s not in sensors(item))
        if missing_sensors:
            return refuse("sensor_missing", needs=missing_sensors, scene=index)
        event("sensors", "pass", required=list(required_sensors), scene=index)
        missing = [b for b in required if b not in item.bands]
        if missing:
            return refuse("bands_missing", needs=required_sensors, missing_bands=missing, scene=index)
        event("bands", "pass", required=list(required), scene=index)
        if query.needs_gsd_m is not None:
            event("gsd_threshold", "used", threshold=query.needs_gsd_m,
                  observed=item.gsd_m, provenance="Query.needs_gsd_m", scene=index)
            if item.gsd_m > query.needs_gsd_m:
                return refuse("gsd_too_coarse", scene=index)
    if not training_stats and selected in DETERMINISTIC:
        event("ood", "not_applicable")
    else:
        event("ood_threshold", "used", threshold=OOD_STD, provenance="PROVISIONAL")
        # Every required band/date needs a usable observation; absence is uncertainty.
        bands = required or scene.bands
        for index, item in enumerate(scenes):
            means = band_means[index] if index < len(band_means) else {}
            for band in bands:
                stats = (training_stats or {}).get(band)
                observed = means.get(band)
                if (not isinstance(stats, (tuple, list)) or len(stats) != 2 or observed is None
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                   and math.isfinite(v) for v in (*stats, observed)) or stats[1] <= 0):
                    event("ood", "unknown", band=band, scene=index, warning="ood_unknown")
                else:
                    shift = abs(observed - stats[0]) / stats[1]
                    if not math.isfinite(shift):
                        event("ood", "unknown", band=band, scene=index, warning="ood_unknown")
                        continue
                    event("ood", "shift" if shift > OOD_STD else "pass", band=band,
                          scene=index, std_shift=shift,
                          warning="out_of_distribution" if shift > OOD_STD else None)
    event("gate", "pass", clean=not any(e.summary.get("warning") for e in events))
    return GateDecision(selected, mode, None, tuple(events))
