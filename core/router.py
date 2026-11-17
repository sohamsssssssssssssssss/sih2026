"""Deterministic English routing; no model imports or inference."""
from dataclasses import dataclass
import re

from core.schema import Query, Scene

NON_IMAGERY = re.compile(
    r"\b(price|prices|cost|costs|worth|owner|owners|owns|ownership|name|names|named|"
    r"people|person|persons|population|residents|human|humans|who|whose|"
    r"future|tomorrow|predict|prediction|forecast|will|expensive|called|personnel)\b|\bnext (year|month|week|day)\b", re.I
)
RULES = (
    ("change", r"\b(chang\w*|before|after|compare|expanded|expansion|since|difference|differences)\b"),
    ("count", r"\b(count|counting|detect|detection|locate|locating)\b|\bhow many\b"),
    ("flood", r"\b(flood\w*|water|inundat\w*)\b"),
    ("landcover", r"\bland[ -]?cover\b|\b(forest\w*|cropland|vegetation|built[ -]up)\b"),
)


@dataclass(frozen=True)
class Route:
    intent: str
    tools: tuple[str, ...]
    reason: str


def route(query: Query, scene: Scene) -> Route:
    if NON_IMAGERY.search(query.text):
        return Route("not_answerable", (), "Question requests non-imagery information or people")
    intent = next((name for name, pattern in RULES if re.search(pattern, query.text, re.I)), "general")
    if query.qtype == "change":
        intent = "change"
    tools = {
        "landcover": ("landcover",),
        "flood": (("flood-otsu",) if scene.sensor == "S1" else
                  ("optical-sar-deterministic", "flood-otsu")),
        "change": ("change-deterministic",),
        "count": ("grounding-dino-swint",),
        "general": ("qwen2.5vl-3b",),
    }[intent]
    return Route(intent, tools, f"Deterministic {intent} pattern" if intent != "general" else "Unknown/general text: VLM fallback")
