"""Deterministic English routing; no model imports or inference."""
from dataclasses import dataclass
import re

from core.schema import Query, Scene

NON_IMAGERY = re.compile(
    r"\b(price|prices|cost|costs|worth|owner|owners|owns|ownership|name|names|named|"
    r"people|person|persons|population|residents|human|humans|who|whose|"
    r"future|tomorrow|predict|prediction|forecast|will|expensive|called|personnel)\b|\bnext (year|month|week|day)\b", re.I
)
CHANGE_UNSUPPORTED = (
    "Change questions are not supported because the methods are unvalidated. T18 found that image differencing did not beat "
    "the majority baseline on CDVQA, while question text alone strongly predicted the labels."
)
TEMPORAL = re.compile(
    r"\b(change(?:s|d)?|changing|before|after|versus|expand(?:s|ed|ing)?|expansion|since|"
    r"differences?|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
    r"shrank|shr(?:ink|unk)|grew|growth|appeared|disappeared)\b"
    r"|\b(?:two|both|different) (?:dates|time ?points)\b"
    r"|\b(?:19|20)\d{2}\b.*\b(?:19|20)\d{2}\b",
    re.I | re.S,
)
COMPARE = re.compile(r"\bcompar(?:e|es|ed|ing|ison)\b", re.I)

RULES = (
    ("count", r"\b(count|counting|detect|detection|locate|locating)\b|\bhow many\b"),
    ("flood", r"\b(flood\w*|water|inundat\w*)\b"),
    ("landcover", r"\bland[ -]?cover\b|\b(forest\w*|cropland|vegetation|built[ -]up)\b"),
)


@dataclass(frozen=True)
class Route:
    intent: str
    tools: tuple[str, ...]
    reason: str


def route(query: Query, scene: Scene, *, has_second_scene: bool = False) -> Route:
    if has_second_scene or query.qtype == "change" or TEMPORAL.search(query.text) or COMPARE.search(query.text):
        return Route("change", (), CHANGE_UNSUPPORTED)
    if NON_IMAGERY.search(query.text):
        return Route("not_answerable", (), "Question requests non-imagery information or people")
    intent = next((name for name, pattern in RULES if re.search(pattern, query.text, re.I)), "general")
    tools = {
        "landcover": ("landcover",),
        "flood": ("flood-otsu",),
        "count": ("grounding-dino-swint",),
        "general": ("qwen2.5vl-3b",),
    }[intent]
    return Route(intent, tools, f"Deterministic {intent} pattern" if intent != "general" else "Unknown/general text: VLM fallback")
