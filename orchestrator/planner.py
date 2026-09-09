"""Pure Phase 0 request planning with deterministic, auditable rules.

Precedence is explicit capability, optical/SAR, combined temporal change plus
spatial localization, temporal change, grounding, then single-image VQA. The
combined rule is the only multi-step decomposition: it selects change_vqa as
the primary capability while the execution-plan layer represents the future
change_vqa → grounding chain. Such plans remain representation only until
providers exist; there is no replanning and no autonomy.
"""

import re
from dataclasses import dataclass

from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    KNOWN_CAPABILITIES,
    OPTICAL_SAR,
    SINGLE_IMAGE_VQA,
    CapabilityUnavailable,
    UnknownCapability,
    resolve_provider,
)

PLANNER_VERSION = "phase0-rules-v1"

# Narrow combined intent: temporal change AND spatial localization together.
TEMPORAL_LOCALIZATION_RULE_ID = "temporal_change_then_grounding"


class InvalidPlanRequest(ValueError):
    """The structured planning request is incomplete or invalid."""


@dataclass(frozen=True)
class PlanRequest:
    question: str
    scene_ids: tuple[str, ...]
    sensor: str | None = None
    requested_capability: str | None = None


@dataclass(frozen=True)
class Plan:
    planner_version: str
    rule_id: str
    requested_capability: str | None
    selected_capability: str
    executable: bool
    reason: str
    required_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    provider_available: bool
    provider: str | None
    unavailable_reason: str | None


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _phrase(tokens: tuple[str, ...], words: str) -> bool:
    needle = tuple(words.split())
    size = len(needle)
    return any(tokens[index : index + size] == needle for index in range(len(tokens)))


def _has_optical_sar_intent(tokens: tuple[str, ...]) -> bool:
    if any(_phrase(tokens, phrase) for phrase in ("cross modal", "multi sensor")):
        return True
    sar = (
        "sar" in tokens
        or "radar" in tokens
        or _phrase(tokens, "sentinel 1")
        or _phrase(tokens, "synthetic aperture radar")
    )
    optical = "optical" in tokens or _phrase(tokens, "sentinel 2")
    paired = _phrase(tokens, "sar optical") or _phrase(tokens, "optical sar")
    joint = any(
        word in tokens
        for word in (
            "and",
            "both",
            "compare",
            "together",
            "use",
            "using",
            "versus",
            "vs",
            "confirm",
            "miss",
            "misses",
        )
    )
    return sar and optical and (joint or paired)


def _has_change_intent(tokens: tuple[str, ...]) -> bool:
    phrases = (
        "what changed",
        "what change",
        "before and after",
        "between these images",
        "between the images",
        "between two images",
        "between the two images",
        "between these scenes",
        "between the scenes",
        "between two scenes",
        "between the two scenes",
        "over time",
        "new since",
        "removed since",
        "appeared since",
        "disappeared since",
    )
    if any(_phrase(tokens, phrase) for phrase in phrases):
        return True
    temporal_verbs = {
        "change",
        "changed",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "expand",
        "expanded",
        "shrink",
        "shrunk",
        "appear",
        "appeared",
        "disappear",
        "disappeared",
    }
    temporal_cues = {"has", "have", "did", "where", "since", "between", "from"}
    if temporal_verbs.intersection(tokens) and temporal_cues.intersection(tokens):
        return True
    return (
        "from" in tokens
        and "to" in tokens
        and sum(token.isdigit() and len(token) == 4 for token in tokens) >= 2
    )


def _has_grounding_intent(tokens: tuple[str, ...]) -> bool:
    phrases = (
        "where is",
        "where are",
        "where can",
        "show me where",
        "which part of the image",
        "location of",
        "point to",
        "give me the bounding box",
        "bounding box of",
        "draw a bounding box",
        "coordinates in the image",
        "can you locate",
        "can you localize",
    )
    if any(_phrase(tokens, phrase) for phrase in phrases):
        return True
    if "locate" in tokens or "localize" in tokens:
        return True
    if tokens and tokens[0] in {
        "mark",
        "highlight",
        "segment",
        "mask",
    }:
        return True
    if "bbox" in tokens or (
        "polygon" in tokens and tokens and tokens[0] in {"give", "draw", "return"}
    ):
        return True
    directions = {
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    }
    return "concentrated" in tokens and bool(directions.intersection(tokens))


def _has_temporal_verb(tokens: tuple[str, ...]) -> bool:
    temporal_verbs = {
        "change",
        "changed",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "expand",
        "expanded",
        "shrink",
        "shrunk",
        "appear",
        "appeared",
        "disappear",
        "disappeared",
    }
    return bool(temporal_verbs.intersection(tokens))


def _has_combined_temporal_localization_intent(tokens: tuple[str, ...]) -> bool:
    """Narrow trigger: temporal change AND spatial localization, both clear.

    Matches requests like "Where did flooding increase?" or "Locate the
    areas that changed." Single-intent questions ("What changed?", "Where
    is the building?", "Is flooding visible?") stay single-step.
    """
    if not _has_temporal_verb(tokens):
        return False
    return "where" in tokens or _has_grounding_intent(tokens)


def _selection(
    request: PlanRequest, tokens: tuple[str, ...]
) -> tuple[str, str, str, str | None]:
    requested = (
        request.requested_capability.strip()
        if request.requested_capability is not None
        else None
    )
    if requested == "":
        raise UnknownCapability("Unknown capability: ")
    if requested is not None:
        if requested not in KNOWN_CAPABILITIES:
            raise UnknownCapability(f"Unknown capability: {requested}")
        return (
            requested,
            "explicit_capability",
            "The request explicitly selects this capability.",
            requested,
        )
    if _has_optical_sar_intent(tokens):
        return (
            OPTICAL_SAR,
            "optical_sar_cross_modal",
            "The request asks for combined optical and SAR analysis.",
            None,
        )
    if _has_combined_temporal_localization_intent(tokens):
        return (
            CHANGE_VQA,
            TEMPORAL_LOCALIZATION_RULE_ID,
            "The request requires temporal change analysis followed by spatial localization.",
            None,
        )
    if _has_change_intent(tokens):
        return (
            CHANGE_VQA,
            "change_temporal_compare",
            "The request asks for temporal comparison.",
            None,
        )
    if _has_grounding_intent(tokens):
        return (
            GROUNDING,
            "grounding_spatial_localization",
            "The request asks for spatial localization.",
            None,
        )
    return (
        SINGLE_IMAGE_VQA,
        "default_single_image_vqa",
        "The request asks about the contents of a single scene.",
        None,
    )


def _inputs(capability: str, scene_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if capability == CHANGE_VQA:
        missing = (
            ()
            if scene_count >= 2
            else (("scene",) if scene_count == 0 else ("second_scene",))
        )
        return ("scene_pair",), missing
    if capability == OPTICAL_SAR:
        missing = (
            ()
            if scene_count >= 2
            else (("scene",) if scene_count == 0 else ("second_scene",))
        )
        return ("optical_scene", "sar_scene"), missing
    return ("single_scene",), (() if scene_count >= 1 else ("scene",))


def plan_request(request: PlanRequest) -> Plan:
    """Return the same immutable plan for the same request and registry state."""
    tokens = _tokens(request.question)
    if not tokens:
        raise InvalidPlanRequest("A non-empty question is required.")
    capability, rule_id, reason, requested = _selection(request, tokens)
    required, missing = _inputs(capability, len(request.scene_ids))
    try:
        resolved = resolve_provider(capability)
    except CapabilityUnavailable:
        provider_available = False
        provider = None
        unavailable_reason = "No provider is registered for the selected capability."
    else:
        provider_available = True
        provider = resolved.provider_name
        unavailable_reason = None

    sensor = " ".join(_tokens(request.sensor or ""))
    if capability == SINGLE_IMAGE_VQA and sensor in {
        "sar",
        "radar",
        "sentinel 1",
        "synthetic aperture radar",
    }:
        unavailable_reason = "Single-image SAR interpretation is not currently supported."

    return Plan(
        planner_version=PLANNER_VERSION,
        rule_id=rule_id,
        requested_capability=requested,
        selected_capability=capability,
        executable=provider_available and not missing and unavailable_reason is None,
        reason=reason,
        required_inputs=required,
        missing_inputs=missing,
        provider_available=provider_available,
        provider=provider,
        unavailable_reason=unavailable_reason,
    )
