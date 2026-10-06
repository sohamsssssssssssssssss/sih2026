"""Pure request planning with deterministic, auditable rules.

An explicit capability always wins. Otherwise the parsed question
(orchestrator.question) routes, and the keyword rules are the fallback:
optical/SAR, combined temporal change plus spatial localization, temporal
change, grounding, then single-image VQA. The parse only overrides the keyword
rules when it picks a different capability; then the rule id is
``parsed_<intent>``. When both agree the keyword rule id is kept, so a
``parsed_*`` id marks exactly the routings the keyword rules got wrong.

The combined rule is the only multi-step decomposition: it selects change_vqa
as the primary capability while the execution-plan layer represents the future
change_vqa → grounding chain. Such plans remain representation only until
providers exist; there is no replanning and no autonomy. It holds unless the
question is a flood_change over a SAR pair, whose water-change polygons
already localize, so one change_vqa step answers it.
"""

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
from orchestrator.question import (
    CHANGE,
    DESCRIBE,
    FLOOD_CHANGE,
    LOCATE,
    ParsedQuestion,
    has_change_intent,
    has_grounding_intent,
    has_optical_sar_intent,
    parse_question,
    tokenize,
)
from orchestrator.question import OPTICAL_SAR as OPTICAL_SAR_INTENT

PLANNER_VERSION = "phase3-parsed-v1"

SAR_SENSORS = frozenset({"sar", "radar", "sentinel 1", "synthetic aperture radar"})
# Parsed intent -> capability; "unknown" has none and leaves routing to the keyword rules.
INTENT_CAPABILITIES = {
    FLOOD_CHANGE: CHANGE_VQA,
    CHANGE: CHANGE_VQA,
    LOCATE: GROUNDING,
    DESCRIBE: SINGLE_IMAGE_VQA,
    OPTICAL_SAR_INTENT: OPTICAL_SAR,
}
INTENT_REASONS = {
    FLOOD_CHANGE: "The question asks where flooding changed between two dates.",
    CHANGE: "The question asks for temporal comparison.",
    LOCATE: "The question asks to find or localize objects in the scene.",
    DESCRIBE: "The question asks about the contents of a single scene.",
    OPTICAL_SAR_INTENT: "The question asks for combined optical and SAR analysis.",
}

Selection = tuple[str, str, str, str | None]  # capability, rule id, reason, requested

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
    return "where" in tokens or has_grounding_intent(tokens)


def _selection(request: PlanRequest, tokens: tuple[str, ...]) -> Selection:
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
    sar_pair = len(request.scene_ids) >= 2 and _sensor_key(request.sensor) in SAR_SENSORS
    return _parsed_selection(parse_question(request.question), _keyword_selection(tokens), sar_pair)


def _parsed_selection(parsed: ParsedQuestion, keyword: Selection, sar_pair: bool) -> Selection:
    """The parse routes; the keyword selection stands when the parse is unknown or agrees."""
    intent = parsed.intent
    capability = INTENT_CAPABILITIES.get(intent)
    if keyword[1] == TEMPORAL_LOCALIZATION_RULE_ID:
        overrides = intent == FLOOD_CHANGE and sar_pair
    else:
        overrides = capability not in (None, keyword[0])
    if not overrides:
        return keyword
    return capability, f"parsed_{intent}", INTENT_REASONS[intent], None


def _keyword_selection(tokens: tuple[str, ...]) -> Selection:
    if has_optical_sar_intent(tokens):
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
    if has_change_intent(tokens):
        return (
            CHANGE_VQA,
            "change_temporal_compare",
            "The request asks for temporal comparison.",
            None,
        )
    if has_grounding_intent(tokens):
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


def _sensor_key(sensor: str | None) -> str:
    return " ".join(tokenize(sensor or ""))


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
    tokens = tokenize(request.question)
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
        from orchestrator.capabilities import capability_readiness

        readiness = capability_readiness(capability)
        provider_available = bool(readiness["available"])
        provider = resolved.provider_name
        unavailable_reason = None if provider_available else str(readiness["detail"])

    if rule_id == TEMPORAL_LOCALIZATION_RULE_ID:
        unavailable_reason = "Multi-step change-to-grounding execution is not implemented."

    if capability == SINGLE_IMAGE_VQA and _sensor_key(request.sensor) in SAR_SENSORS:
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
