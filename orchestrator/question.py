"""Deterministic, rules-first question parsing (Phase 3).

``parse_question`` turns a question into an intent, a place name and a
before/after date pair, and lists what the intent needs but the text lacks.
It is pure: no I/O, no clock and no gazetteer. Unknown values stay ``None`` or
are listed in ``missing``; nothing is guessed, and a date without a year never
takes the current year. A later LLM extractor may fill the same schema.

The keyword predicates the planner falls back on live here, so the parse and
the fallback cannot drift apart: the parse is those rules plus flood
vocabulary, dates and imperative locate verbs.

Ceilings: English only. A place is the first capitalised name after a
preposition, so "Patna: which villages flooded?" finds none. Dates are whole
days ("August 2024" and "last week" are not dates). With three or more dates
only the earliest and latest are kept. Any four-digit 19xx/20xx number counts
as a stated year, so "2000 hectares" can block a yearless date.
"""

import re
from dataclasses import dataclass
from datetime import date
from itertools import takewhile

INTENTS = ("flood_change", "change", "locate", "describe", "optical_sar", "unknown")
FLOOD_CHANGE, CHANGE, LOCATE, DESCRIBE, OPTICAL_SAR, UNKNOWN = INTENTS
MISSING_FIELDS = ("place", "before_date", "after_date", "year")

FLOOD_WORDS = frozenset(
    {"flood", "floods", "flooded", "flooding", "inundated", "inundation", "submerged", "waterlogged", "deluge"}
)
FLOOD_PHRASES = ("under water", "water logged")
# Words that make a flood question temporal even without a date.
FLOOD_TEMPORAL_WORDS = frozenset(
    {"since", "between", "newly", "still", "anymore", "recede", "receded", "receding",
     "spread", "spreading", "rose", "risen", "worsened"}
)
AREA_WORDS = frozenset(
    {"village", "villages", "area", "areas", "region", "regions", "district", "districts",
     "block", "blocks", "town", "towns", "place", "places", "locality", "localities", "part", "parts"}
)
LOCATE_VERBS = frozenset(
    {"find", "show", "detect", "mark", "highlight", "count", "locate", "localize", "localise", "segment", "mask",
     "draw", "outline", "spot", "pinpoint", "delineate", "circle", "box"}
)
# "Identify the land cover" describes; "identify each brick kiln" locates.
ENUMERATING_WORDS = frozenset({"each", "every", "all"})
# A flood word about the scene in hand ("is this field waterlogged?") is not a flood event.
DEICTIC_PHRASES = (
    "this image", "the image", "this scene", "the scene", "this picture", "this photo",
    "this field", "this tile", "this area",
)
COMPARISON_WORDS = frozenset(
    {"compare", "compared", "comparing", "comparison", "comparision", "versus", "vs"}
)
COMPARISON_PHRASES = (
    "before after", "before and after", "earlier and later", "old and new", "these two", "the two images",
    "first one", "second one", "first image", "second image", "than before", "over the last", "over the past",
)
# Change verbs and nouns that imply two times on their own.
CHANGE_WORDS = frozenset(
    {"reduced", "reduction", "retreat", "retreated", "retreating", "grown", "grew", "shrank", "shrinking",
     "deforestation", "encroachment", "encroached"}
)
# Sensors joined in one request ("Sentinel-1 and Sentinel-2", "fuse the optical and SAR")
# ask for joint use; optical named only as the reason to use radar does not.
SENSOR_CONJUNCTIONS = frozenset({"and", "with", "plus", "vs", "versus"})
JOINT_USE_WORDS = frozenset({"both", "fuse", "fusion", "fused", "combine", "combined", "together", "jointly"})
SINCE_YEAR_RE = re.compile(r"\bsince\s+(?:19|20)\d{2}\b", re.IGNORECASE)
NOT_AN_OBJECT = frozenset({"me", "us"})
POLITE_PREFIXES = (("please",), ("can", "you"), ("could", "you"), ("would", "you"), ("will", "you"))
YES_NO_STARTERS = frozenset(
    {"is", "are", "was", "were", "does", "do", "did", "has", "have", "had", "can", "could", "will", "would", "should"}
)
WHAT_IS_STARTS = frozenset(
    {("what", "is"), ("what", "are"), ("what", "was"), ("what", "were"), ("what", "s"),
     ("what", "does"), ("what", "do"), ("how", "many"), ("how", "much")}
)

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_ORDINAL = r"(?:st|nd|rd|th)?"
# Numeric dates are day-first (Indian usage) unless they start with the year.
DATE_RE = re.compile(
    rf"""(?<![\w/-])(?:
        (?P<iso_y>\d{{4}})-(?P<iso_m>\d{{1,2}})-(?P<iso_d>\d{{1,2}})
      | (?P<num_d>\d{{1,2}})(?P<sep>[/-])(?P<num_m>\d{{1,2}})(?P=sep)(?P<num_y>\d{{4}})
      | (?P<dm_d>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?(?P<dm_m>{_MONTH})\b\.?(?:,?\s+(?P<dm_y>\d{{4}}))?
      | (?P<md_m>{_MONTH})\b\.?\s+(?P<md_d>\d{{1,2}}){_ORDINAL}(?:,?\s+(?P<md_y>\d{{4}}))?
    )(?![\w/-])""",
    re.IGNORECASE | re.VERBOSE,
)
DATE_FORMATS = (
    ("iso", "date_iso"),
    ("num", "date_day_first_numeric"),
    ("dm", "date_day_month"),
    ("md", "date_month_day"),
)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
SINCE_OR_BEFORE_RE = re.compile(r"\b(?:since|before)\s+(?:the\s+)?$", re.IGNORECASE)

PLACE_PREPOSITIONS = ("near", "in", "around", "at", "of", "across", "over", "for")
PLACE_SUFFIXES = frozenset({"district", "block", "town", "city", "village"})
SENSOR_WORDS = frozenset({"sentinel", "sar", "radar", "optical", "landsat", "modis", "copernicus"})
NOT_PLACE_WORDS = SENSOR_WORDS | frozenset(MONTHS)
PLACE_RE = re.compile(
    rf"\b(?i:{'|'.join(PLACE_PREPOSITIONS)})\s+(?:(?i:the)\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)"
)


@dataclass(frozen=True)
class ParsedQuestion:
    text: str
    intent: str  # one of INTENTS
    place: str | None  # as written, without "near"/"district"; never looked up here
    before: date | None  # earlier of two dates, or the date in "since X" / "before X"
    after: date | None  # later of two dates, or the single date of "flooded on X"
    missing: tuple[str, ...]  # subset of MISSING_FIELDS, in that order
    rules: tuple[str, ...]  # ids of the rules that fired, for the trace


# Keyword predicates shared with the planner's fallback rules.


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def has_phrase(tokens: tuple[str, ...], words: str) -> bool:
    needle = tuple(words.split())
    size = len(needle)
    return any(tokens[index : index + size] == needle for index in range(len(tokens)))


def has_optical_sar_intent(tokens: tuple[str, ...]) -> bool:
    if any(has_phrase(tokens, phrase) for phrase in ("cross modal", "multi sensor")):
        return True
    sar = (
        "sar" in tokens
        or "radar" in tokens
        or has_phrase(tokens, "sentinel 1")
        or has_phrase(tokens, "synthetic aperture radar")
    )
    optical = "optical" in tokens or has_phrase(tokens, "sentinel 2")
    paired = has_phrase(tokens, "sar optical") or has_phrase(tokens, "optical sar")
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


def has_change_intent(tokens: tuple[str, ...]) -> bool:
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
    if any(has_phrase(tokens, phrase) for phrase in phrases):
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


def has_grounding_intent(tokens: tuple[str, ...]) -> bool:
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
    if any(has_phrase(tokens, phrase) for phrase in phrases):
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


# Parser-only rules.


def _is_flood(tokens: tuple[str, ...]) -> bool:
    return bool(FLOOD_WORDS.intersection(tokens)) or any(has_phrase(tokens, p) for p in FLOOD_PHRASES)


def _asks_which_area(tokens: tuple[str, ...]) -> bool:
    return ("which" in tokens or "what" in tokens) and bool(AREA_WORDS.intersection(tokens))


def _is_temporal(tokens: tuple[str, ...], date_count: int) -> bool:
    return date_count > 0 or has_change_intent(tokens) or bool(FLOOD_TEMPORAL_WORDS.intersection(tokens))


def _is_imperative_locate(tokens: tuple[str, ...]) -> bool:
    """A locate request with an object: "Show the ships", "Can you point out the bridge"."""
    asks_for_boxes = has_phrase(tokens, "bounding box") or has_phrase(tokens, "bounding boxes")
    if asks_for_boxes and tokens[0] not in YES_NO_STARTERS:  # not "does the image show a bounding box?"
        return True
    rest = next((tokens[len(p) :] for p in POLITE_PREFIXES if tokens[: len(p)] == p), tokens)
    if rest[:2] == ("point", "out"):
        verb, objects = "point out", rest[2:]
    else:
        verb, objects = (rest[0], rest[1:]) if rest else ("", ())
    if verb == "identify":
        return bool(ENUMERATING_WORDS.intersection(objects[:2]))
    return (verb == "point out" or verb in LOCATE_VERBS) and any(t not in NOT_AN_OBJECT for t in objects)


def _is_comparison(tokens: tuple[str, ...], text: str) -> bool:
    """Two times without a change verb: "compared with", "before after", "since 2020", "reduced"."""
    return bool(
        COMPARISON_WORDS.intersection(tokens)
        or CHANGE_WORDS.intersection(tokens)
        or any(has_phrase(tokens, phrase) for phrase in COMPARISON_PHRASES)
        or SINCE_YEAR_RE.search(text)
    )


def _sensor_families(tokens: tuple[str, ...]) -> tuple[str | None, ...]:
    """Each token's sensor family ("sar", "optical") or None, with "sentinel 1/2" folded into one slot."""
    families: list[str | None] = []
    index = 0
    while index < len(tokens):
        pair = tokens[index : index + 2]
        if pair in (("sentinel", "1"), ("sentinel", "2")):
            families.append("sar" if pair[1] == "1" else "optical")
            index += 2
            continue
        token = tokens[index]
        families.append("sar" if token in {"sar", "radar"} else "optical" if token == "optical" else token)
        index += 1
    return tuple(families)


def _names_both_sensors(tokens: tuple[str, ...]) -> bool:
    families = set(_sensor_families(tokens))
    return {"sar", "optical"} <= families


def _joins_sensors(tokens: tuple[str, ...]) -> bool:
    """Both sensors named and asked for together: adjacent ("SAR and optical") or "both"/"fuse"."""
    if not _names_both_sensors(tokens):
        return False
    if JOINT_USE_WORDS.intersection(tokens):
        return True
    families = _sensor_families(tokens)
    return any(
        {families[i], families[i + 2]} == {"sar", "optical"} and families[i + 1] in SENSOR_CONJUNCTIONS
        for i in range(len(families) - 2)
    )


def _is_flood_event(tokens: tuple[str, ...], text: str, date_count: int, place: str | None) -> bool:
    """Flood words about a real event, not about the scene in hand.

    "Flood map for Assam", "Did it flood?" and "Kerala floods August 2018" are
    events; "Is the area flooded?" and "the flooded fields in this image" are not.
    """
    if not _is_flood(tokens) or any(has_phrase(tokens, phrase) for phrase in DEICTIC_PHRASES):
        return False
    return bool(
        place or date_count or YEAR_RE.search(text) or tokens[:1] == ("did",)
        or _is_temporal(tokens, date_count) or _is_comparison(tokens, text) or _asks_which_area(tokens)
    )


def _is_describe(tokens: tuple[str, ...]) -> bool:
    return bool(tokens) and (
        tokens[0] in YES_NO_STARTERS or tokens[0] == "describe" or tokens[:2] in WHAT_IS_STARTS
    )


def _intent(tokens: tuple[str, ...], text: str, date_count: int, place: str | None) -> str:
    """Joined sensors first, then dated flood events, then two named sensors, then time, then space.

    "Use Sentinel-1 and Sentinel-2 to map flooding" asks for joint use. A dated
    flood question that names optical only as the reason to use radar is still a
    flood question. Without dates, naming both sensors is optical/SAR.
    """
    flood_event = _is_flood_event(tokens, text, date_count, place)
    if _joins_sensors(tokens):
        return OPTICAL_SAR
    if flood_event and date_count:
        return FLOOD_CHANGE
    if has_optical_sar_intent(tokens) or _names_both_sensors(tokens):
        return OPTICAL_SAR
    if flood_event:
        return FLOOD_CHANGE
    if has_change_intent(tokens) or date_count >= 2 or _is_comparison(tokens, text):
        return CHANGE
    if has_grounding_intent(tokens) or _is_imperative_locate(tokens):
        return LOCATE
    if _is_describe(tokens):
        return DESCRIBE
    return UNKNOWN


def _date_parts(match: re.Match[str]) -> tuple[str, int | None, int, int]:
    """(rule id, year or None, month, day) for whichever DATE_RE branch matched."""
    for prefix, rule in DATE_FORMATS:
        day = match.group(f"{prefix}_d")
        if day is not None:
            year, month = match.group(f"{prefix}_y"), match.group(f"{prefix}_m")
            month_number = int(month) if month.isdigit() else MONTHS[month.casefold()]
            return rule, (int(year) if year else None), month_number, int(day)
    raise AssertionError("DATE_RE matched no branch")  # unreachable: every branch has a day


def _resolve(match: re.Match[str], years: frozenset[int]) -> tuple[date | None, tuple[str, ...]]:
    rule, year, month, day = _date_parts(match)
    rules = (rule,)
    if year is None:
        if len(years) != 1:
            return None, rules + ("date_year_missing",)
        (year,) = years
        rules += ("date_year_from_question",)
    try:
        return date(year, month, day), rules
    except ValueError:
        return None, rules + ("date_invalid",)


def _dates(
    text: str, matches: tuple[re.Match[str], ...]
) -> tuple[date | None, date | None, bool, tuple[str, ...]]:
    """(before, after, year missing, rule ids). Any unresolved date leaves the order unknown."""
    years = frozenset(int(year) for year in YEAR_RE.findall(text))
    resolved = [_resolve(match, years) for match in matches]
    values = [value for value, _ in resolved]
    rules = tuple(rule for _, ids in resolved for rule in ids)
    year_missing = "date_year_missing" in rules
    if not values or None in values:
        return None, None, year_missing, rules
    distinct = sorted(set(values))
    if len(distinct) > 1:
        return distinct[0], distinct[-1], year_missing, rules
    if SINCE_OR_BEFORE_RE.search(text[: matches[0].start()]):
        return distinct[0], None, year_missing, rules + ("date_since_or_before",)
    return None, distinct[0], year_missing, rules


def _place(text: str) -> tuple[str | None, tuple[str, ...]]:
    for match in PLACE_RE.finditer(text):
        words = list(takewhile(lambda word: word.casefold() not in NOT_PLACE_WORDS, match.group(1).split()))
        kept = words
        while kept and kept[-1].casefold() in PLACE_SUFFIXES:
            kept = kept[:-1]
        if kept:
            stripped = ("place_suffix_stripped",) if len(kept) < len(words) else ()
            return " ".join(kept), ("place_after_preposition",) + stripped
    return None, ()


def _missing(intent: str, place: str | None, before: date | None, after: date | None, year_missing: bool) -> tuple[str, ...]:
    needs = intent == FLOOD_CHANGE
    lacking = {
        "place": needs and place is None,
        "before_date": needs and before is None,
        "after_date": needs and after is None,
        "year": year_missing,
    }
    return tuple(field for field in MISSING_FIELDS if lacking[field])


def parse_question(text: str) -> ParsedQuestion:
    """Parse one question; the same text always yields the same ParsedQuestion."""
    tokens = tokenize(text)
    matches = tuple(DATE_RE.finditer(text))
    place, place_rules = _place(text)
    intent = _intent(tokens, text, len(matches), place)
    before, after, year_missing, date_rules = _dates(text, matches)
    intent_rules = () if intent == UNKNOWN else (f"intent_{intent}",)
    return ParsedQuestion(
        text=text,
        intent=intent,
        place=place,
        before=before,
        after=after,
        missing=_missing(intent, place, before, after, year_missing),
        rules=tuple(dict.fromkeys(intent_rules + date_rules + place_rules)),
    )
