"""Deterministic question-parser tests; pure, no I/O, no clock."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from orchestrator.question import INTENTS, ParsedQuestion, parse_question

ALL = ("place", "before_date", "after_date")
D = date

# (question, intent, place, before, after, missing)
CASES = [
    # flood_change: the target demo, then every way it can be incomplete.
    ("Which villages near Patna flooded on 20 Aug 2024 that weren't flooded on 1 Aug?",
     "flood_change", "Patna", D(2024, 8, 1), D(2024, 8, 20), ()),
    ("Which villages near Patna flooded on 20 Aug that weren't flooded on 1 Aug?",
     "flood_change", "Patna", None, None, ("before_date", "after_date", "year")),
    ("Which areas of Darbhanga district were inundated between 1 September 2024 and 30 September 2024?",
     "flood_change", "Darbhanga", D(2024, 9, 1), D(2024, 9, 30), ()),
    ("Was the town of Silchar under water on 20/06/2022?",
     "flood_change", "Silchar", None, D(2022, 6, 20), ("before_date",)),
    ("Show flooded areas in Aluva since 2018-08-14",
     "flood_change", "Aluva", D(2018, 8, 14), None, ("after_date",)),
    ("Did flooding expand?", "flood_change", None, None, None, ALL),
    ("Which villages are waterlogged?", "flood_change", None, None, None, ALL),
    ("How much land around Kiratpur Block was submerged on Sept 30, 2024 compared with Sept 1, 2024?",
     "flood_change", "Kiratpur", D(2024, 9, 1), D(2024, 9, 30), ()),
    ("Has the deluge across Assam receded since 1st July 2022?",
     "flood_change", "Assam", D(2022, 7, 1), None, ("after_date",)),
    ("Which areas over Bihar were water-logged on 20th August 2024 but dry on 1st August 2024?",
     "flood_change", "Bihar", D(2024, 8, 1), D(2024, 8, 20), ()),
    ("Which villages near Patna flooded before 15 Aug 2024?",
     "flood_change", "Patna", D(2024, 8, 15), None, ("after_date",)),
    ("Where did the inundation spread after 2024-08-01 in Patna City?",
     "flood_change", "Patna", None, D(2024, 8, 1), ("before_date",)),
    # Year rule: a yearless date borrows the one year stated, never a guessed one.
    ("Which villages near Patna flooded on 20 Aug 2024 but not on 1 Aug 2023?",
     "flood_change", "Patna", D(2023, 8, 1), D(2024, 8, 20), ()),
    ("Which villages near Patna flooded on 20 Aug 2024 but not on 1 Aug, unlike 2023?",
     "flood_change", "Patna", None, None, ("before_date", "after_date", "year")),
    ("Which villages near Patna flooded in the 2024 monsoon between Aug 1 and Aug 20?",
     "flood_change", "Patna", D(2024, 8, 1), D(2024, 8, 20), ()),
    ("Compare 1 Aug and 20 Aug flood maps for Patna",
     "flood_change", "Patna", None, None, ("before_date", "after_date", "year")),
    ("Was it cloudy on 5 Dec?", "describe", None, None, None, ("year",)),
    # Invalid calendar dates stay unresolved; one bad date leaves the order unknown.
    ("Which villages near Patna flooded on 31 Sep 2024?",
     "flood_change", "Patna", None, None, ("before_date", "after_date")),
    ("Which villages near Patna flooded on 30/02/2024 that were dry on 01/02/2024?",
     "flood_change", "Patna", None, None, ("before_date", "after_date")),
    # change: temporal comparison without flood vocabulary; dates are optional.
    ("What changed in Patna between 2024-08-01 and 2024-08-20?",
     "change", "Patna", D(2024, 8, 1), D(2024, 8, 20), ()),
    ("What changed between 01-08-2024 and 20-08-2024?",
     "change", None, D(2024, 8, 1), D(2024, 8, 20), ()),
    ("Has the river near Gaya changed since Aug 1, 2024?",
     "change", "Gaya", D(2024, 8, 1), None, ()),
    ("How has built-up area in Pune increased from 5 March 2020 to 5 March 2024?",
     "change", "Pune", D(2020, 3, 5), D(2024, 3, 5), ()),
    ("Did the lake shrink between 1 January 2023 and 1 January 2024 in Bhopal?",
     "change", "Bhopal", D(2023, 1, 1), D(2024, 1, 1), ()),
    ("What changed after 20 Aug 2024?", "change", None, None, D(2024, 8, 20), ()),
    ("Compare 1 Jan 2020 and 1 Jan 2024", "change", None, D(2020, 1, 1), D(2024, 1, 1), ()),
    ("What changed from 2025 to 2026?", "change", None, None, None, ()),
    ("Has built-up area increased?", "change", None, None, None, ()),
    ("What changed between these images?", "change", None, None, None, ()),
    ("Where has the forest decreased?", "change", None, None, None, ()),
    ("Has the image changed color?", "change", None, None, None, ()),
    # locate, including the three documented keyword-planner misroutes.
    ("Find all aircraft", "locate", None, None, None, ()),
    ("Show the ships", "locate", None, None, None, ()),
    ("Detect oil spills in this radar image", "locate", None, None, None, ()),
    ("Where is the building?", "locate", None, None, None, ()),
    ("Locate the road.", "locate", None, None, None, ()),
    ("Highlight flooding.", "locate", None, None, None, ()),
    ("Count the boats near Kochi harbour", "locate", "Kochi", None, None, ()),
    ("Can you find the airport in Mumbai?", "locate", "Mumbai", None, None, ()),
    ("Mark the bridges over the Ganga", "locate", "Ganga", None, None, ()),
    ("Are buildings concentrated in the north?", "locate", None, None, None, ()),
    ("Give me the bounding box of the bridge.", "locate", None, None, None, ()),
    ("Which part of the image is flooded?", "locate", None, None, None, ()),
    # describe: yes/no and what-is questions about one scene.
    ("Is there a building in this image?", "describe", None, None, None, ()),
    ("What is visible?", "describe", None, None, None, ()),
    ("Describe what is visible.", "describe", None, None, None, ()),
    ("How many large buildings are visible?", "describe", None, None, None, ()),
    ("Is flooding visible?", "describe", None, None, None, ()),
    ("Is SAR visible in this screenshot?", "describe", None, None, None, ()),
    ("Does the image show a bounding box already drawn?", "describe", None, None, None, ()),
    ("Is it somewhere near water?", "describe", None, None, None, ()),
    ("Was it cloudy over Chennai on 5 Dec 2023?", "describe", "Chennai", None, D(2023, 12, 5), ()),
    # optical_sar: both sources named together, same rule as the planner.
    ("Compare the optical and SAR images.", "optical_sar", None, None, None, ()),
    ("Analyze Sentinel-1 and Sentinel-2 together over Kerala", "optical_sar", "Kerala", None, None, ()),
    ("What does SAR show that the optical image misses?", "optical_sar", None, None, None, ()),
    ("Use Sentinel-1 and Sentinel-2 to map flooding in Assam on 20 June 2022",
     "optical_sar", "Assam", None, D(2022, 6, 20), ()),
    # Place rule: sensor and month words are skipped, suffixes stripped, case kept.
    ("Which villages in Sentinel-1 imagery flooded near Muzaffarpur District on 20 Aug 2024?",
     "flood_change", "Muzaffarpur", None, D(2024, 8, 20), ("before_date",)),
    ("Which villages flooded in August 2024 near Patna?",
     "flood_change", "Patna", None, None, ("before_date", "after_date")),
    ("Which areas near New Delhi were flooded on 13 July 2023?",
     "flood_change", "New Delhi", None, D(2023, 7, 13), ("before_date",)),
    ("which villages near patna flooded on 20 aug 2024 that weren't flooded on 1 aug?",
     "flood_change", None, D(2024, 8, 1), D(2024, 8, 20), ("place",)),
    ("In Patna, which villages flooded on 20 Aug 2024?",
     "flood_change", "Patna", None, D(2024, 8, 20), ("before_date",)),
    # unknown: nothing fired; the keyword planner decides.
    ("Invent an answer", "unknown", None, None, None, ()),
    ("Question", "unknown", None, None, None, ()),
    ("Whereas roads are visible, is there water?", "unknown", None, None, None, ()),
]


@pytest.mark.parametrize(("text", "intent", "place", "before", "after", "missing"), CASES)
def test_parse_question(text, intent, place, before, after, missing) -> None:
    parsed = parse_question(text)
    assert (parsed.intent, parsed.place, parsed.before, parsed.after, parsed.missing) == (
        intent, place, before, after, missing,
    )
    assert parsed.text == text


def test_table_covers_every_intent() -> None:
    assert {case[1] for case in CASES} == set(INTENTS)
    assert len(CASES) >= 40


def test_demo_question_reports_the_rules_that_fired() -> None:
    parsed = parse_question(CASES[0][0])
    assert parsed.rules == (
        "intent_flood_change",
        "date_day_month",
        "date_year_from_question",
        "place_after_preposition",
    )


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("What changed between 2024-08-01 and 2024-08-20?", "date_iso"),
        ("What changed between 01/08/2024 and 20/08/2024?", "date_day_first_numeric"),
        ("What changed between Aug 1, 2024 and Aug 20, 2024?", "date_month_day"),
        ("Which villages near Patna flooded on 31 Sep 2024?", "date_invalid"),
        ("Was it cloudy on 5 Dec?", "date_year_missing"),
        ("Which villages near Patna flooded since 1 Aug 2024?", "date_since_or_before"),
        ("Which villages near Muzaffarpur District flooded?", "place_suffix_stripped"),
    ],
)
def test_rule_ids_name_what_fired(text: str, rule: str) -> None:
    assert rule in parse_question(text).rules


def test_unknown_fires_no_intent_rule() -> None:
    assert parse_question("Question").rules == ()


def test_missing_follows_the_declared_order() -> None:
    order = ("place", "before_date", "after_date", "year")
    for case in CASES:
        missing = case[5]
        assert list(missing) == sorted(missing, key=order.index)


def test_parsed_question_is_frozen() -> None:
    parsed = parse_question("Show the ships")
    assert isinstance(parsed, ParsedQuestion)
    with pytest.raises(FrozenInstanceError):
        parsed.intent = "describe"  # type: ignore[misc]


def test_parsing_is_deterministic() -> None:
    text = CASES[0][0]
    assert parse_question(text) == parse_question(text)


def test_blank_text_is_unknown() -> None:
    parsed = parse_question("   ")
    assert (parsed.intent, parsed.place, parsed.missing, parsed.rules) == ("unknown", None, (), ())
