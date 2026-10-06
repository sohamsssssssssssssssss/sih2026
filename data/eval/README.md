# Question routing evaluation set

`questions.v1.jsonl` holds 110 questions phrased the way real users ask them: district disaster officers, students, journalists and hackathon judges, in Indian English. It measures the Phase 3 target in `docs/research/claude-handoff-2026-10-05.md`: at least 90% routing accuracy on 100 real-phrasing questions. `orchestrator/test_question_eval.py` checks the file and scores routing.

The set was **written blind to the question parser**, which was being built in parallel and never consulted. The planner's keyword rules were not used to choose or word any question either. Labels follow the parser's output contract only. After the set was frozen, the current keyword planner routed 44 of 110 (0.40); no question was changed after that run.

## Format

One JSON object per line:

```json
{"id": "q001", "question": "...", "expected": {"capability": "change_vqa", "intent": "flood_change", "place": "Patna", "before": null, "after": null, "missing": ["year"]}, "category": "flood", "notes": "..."}
```

| category | count | what it covers |
|---|---|---|
| flood | 37 | Bihar, Assam, Kerala, Odisha, Mumbai, Chennai, Delhi, Gujarat, Andhra, Himachal; with and without dates, years and places |
| change | 16 | non-flood temporal change between two scenes |
| locate | 21 | find, show, detect, mark, box, highlight, count-and-mark |
| describe | 21 | yes/no, plain counts, land cover |
| optical_sar | 6 | questions that need the optical and SAR scenes together |
| ambiguous | 9 | underspecified or out-of-scope; the right output has missing fields or intent `unknown` |

Six questions (about 5%) contain typos, one is Hinglish, and several are long or indirect.

## Label conventions

**Intent to capability.** `flood_change` and `change` go to `change_vqa`, `locate` to `grounding`, `describe` to `single_image_vqa`, and `optical_sar` to `optical_sar`. `unknown` is labelled `single_image_vqa` because the planner has no abstain route and that is its default. The answer layer, not routing, has to decline those questions.

**Place.** The place is the name as the user wrote it, case included, with "near", "district" and a leading "the" dropped. When a state and a town both appear, the place is the town the user wants mapped (q015 Puri, q035 Chalakudy). Rivers, streams and cyclone names are not places (q023, q024, q025). Questions about an uploaded image ("the port", "this area") have no place.

**Dates.**
- Numeric dates are day-first, the Indian convention: `05/06/2022` is 5 June.
- A date without a year takes the question's year if exactly one distinct year is stated, even if that year is written twice (q031). Otherwise the date is unresolved: it stays `null` and `year` goes into `missing`. The current year is never assumed (q001, q103, q108).
- `before` is the earlier of two dates, whatever order they are written in, or the date in "since X" / "before X". `after` is the later date, or the single date in "flooded on X".
- Month-and-year ("August 2018"), bare years ("since 2020"), periods ("early July") and relative phrases ("this year", "last decade") are not dates. Both fields stay `null`.

**Missing.** Only `flood_change` needs fields: place, before and after. A field that is absent from the text is listed as `place`, `before_date` or `after_date`. A date that is written but cannot get a year counts as present, so it is reported as `year` rather than `before_date` or `after_date`. For every other intent, `missing` is empty; no non-flood question has a yearless date. The schema test enforces this.

## How the ambiguous cases were decided

- **Describe or locate.** Asking whether something is present, or asking for a plain count ("How many buildings are there?", "Can you see any ships?"), is `describe`. Asking to find, show, mark, box, outline or highlight instances is `locate`, and that includes counting by marking (q060, q070) and "Show the ships".
- **Flood words on one image.** "Is the area flooded?" and "standing water on the fields" ask about one scene, so they are `describe`. "Highlight the flooded fields" is `locate`. `flood_change` needs two times.
- **Flood change or optical-SAR.** Naming optical and Sentinel-1 only to explain why SAR is used is still `flood_change` (q029). Asking to use or compare the two sensors at one time is `optical_sar` (q101). A single radar image ("Detect oil spills in this radar image") is `locate`.
- **Change then localise.** "Where did flooding increase…" and "Where has the forest decreased?" are labelled with their change intent. Their primary capability is `change_vqa`.
- **Vague comparisons.** "compare these two" and "Is it worse now than before?" are `change`. A comparison between sensors would have to name them.
- **Nothing to do.** "hi", "What about Assam?", "Can you check Patna for me?" and a weather forecast are `unknown`. The place is still recorded when one is named.

## Real events

Questions that name a real flood use real places and plausible dates. The Kosi 2024 (Kiratpur, Ghanshyampur, Darbhanga), Silchar 2022 and Aluva 2018 questions use dates inside the windows in `data/manifests/flood_events.v1.json`. Every other question is a user phrasing, not a claim that a flood happened on those dates. `notes` explains the labelling and asserts no flood facts beyond that manifest.

## Running

```bash
PYTHONPATH=. python -m pytest orchestrator/test_question_eval.py -q -s
```

- **Schema test:** always runs.
- **Accuracy test:** skipped until `orchestrator.question` exists. It plans each question with `plan_request(PlanRequest(question=..., scene_ids=("a", "b")))` and asserts at least 90% capability accuracy, printing the misrouted ids.
- **Field accuracy:** for intent, place, before, after and missing, informational only. It is measured when that module has a `parse_question` or `parse` function returning the contract's fields.

Once a reported number depends on this file, don't relabel it in place: add `questions.v2.jsonl`.

## Results so far (read before quoting a number)

| Router | Routing accuracy on v1 | Blind? |
|---|---|---|
| Keyword planner (`phase0-rules-v1`) | 44/110 (40%) | yes |
| First rules-first parser (`phase3-parsed-v1`, commit 7a4714c) | 85/110 (77.3%) | yes |
| Parser after fixing the v1 misroutes | 110/110 (100%) | **no** |

The fixes after the blind run were general rules, each covered by parser unit tests worded differently from this set:
- flood events named by place, year or "did ... flood", excluding "this image";
- comparison words such as "compared with", "since 2020" and "reduced";
- more locate verbs;
- jointly requested sensors.

Still, they were chosen by looking at v1's misroutes, so 100% overstates real accuracy. The honest estimate is the blind 77.3%, until a new set written without seeing `orchestrator/question.py` is scored once.
