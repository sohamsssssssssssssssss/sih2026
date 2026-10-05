# SatQuery AI — Science & Evaluation Plan (Phases 3, 4, 5, 7, 8)

**Track A (Soham) · backend/science · 13 Sept 2026 → SIH finale 8–15 Dec 2026**
Written against repo `/Users/atharva/SIH` @ `93b21b0`. Read-only analysis; no repo files modified.

---

## 0. What the repo actually is today (findings, not assumptions)

This section is load-bearing. Every phase below is designed around these facts.

### 0.1 The eval harness — mature in rigor, single-metric in capability

`eval/eval.py` is genuinely good. It records `git_sha`, `gpu`, `timestamp`, full `config`,
every per-sample prediction, and archives a second copy to `results/{model}__{suite}__{stamp}Z.json`
with the comment *"a number that exists only in a Kaggle session does not exist."* It carries
three real scientific guards:

| Guard | Constant | What it catches |
|---|---|---|
| Two-sided degeneracy | `DEGENERATE_YES_RATE=0.85` / `DEGENERATE_NO_RATE=0.15` | Answer collapse to one token |
| Open-sample floor | `MIN_OPEN_SAMPLES=500` | `open_accuracy` quoted at unusable *n* |
| Per-rung degeneracy | `MIN_BINARY_FOR_RUNG_GUARD=30` | Collapse hidden by dilution across rungs |

**But it scores exactly one thing.** The whole scoring surface is:

```python
is_correct = answer_matches(prediction["answer"], sample["expected_answer"])
```

`answer_matches` is string/regex/number-word matching. There is **no IoU, no polygon metric, no
per-condition stratification, no calibration, no metric dispatch**. The sample contract is
`{image_paths, question, expected_answer}` (+ optional `gsd`, `tile_id` on the ladder).

Every capability I own in Phases 3, 4, 5 produces something that is **not a string**: a box, a
polygon, a three-condition contrast, a fused decision, a confidence. **This is the single biggest
science risk in the plan** — see §9.

### 0.2 The capability registry already *is* the floor mechanism

`orchestrator/capabilities.py` declares the exact vocabulary for my phases:

```python
IMPLEMENTED_CAPABILITIES = frozenset({SINGLE_IMAGE_VQA})
UNAVAILABLE_CAPABILITIES = (GROUNDING, CHANGE_VQA, OPTICAL_SAR)
```

`Provider.__post_init__` **raises** if a provider advertises a capability not in
`IMPLEMENTED_CAPABILITIES`. `resolve_provider` raises `CapabilityUnavailable`, which the API surfaces
truthfully via `/api/capabilities` (`{"name": "grounding", "available": false, "provider": null}`).
`orchestrator/planner.py` and `execution_plan.py` already plan *through* these capabilities and mark
plans non-executable when providers are missing.

**Implication: I do not need to build a floor mechanism. It exists.** Each of Phases 3/4/5 is
literally "move one string from `UNAVAILABLE_CAPABILITIES` into `IMPLEMENTED_CAPABILITIES` and
register a `Provider` — but only after its eval gate passes." Failing a gate = the string stays
where it is, the system keeps running, the UI truthfully says the capability is unavailable.
That is the floor rule, already enforced by code that raises.

### 0.3 `evidence` is `[]` everywhere. Always. In every code path.

Grepped the whole repo: `models/mock` returns `"evidence": []`, `models/qwen_vl` returns
`"evidence": []`, every test fixture returns `"evidence": []`. `router._validate_result` only checks
`isinstance(result["evidence"], list)`.

The "Execution evidence" panel in the frontend and the evidence cards in `demo_gui/app.py` render
**execution provenance** — trace `record_hash`, `prev_hash`, `model_version`, `execution_mode` —
not spatial evidence. That provenance work is genuinely good and is a real differentiator. But
**zero spatial evidence exists today.** Phase 3 introduces the concept from nothing.

### 0.4 `confidence` is a hardcoded lie in the only real model

```python
# models/qwen_vl/model.py
"confidence": 1.0,   # "Placeholder only: frozen generation does not provide calibrated confidence."
```

Mock returns `0.5`. Frontend renders the literal string `"Confidence calibration pending"`
(`frontend/components/analysis/AnalysisResult.tsx:10`, `demo_gui/app.py:369`, asserted in two
golden-path tests). See §8 — under a **zero-fabrication** requirement, shipping `confidence: 1.0`
is itself the fabrication.

### 0.5 Compute reality is worse than "no CUDA"

`QwenVLModel._load()` contains:

```python
if not torch.cuda.is_available():
    raise RuntimeError("Qwen2.5-VL inference requires a CUDA GPU for this baseline; use the Kaggle T4 runner")
```

I verified `backend/.venv/bin/python`: **torch 2.14.0, `torch.backends.mps.is_available() == True`**,
transformers 5.16.1, Python 3.13. The M5 Mac *can* run Qwen2.5-VL-3B on MPS. It is blocked by an
explicit hardcoded CUDA check, not by hardware.

**This means Track A currently has zero local science iteration loop.** Every experiment is gated on
Kaggle or a GPU that has not been rented. Fixing this is a ~5-line change (`device = "cuda" if
available else "mps" if available else raise`) and it is **the highest-leverage 30 minutes in this
entire document.** It is scheduled in Week 0 (§1).

Also installed: `rasterio 1.5.1`, `shapely 2.1.2`, `scipy`, `gdown`, `accelerate`.
**Not** installed: `peft`, `datasets`, `bitsandbytes`, `trl` — all needed for Phase 8.

### 0.6 What is actually on disk

| Path | Size | Contents |
|---|---|---|
| `data/raw/loveda/` | **12 GB** | 8,382 PNGs, Train+Val × Urban+Rural, images_png + masks_png, 0.3 m GSD |
| `data/raw/rsvqa_lr/` | 291 MB | Full official RSVQA-LR release — **including `LR_split_train_*` and `LR_split_val_*`** |
| `data/ladder/` | 383 MB | 200 sources × 5 rungs (0.3/1/2/5/10 m) + `manifest.jsonl` |
| `data/sar_gate/` | 52 KB | **Scripts and annotations only.** `jobs.json`, `raw/`, `rendered/` are gitignored and absent |
| `data/runtime/scenes/` | 1.8 MB | One uploaded scene |

**No DOTA. No SAR rasters. No change data. No grounding data.**

Two things this makes possible that the team may not have noticed:

1. **RSVQA-LR train + val splits are already downloaded.** `download_rsvqa_lr()` fetches all 12
   files. That is a zero-acquisition-risk fine-tuning corpus (Phase 8) and a zero-acquisition-risk
   calibration set (§8) sitting on disk right now.
2. **LoveDA ships per-pixel semantic masks**, and `eval/ladder.py:label_answers()` already derives
   gold VQA answers from them. The same masks yield **connected-component bounding boxes and
   polygons** — real grounding targets, zero download. This is the Phase 3 guaranteed floor.

### 0.7 The SAR half of Phase 5 is already de-risked; the *optical* half is not

`data/sar_gate/order_scenes.py` + `process_scenes.py` are a **working, previously-executed**
Sentinel-1 pipeline: `asf_search` geo-search for IW/GRD_HD/VV+VH → HyP3 `submit_rtc_job`
(gamma0, 30 m, power scale, no speckle filter) → Lee 7×7 → dB → fixed-scale VV/VH/VV−VH false
colour. `data/sar_gate/annotation_template.md` contains **completed manual annotations citing real
HyP3 job IDs** (e.g. `71cf874e-4303-4e1f-9ab7-74037b1956c9`) across five Indian AOIs.

The AOIs are already chosen and committed:
`mumbai_coastal (19.05, 72.85)`, `maharashtra_farmland (19.9, 75.3)`,
`western_ghats_forest (17.9, 73.5)`, `konkan_coast (16.7, 73.3)`,
and one of `{Kutch (23.3, 70.3), MP (23.5, 78.5), Rajasthan (27.0, 75.5)}`.

**So Phase 5's unknown was never "can we get SAR."** It is **"can we get a co-located,
temporally-close, low-cloud *optical* scene for the same AOIs."** That is a metadata question
answerable by a STAC query in an afternoon — hence the September spike (§6).

### 0.8 A correction to `eval/suites/RECON.md`

RECON.md states: `cdvqa | BLOCKED: not on the HF Hub at all`, and calls it *"a track blocker, not a
missing suite."*

That is a search-scope error, not a true blocker. **CDVQA is published on GitHub at
`github.com/YZHJessica/CDVQA`** — 2,968 bi-temporal 512×512 pairs from the public subset of SECOND,
with >122,000 generated QA pairs. Absence from the HuggingFace Hub was mistaken for absence from
the world. Phase 4 is not blocked. (Verify the repo resolves on day 1 of Phase 4 before relying on
it — this is a secondary option, not the spine; see §3.)

---

## 1. Week 0 — prep before Phase 3 (14 Sept – 5 Oct, fits alongside Phases 1–2)

Five items. Total ≈ 2.5 days of Track A time spread over three weeks. Everything downstream
assumes these are done.

| # | Item | Effort | Why it cannot wait |
|---|---|---|---|
| W0.1 | **MPS device support in `QwenVLModel._load()`** | 30 min | Unlocks the entire local dev loop. Without it Phase 3 development is blind. |
| W0.2 | **Metric dispatch in `eval/eval.py`** (§9) | 1 day | Phase 3's deliverable is "reported accuracy" and there is no scorer for a box. |
| W0.3 | **Phase 5 data-availability spike** (§6) | 1 day, Sat 19 Sept | Kills or confirms a whole phase 6 weeks early. |
| W0.4 | **Rent the GPU / confirm the vendor + image** | 2 h | "Rented GPU" is currently a noun with no invoice. Phase 8 is non-negotiable. |
| W0.5 | **`pip install peft datasets`** into `backend/.venv`, pin versions | 30 min | transformers 5.16.1 is very new; peft compat must be proven before 1 Dec. |

**W0.1 detail.** Replace the hard CUDA raise with device selection + honest trace metadata:
`cuda` → `mps` → raise. `eval.py:gpu_name()` must report `"mps"` (it currently returns `"cpu"` for
MPS, which would silently mislabel every local run). fp16 on MPS is unreliable for some ops — use
fp32 on MPS and **record the dtype in the results JSON**, because an fp32-MPS number and an
fp16-CUDA number are not the same measurement and the repo's own standards demand that be visible.

**Expectation management on W0.1:** Qwen2.5-VL-3B fp32 on an M5 will be roughly 3–10 s/sample. That
is useless for a 10,004-sample run and *perfect* for a 50-sample "does my box parser work" loop.
That is exactly what Phase 3 development needs.

---

## 2. Phase 3 — Grounding (Tue 6 Oct – Mon 19 Oct, 2 weeks)

**Deliverable:** a real model/tool producing typed spatial evidence (bounding box minimum, polygon
where cheap), a held-out eval harness, reported Acc@0.5 IoU, and an explicit no-evidence state.

### 2.1 Dataset

**Primary (guaranteed, zero download): LoveDA-derived grounding set — `ground_loveda`.**

LoveDA masks are already on disk (12 GB, 8,382 tiles) with 7 classes. Derive referring targets by
connected-component analysis on the mask:

- `building` (class 2) → largest component → bbox + simplified polygon
- `water` (class 4) → largest component → bbox + polygon
- `forest` (6), `agriculture` (7) → region bbox

Question templates: *"Where is the largest building in this image?"*,
*"Where is the water body?"*, *"Locate the largest agricultural field."*

Gold = the component bbox in pixel coordinates. Held out by **source tile**, taken from LoveDA
`Val/` only (`Train/` is reserved so Phase 8 can never leak).

Honest limitation, to be stated in the deck: this is **region grounding, not free-form referring
expression grounding.** The language is templated, not natural. It proves the *evidence pipeline*
(model → typed box → IoU → UI overlay) is real; it does not prove open-vocabulary grounding.

**Upgrade (attempt in parallel, do not depend on): DIOR-RSVG.**
17,402 images / 38,320 language expressions over 20 object categories; expression splits
26,991 / 3,829 / 7,500 (train/val/test); annotations are per-expression XML with bounding boxes,
images are JPG. Average expression length 7.47 tokens — genuinely natural language.
**Risk: distributed via Google Drive.** This repo has *already been burned* by exactly this —
`data/download_ladder_data.py:acquire_dota()` falls back to a manual-download instruction because
`gdown --folder` cannot get past Drive confirmation, and DOTA is consequently not on disk. Assume
DIOR-RSVG will need one human with a browser. **Size uncertain — I did not verify the archive size;
DIOR imagery is 800×800 JPG × 23k images, so plausibly 20–30 GB, but do not quote that number until
measured.** Licence: DIOR is research-use; treat as academic/non-commercial until verified.

**Second fallback: RSVG (Sun et al. 2022)** — the smaller predecessor set. Only if DIOR-RSVG fails
*and* the LoveDA floor somehow proves inadequate. Do not plan around it.

### 2.2 Model / approach

**Primary: Qwen2.5-VL-3B — the model already in the registry.**

Qwen2.5-VL was trained for grounding and emits boxes natively in **absolute pixel coordinates**
(unlike Qwen2-VL's 0–1000 normalised scheme — getting this backwards produces boxes that are
silently wrong by a scale factor, and silently-wrong is the worst failure mode under a
zero-fabrication rule). Prompt for JSON output:
`'Output the bounding box of the largest building as JSON: [{"bbox_2d": [x1,y1,x2,y2], "label": "building"}]. If there is none, output [].'`

**No new model. No new checkpoint. No training.** This is the correct first rung: the capability may
already exist in weights we have measured.

**Fallback if Qwen grounding measures below ~0.25 Acc@0.5: GroundingDINO (Swin-T, ~172 M params,
Apache-2.0)** as a *tool* the orchestrator calls, with Qwen doing the language→class mapping. Runs
on MPS/CPU. This also better matches "a real model/**tool** producing typed spatial evidence" and
keeps the multi-agent story honest — the VLM delegates to a detector.

Decision point: **Fri 9 Oct** (day 4), on the 50-sample MPS pilot.

### 2.3 Typed evidence contract

New, minimal, additive — does not break `router._validate_result` (which only checks `evidence` is
a list and passes unknown keys through via `{**result}`):

```python
# evidence item
{"type": "bounding_box", "coords": [x1, y1, x2, y2], "frame": "pixel",
 "image_index": 0, "source": "qwen2.5vl-3b", "score": null}
{"type": "polygon", "coords": [[x, y], ...], "frame": "pixel", "image_index": 0, ...}

# sibling keys on the response
"evidence_status": "grounded" | "no_evidence" | "unsupported"
"evidence_reason": "model returned no parsable box" | null
```

`frame: "pixel"` is mandatory now so that adding `frame: "geo"` (lon/lat via rasterio) later is a
value change, not a schema migration.

**Validation at the boundary (non-negotiable, per repo standards):** reject any box where
`x1>=x2 or y1>=y2`, any coordinate outside image bounds, any NaN. A malformed box is
`no_evidence` with a reason — **never** a clipped-to-fit box. Clipping a hallucinated box into
validity is fabrication with extra steps.

### 2.4 Evaluation harness

Plugs into `eval/eval.py` via the metric dispatch from W0.2 (§9).

| | |
|---|---|
| **Primary metric** | `acc_at_50` — fraction with IoU(pred, gold) ≥ 0.5. Standard for RSVG; comparable to published numbers. |
| **Secondary** | `mean_iou` over attempted samples; `acc_at_25` (partial credit, shows the trend before the threshold flips) |
| **Guard metric** | `no_evidence_rate` — fraction where the model produced nothing parsable |
| **Degeneracy guard** | `mean_box_area_fraction`. A model that returns the full image frame scores IoU ≈ the object's area fraction "for free." **Flag if >0.60 of samples have box area >50% of image area.** This is the grounding analogue of the always-yes guard and it is exactly the failure the existing `degenerate()` philosophy is built to catch. |
| **Held-out split** | LoveDA `Val/` source tiles only, never `Train/`. Fixed seed, manifest hashed into the results JSON. |
| **Sample size** | **n = 500.** For a proportion near 0.5, ±5 pp at 95% confidence needs n ≈ 384; 500 gives headroom and round numbers. Report **Wilson 95% CI on every accuracy**, not a bare point estimate. |

`no_evidence_rate` must be reported **alongside** `acc_at_50`, never folded into it. A model that
refuses on 70% of samples and nails the remaining 30% has `acc_at_50` computed over attempts of
0.9 — which is a lie unless the refusal rate sits next to it. Compute both: `acc_at_50_over_all`
(refusals count as wrong) is the headline; `acc_at_50_over_attempted` is the diagnostic.

### 2.5 Failure / fallback state (the floor)

Three levels, all already supported by existing code:

1. **Per-sample:** no parsable box → `evidence: []`, `evidence_status: "no_evidence"`,
   `evidence_reason: "..."`. The text answer still returns. UI renders "no spatial evidence
   located" — never an empty overlay implying a null result, never a guessed box.
2. **Per-capability:** if the Phase 3 gate fails, `GROUNDING` **stays in
   `UNAVAILABLE_CAPABILITIES`**. `/api/capabilities` reports `available: false`. The planner already
   produces a non-executable plan with `unavailable_reason`. Single-image VQA is untouched.
   **Phase 2's floor is fully intact.**
3. **Demo:** the grounding tab shows the measured Acc@0.5 with its CI and the honest
   `no_evidence_rate`. A weak-but-measured number is a stronger demo artifact than a silent
   feature — and it is the only kind this project is allowed to show.

### 2.6 Day by day

| Day | Date | Work |
|---|---|---|
| 1 | Tue 6 Oct | Write `data/grounding_loveda.py`: mask → connected components (scipy.ndimage.label, already installed) → bbox + shapely-simplified polygon → `manifest.jsonl` with the ladder's schema. Kick off DIOR-RSVG download in background. |
| 2 | Wed 7 Oct | Generate 500-sample held-out set from LoveDA `Val/`. **Eyeball 20 overlays.** A grounding gold set nobody looked at is not a gold set. |
| 3 | Thu 8 Oct | `iou()` + `score_grounding()` in `eval/metrics.py`; wire into the W0.2 dispatch. Unit-test IoU against hand-computed cases including zero-overlap and containment. |
| 4 | Fri 9 Oct | 50-sample Qwen-on-MPS pilot. Parse boxes. **DECISION: Qwen vs GroundingDINO.** |
| 5–6 | Sat–Sun 11 Oct | Box parser + boundary validation + `evidence_status`. Implement whichever path day 4 chose. |
| 7 | Mon 12 Oct | Full 500-sample run on rented GPU (or Kaggle). First reported Acc@0.5 + CI. |
| 8 | Tue 13 Oct | Read the failure cases. Prompt iteration (one variable at a time; the repo's `RESCORE_NOTE.md` is the standard for documenting any scorer change). |
| 9 | Wed 14 Oct | Second 500-sample run. If DIOR-RSVG landed, generate its 500-sample test subset too. |
| 10 | Thu 15 Oct | DIOR-RSVG run if available. This is the *natural-language* grounding number and it is the better headline if it exists. |
| 11 | Fri 16 Oct | **GATE.** Register `GroundingProvider`, move `GROUNDING` into `IMPLEMENTED_CAPABILITIES` — only if the gate passes. |
| 12 | Sat 17 Oct | Hand the typed evidence schema to Track B for overlay rendering. Commit the results JSON. |
| 13–14 | Sun–Mon 19 Oct | Buffer. Write the phase's honest one-pager: what was measured, on what, with what CI, what the no-evidence rate was. |

### 2.7 Kill criterion

**By EOD Wed 14 Oct (day 9): if `acc_at_50 < 0.20` on the LoveDA held-out set with both Qwen and
GroundingDINO tried, stop.**

Rationale for 0.20: template questions on the single largest component of a class is an *easy*
grounding task. Below 0.20 the pipeline is not producing usable spatial evidence and two more days
will not change that.

**Fallback on kill:** ship grounding as **retrieval-only evidence, honestly labelled.** Return
`evidence_status: "unsupported"` with the mask-derived region as a *dataset annotation*, explicitly
marked `"source": "loveda_ground_truth_mask"` and **never** attributed to the model. `GROUNDING`
stays unavailable in the capability registry. Phase 4 starts on time with the Phase 2 floor intact.

---

## 3. Phase 4 — Bi-temporal Change-VQA (Tue 20 Oct – Mon 2 Nov, 2 weeks)

**Deliverable:** explicit t1/t2 scene-pair contract, a single-frame control proving change detection
isn't noise, evaluation.

### 3.1 Dataset

**Primary: LEVIR-CD + LEVIR-CC.**

- **LEVIR-CD**: 637 bi-temporal pairs, 1024×1024, **0.5 m/px** Google Earth, binary building-change
  masks. Small download (hundreds of MB).
- **LEVIR-CC**: the captioned extension — **10,077 bi-temporal image pairs, 50,385 sentences**,
  **2.5 GB**, GitHub `Chen-Yang-Liu/LEVIR-CC-Dataset`.
- **Licence: academic / research use only, commercial use prohibited** (inherited from LEVIR-CD).
  Fine for SIH, must be stated in the deck, and must go into `RECON.md` next to LoveDA's identical
  restriction.

**Derive the QA gold from the change masks, exactly as `eval/ladder.py:label_answers()` already
does for LoveDA.** This is the key design decision and it is what removes the acquisition risk:

- *"Did any buildings appear between these two images?"* → `yes` if changed-pixel count > threshold
- *"Did any buildings disappear?"* → from the directional mask where available
- *"How many separate areas changed?"* → connected components on the change mask (open-ended,
  integer answer — `answer_matches` already handles number-words → digits)
- *"What kind of change occurred?"* → `construction` / `demolition` / `none` (open-ended)

This gives a **binary + open-ended split**, which means the existing `summarise()`, the two-sided
degeneracy guard, and `open_accuracy` as headline metric **all work unchanged.** Zero new metric
machinery for the accuracy half.

LEVIR-CC's natural-language captions are the **upgrade path**, not the spine — use them to
cross-check that derived gold answers agree with human captions on a 50-pair sample. If they
disagree, the derivation threshold is wrong, and that check costs an hour.

**Fallback: CDVQA** — `github.com/YZHJessica/CDVQA`, 2,968 pairs at 512×512 from the public subset
of SECOND, >122,000 generated QA pairs. Multi-class semantic change (not just buildings), so richer
questions. **Correcting `RECON.md`: this is not blocked — it is on GitHub, just not on the HF Hub.**
Requires obtaining SECOND imagery separately; verify both resolve on day 1 of Phase 4.

**Second fallback (zero acquisition risk): synthetic-pair control on LoveDA.** LoveDA has no
temporal dimension, so this cannot measure real change. Its *only* legitimate use is the
identical-pair null condition below. Never present it as change detection.

### 3.2 Model / approach

**Qwen2.5-VL-3B with two images in one message.** `QwenVLModel._generate_answer()` already builds
`content` as a list comprehension over `image_paths` — **multi-image is already supported by the
existing code** and needs no change. The scene-pair contract is passing
`image_paths=[t1_path, t2_path]` with the order fixed and stated in the prompt
(*"Image 1 was captured before Image 2."*).

`data/dataset.py` already has the field: `optical_t2: NDArray[np.float32] | None`, validated by
`validate_tile_sample`. **The bi-temporal contract is already in the schema.** Phase 4 populates it.

No fine-tuning here. A change-detection head (e.g. BIT, ChangeFormer on LEVIR-CD) is tempting and is
**out of scope** — it would be a second training pipeline competing with Phase 8's mandatory one, on
one GPU, for two people. If Qwen cannot do change VQA zero-shot, that is a finding worth reporting,
not a reason to open a training project in week 7.

### 3.3 Evaluation harness — the three-condition design

This is the scientific core of Phase 4. Identical questions, identical gold, three input conditions:

| Condition | Input | Gold | What it proves |
|---|---|---|---|
| **A — pair** | `[t1, t2]` | derived from change mask | The capability |
| **B — single-frame control** | `[t2]` only | *same gold* | **A ≫ B ⇒ the model is using t1.** A ≈ B ⇒ it is answering from priors and the "change detection" is noise. |
| **C — identical-pair null** | `[t2, t2]` | `no` / `0` / `none` | **Hallucination probe.** Any "yes, buildings appeared" here is pure fabrication with a two-image input. Directly measures the zero-fabrication claim. |

Condition C is cheap, is not in the original brief, and is the single most persuasive number in the
phase. A judge asking *"how do you know it isn't making this up?"* gets a measured hallucination
rate under a provably-no-change input.

**Sample size: n = 400 per condition (1,200 total).** Derivation, stated because the repo's own
docstring standard demands it: to detect a 10-point difference between conditions A and B at 80%
power, α=0.05, two-proportion test, n ≈ 16·p(1−p)/d² = 16·0.25/0.01 = 400 per arm. Anything smaller
cannot distinguish "uses t1" from "doesn't."

**Report the A−B delta with its CI as the headline**, not condition A's accuracy. Condition A alone
is uninterpretable — that is the entire point of the control.

Plugs into `eval.py` as suite `change_vqa` with a `condition` field on each sample; `summarise()`
runs per condition exactly as `per_rung` already runs per GSD. The stratification machinery exists.

### 3.4 Failure / fallback state

- Per-sample: only one scene supplied for a change question → the planner already refuses this
  (`missing_inputs` on `PlanRequest`). Return the plan's `unavailable_reason`. Never silently
  single-frame a change question.
- Per-capability: gate fails → `CHANGE_VQA` stays in `UNAVAILABLE_CAPABILITIES`. Phase 3's grounding
  and Phase 2's VQA are untouched.
- **Special case — the honest partial ship:** if A works but A ≈ B, do **not** register the
  capability. Report it as a *measured negative result*: *"Qwen2.5-VL-3B answers change questions
  at 0.71 with both frames and 0.69 with one frame; it is not using the temporal signal."* That is
  a real scientific finding, it is defensible in front of judges, and shipping it as a working
  feature would be the exact fabrication the governing rule forbids.

### 3.5 Day by day

| Day | Date | Work |
|---|---|---|
| 1 | Tue 20 Oct | Download LEVIR-CD + LEVIR-CC (2.5 GB). Verify CDVQA/SECOND resolve as backup. Record licences in `RECON.md`. |
| 2 | Wed 21 Oct | `data/change_levir.py`: change mask → QA gold. Thresholds for "any change" and connected-component counting. |
| 3 | Thu 22 Oct | Cross-check derived gold vs 50 LEVIR-CC human captions. Fix thresholds. **Evening: write `scripts/finetune_lora.py` for the Oct 25 dry run (§7).** |
| 4 | Fri 23 Oct | Build the three-condition manifest (400 × 3). Wire `condition` stratification into `eval.py`. |
| 5 | **Sat 24 Oct** | **PHASE 8 FINE-TUNING DRY RUN — see §7.** Blocks this day entirely. |
| 6 | Sun 25 Oct | Two-image prompt design. 50-sample pilot on MPS across all three conditions. |
| 7 | Mon 26 Oct | Full 1,200-sample run on rented GPU. |
| 8 | Tue 27 Oct | **Read the A−B delta.** This is the moment the phase is decided. |
| 9 | Wed 28 Oct | Prompt iteration if A ≈ B — most likely fix is making the temporal order explicit and asking for a difference rather than a description. |
| 10 | Thu 29 Oct | Second full run. |
| 11 | Fri 30 Oct | **GATE.** Register the provider if A−B is significant and C's hallucination rate is acceptable. |
| 12 | Sat 31 Oct | Hand the t1/t2 pair contract to Track B. Commit results JSON. |
| 13–14 | Sun–Mon 2 Nov | Buffer + phase one-pager. |

### 3.6 Kill criterion

**By EOD Wed 28 Oct (day 9): if the A−B delta is < 5 points, or its 95% CI includes zero, the
capability does not ship.**

**By EOD Wed 28 Oct: if condition C's false-change rate > 0.30**, the capability does not ship
regardless of A's accuracy — a model that invents change in 1 of 3 identical-image pairs cannot be
demonstrated under a zero-fabrication claim.

**Fallback on kill:** report the negative result in the deck (this is genuinely good content —
"we tested whether it was real and it wasn't" is a rare and credible thing to show). `CHANGE_VQA`
stays unavailable. Roll the saved time straight into Phase 7, which is the headline differentiator
and always benefits from more time.

---

## 4. Phase 5 — Optical-SAR Fusion (Tue 3 Nov – Mon 16 Nov, 2 weeks)

**This is the one phase permitted to be cut.** It is gated on the September spike (§6) and must not
begin at all if that spike returns NO-GO.

### 4.1 Dataset

**Ranked by (ISRO-relevance × obtainability). Chosen by the September spike result.**

**Option 1 — self-acquired S1 + S2 over the five committed Indian AOIs. *Preferred if the spike says GO.***
- SAR: Sentinel-1 IW GRD_HD VV+VH → HyP3 RTC gamma0 @ 30 m. **The pipeline exists and has run**
  (`data/sar_gate/`).
- Optical: Sentinel-2 L2A, same AOI, ±5 days, <20% cloud. Free, global, no licence obstacle
  (Copernicus open licence).
- Best possible ISRO story: real Indian scenes, real dual-sensor, real preprocessing provenance.
- Highest risk — hence §6.

**Option 2 — BigEarthNet-MM / BigEarthNet v2.0.** 590,326 co-registered S1+S2 patch pairs over 10
European countries, **CDLA-Permissive-1.0**. Pre-paired, pre-co-registered, zero acquisition risk.
**Blocker already documented in this repo** (`data/BEN_TXT_FACTS.md`, `RECON.md`): the imagery is a
**118 GB monolithic `.tar.zst`** on Zenodo 10891137 with no per-patch fetch. That will not fit
Kaggle's ~73 GB scratch and is a painful pull on a home connection. Europe-only also weakens the
ISRO argument. **Viable only if someone with fast disk and bandwidth commits to it in early
November — decide by 3 Nov, not later.**

**Option 3 — SEN12MS.** 180,662 triplets of S1 dual-pol + S2 multispectral + MODIS land cover,
256×256, global coverage across four seasons, CC-BY. Distributed as **season-wise tarballs, which
means it can be partially downloaded** — a genuine advantage over BigEarthNet's monolith. I did not
verify the per-tarball sizes; **check before committing, do not quote a number.** Global coverage
means some South Asian scenes plausibly exist — *plausibly*, unverified; do not claim Indian
coverage without querying the metadata.

**Option 4 — SpaceNet 6 (Rotterdam).** Paired SAR + optical at 0.5 m with expert building
footprints, on public AWS S3. Best co-registration and label quality of any option. **Single
European city — the weakest ISRO story.** Use only as a last-resort proof that the fusion *code*
works.

**My recommendation: Option 1 if the spike says GO; Option 3 if not; cut the phase if neither is
secured by Fri 6 Nov.**

### 4.2 Paired-sensor contract & preprocessing provenance

`data/dataset.py` already carries `sar: NDArray[np.float32] | None` validated as `float32[3,H,W]`,
and `Sensor` already includes `"S1"`, `"S2"`, `"risat"`. **The contract exists in the schema.**

What Phase 5 must add is **provenance in `meta`**, because a fusion claim without preprocessing
provenance is unauditable:

```python
"meta": {
  "sar_granule": "S1A_IW_GRDH_...",       "sar_acquisition": "2026-10-14T00:41:12Z",
  "optical_granule": "S2B_MSIL2A_...",    "optical_acquisition": "2026-10-16T05:24:09Z",
  "temporal_gap_days": 2,                  "cloud_cover_pct": 7.3,
  "sar_processing": "hyp3_rtc_gamma0_30m_power_lee7x7",
  "coreg_method": "utm_reproject_nearest", "coreg_residual_px": null,
  "sar_scale": "fixed_db_-25_0"
}
```

`coreg_residual_px: null` is deliberate: **if we cannot measure co-registration residual, we record
null, not a number.** An asserted co-registration quality is exactly the class of claim this project
forbids.

### 4.3 Model / approach — decision-level fusion, not feature-level

**Do not build a fusion architecture.** Two people, one GPU, two weeks, in November. Instead:

Ask the **same question three ways** and let the orchestrator combine at the decision level:
optical-only → answer_o; SAR-only (rendered VV/VH/VV−VH false colour, with
`data/sar_gate/sar_cheatsheet.md` physics injected into the prompt) → answer_s; both images → answer_f.

This is honest ("late fusion"), it costs zero new model code (`image_paths` is already a list), it
produces the three-condition contrast the evaluation needs **for free**, and — critically — the
existing `sar_cheatsheet.md` gives the VLM real backscatter physics as prompt context. That is the
kind of grounded prompt engineering that is legitimately defensible.

The most persuasive demo case is already written down in the repo's own SAR annotations: the Mumbai
scene, where **a runway and calm water both render dark in SAR** and only shape/context
distinguishes them — while optical separates them trivially. That is the fusion argument in one
image, and it is already annotated.

### 4.4 Evaluation harness

Three conditions, same structure as Phase 4:

| Condition | Input | Purpose |
|---|---|---|
| Optical-only | S2 | Baseline |
| SAR-only | S1 RTC render | Does the model read SAR at all, or is it pattern-matching a grey photo? |
| Fused | both | Does the pair beat the better single sensor? |

**Headline metric: `fused_accuracy − max(optical_accuracy, sar_accuracy)`, with CI.** Beating the
*worse* sensor is not fusion. Beating the *mean* is not fusion. Only beating the best single sensor is.

**n = 400 per condition (1,200 total)**, same power derivation as Phase 4.

**The question set must be chosen where SAR genuinely adds information**, or the experiment is
rigged to fail: cloud-obscured optical scenes (SAR sees through cloud — *the* canonical argument),
water/flood extent, and urban-vs-bare-ground discrimination. Say this openly in the deck: the
question set is selected for sensor-complementarity, and that selection is a stated design choice,
not a hidden one.

Gold labels: **ESA WorldCover 10 m (CC-BY-4.0, global, covers India)** as the authoritative label
source — same trick as deriving VQA gold from LoveDA masks, applied to a public global product.

### 4.5 Failure / fallback state

- Per-sample: SAR present, optical missing (cloud) → run optical-only, set
  `"degraded_to": "sar_only"` in the response. State it in the UI. **Never silently substitute.**
- Per-capability: gate fails → `OPTICAL_SAR` stays in `UNAVAILABLE_CAPABILITIES`.
- **Phase cut:** if the spike said NO-GO or the November deadline passes — **the SAR tab still
  ships**, because `backend/routes/sar.py`, `sar_annotation()`, and the five completed human
  annotations already exist. It ships as *"SAR interpretation with expert human annotations and
  documented backscatter physics"* — which is true, is already built, and is a perfectly good demo.
  It just is not *fusion*, and must not be labelled as such.

### 4.6 Day by day (only if the spike says GO)

| Day | Date | Work |
|---|---|---|
| 1 | Tue 3 Nov | **HARD DECISION POINT: Option 1 / 3 / cut.** Submit HyP3 RTC jobs (20–90 min each, run overnight). Start S2 L2A downloads for the same AOIs. |
| 2 | Wed 4 Nov | Co-registration: reproject S1 RTC and S2 to a common UTM grid, chip to 512×512 (`rasterio` is installed). Record every provenance field. |
| 3 | Thu 5 Nov | Visual co-registration check on 10 chips, side by side. If they do not line up, **stop here, not on day 10.** |
| 4 | Fri 6 Nov | **GO/NO-GO #2.** Paired chips exist and are co-registered, or the phase is cut today. |
| 5–6 | Sat–Sun 8 Nov | Derive gold from ESA WorldCover. Build the 1,200-sample three-condition manifest. |
| 7 | Mon 9 Nov | SAR prompt with `sar_cheatsheet.md` physics context. 50-sample pilot. |
| 8 | Tue 10 Nov | Full 1,200-sample run. |
| 9 | Wed 11 Nov | **Read fused − max(single).** Phase decided here. |
| 10 | Thu 12 Nov | Iterate or accept. |
| 11 | Fri 13 Nov | Second run. **GATE.** |
| 12 | Sat 14 Nov | Commit results. Hand paired-scene contract to Track B. |
| 13–14 | Sun–Mon 16 Nov | Buffer. **If cut, this entire window rolls into Phase 7.** |

### 4.7 Kill criteria (three, any one triggers)

1. **Fri 6 Nov (day 4):** fewer than 200 usable co-registered paired chips exist → **cut.**
2. **Wed 11 Nov (day 9):** `fused − max(single)` CI includes zero → do not register the capability;
   report the null result honestly.
3. **Any date:** if Phase 7 preparation is slipping, **cut Phase 5 immediately and without
   debate.** Phase 7 is the headline differentiator; Phase 5 is the designated sacrifice. This is
   pre-agreed so that nobody has to argue for it in week 10 under pressure.

---

## 5. Phase 7 — Resolution/domain-gap ladder + Indian/high-res proxy (Tue 24 Nov – Mon 30 Nov, 1 week)

**The headline differentiator. One week. Two deliverables that must both land.**

### 5.1 Deliverable A — the canonical ladder, re-run across every shipped capability

The ladder methodology already exists and is honest: real LoveDA pixels, only resolution and sensor
noise simulated (Gaussian PSF σ = 0.5·(target/source), area-average BOX decimation, Poisson shot +
Gaussian read noise), `sensor` stays `loveda` and never `synthetic`, native 0.3 m rung is a
zero-degradation anchor. Rungs: 0.3 / 1 / 2 / 5 / 10 m.

Phase 7 extends it along a second axis: **capability × GSD**, not just accuracy × GSD.

| Capability | Ladder metric | Expected shape |
|---|---|---|
| VQA (Phase 2) | `open_accuracy` | Already measured: 0.335 → 0.245 |
| Grounding (Phase 3) | `acc_at_50`, `mean_iou` | Should fall *hard* — small objects vanish first. **This is the most visually compelling curve in the project.** |
| Change VQA (Phase 4) | A−B delta | Should collapse toward zero as blur destroys the temporal signal |
| Fusion (Phase 5, if shipped) | `fused − max(single)` | May *rise* with coarser optical — SAR's relative value grows as optical degrades. A genuinely interesting hypothesis. |

**Two prerequisite fixes, both small, both mandatory:**

1. **Lift the ladder sample cap.** `eval/ladder.py` hard-errors on `--limit > 200`
   (`if not 1 <= args.limit <= 200`). At 200 sources × 2 questions = 400/rung (200 binary, 200 open),
   the **minimum detectable effect is ≈ 13 points at 80% power** — too coarse to measure Phase 8's
   before/after. Regenerate at `--limit 1000` (LoveDA has 8,382 tiles; ~1.9 GB, CPU-bound
   generation). Then MDE ≈ 6 points, which is the resolution Phase 8 needs.
2. **Add `run_id` and `dataset_version` to the results JSON.** `eval.py` currently records
   `git_sha`, `gpu`, `timestamp`, `config` — but **no run_id and no dataset version.** The brief
   requires "every number backed by dated run_id + git SHA + dataset version." Concretely:
   - `run_id = f"{git_sha}-{utc_compact_timestamp}-{uuid4().hex[:6]}"`
   - `dataset_version = sha256(manifest.jsonl)[:16]` for generated suites; for RSVQA, the existing
     per-file MD5 table in `eval/suites/rsvqa.py` is already a dataset version — hash the tuple.
   - `dtype` and `device` (see W0.1 — an fp32/MPS number is not an fp16/CUDA number).
   ~25 lines. Do it in Week 0 (W0.2) alongside the metric dispatch so that **every phase from 3
   onward produces stamped artifacts** and Phase 7 is not retrofitting provenance onto six weeks
   of runs.

### 5.2 Deliverable B — the Indian / high-resolution proxy evaluation (the ISRO argument)

**Honesty first, because this is the section judges will probe hardest.**

**What is a genuine proxy:**

**(i) Self-acquired Sentinel-2 over the five committed Indian AOIs + ESA WorldCover gold. — GENUINE.**
Real Indian geography (Mumbai coastal, Maharashtra farmland, Western Ghats forest, Konkan coast,
one arid inland plain — already committed in `order_scenes.py` and already human-annotated for SAR).
S2 is free, global, 10 m. Gold from ESA WorldCover 10 m (CC-BY-4.0, global, covers India), derived
exactly as `label_answers()` derives gold from LoveDA masks. **Identical question templates to the
ladder**, so the comparison LoveDA-China ↔ S2-India is like-for-like and the delta *is* the measured
domain gap. This is the single strongest, most obtainable ISRO artifact in the plan and it reuses
tooling that already exists. **Build this one.**

**(ii) The GSD ↔ ISRO-sensor mapping table. — GENUINE, and it costs a table.**
The ladder's rungs already correspond to real Indian sensors. Making that correspondence explicit
converts an abstract curve into a direct answer to *"will this work on our data?"*:

| Ladder rung | Indian sensor analogue | Note |
|---|---|---|
| 0.3 m | Cartosat-3 PAN (~0.25 m) | Near-native; ladder anchor |
| 1 m | Cartosat-2S PAN (~0.65 m) | Interpolated between rungs |
| 2 m | Cartosat-2S MX / Cartosat-3 MX | |
| 5 m | Resourcesat-2/2A LISS-IV (5.8 m) | **Closest match in the ladder** |
| 10 m | Sentinel-2 / Resourcesat LISS-III (23.5 m is coarser) | S2 is the exact match |

**State the caveat plainly:** these are *ground-sample-distance* analogues, not sensor simulations.
We do not model Cartosat's PSF, spectral response, or radiometry. The claim is "at the GSD of
LISS-IV, grounding accuracy is X" — not "this was tested on LISS-IV." **Verify every GSD figure
against ISRO/NRSC published specs before printing the table.**

**What is a stretch, and should be labelled as one:**

- **LoveDA as an "Indian" proxy — NO.** LoveDA is Nanjing, Changzhou and Wuhan. Chinese. It is a
  *resolution* proxy and a *land-cover-type* proxy (South/East Asian monsoon urban-rural gradient,
  which is a real and defensible similarity), **not a geography proxy.** Saying otherwise in front
  of ISRO judges is the kind of claim that loses a finale.
- **DOTA — NO, and it is not even on disk.** Google Earth + GF-2 + JL-1 scenes, predominantly
  Chinese. `acquire_dota()` already fails on Google Drive confirmation.
- **xView — UNVERIFIED.** WorldView-3 at 0.3 m, 1,400 km², global coverage; South Asian tiles
  *may* exist. **I did not verify this and neither should you assume it.** CC-BY-NC-SA-4.0.
  Only usable if someone actually queries the tile geography. Do not build the ISRO argument on it.
- **Bhuvan / NRSC open data — UNVERIFIED licence and API.** Ideal on paper. Do not plan around it
  in seven weeks; if someone has spare time in October, a one-hour look is worth it.

**Recommended Phase 7 deliverable B = (i) + (ii).** One real Indian evaluation on free global
imagery with authoritative gold labels, plus one honest sensor-mapping table. Both obtainable, both
defensible, neither overclaimed.

### 5.3 Deliverable C — identify the weak slice for Phase 8

Phase 7's last job is to name, with numbers, **one** (capability × rung × question-type) cell as
the weakest, and hand it to Phase 8 as the fine-tuning target.

**Prediction from data already in the repo:** it will be the **5 m and 10 m rungs on binary
questions**, where the existing baseline shows yes-rates of 0.975 and 1.000 — total answer
collapse. The README already documents this as *"the collapse landing on the base rate."*
Convenient and important: **RSVQA-LR is 10 m Sentinel-2 imagery, and its train split is already on
disk.** The weakest slice and the available adaptation corpus are at the same GSD. That is the
cleanest possible Phase 8 experiment and it needs zero new data.

Do not assume it — measure it on Thu 26 Nov and let the number choose.

### 5.4 Day by day

| Day | Date | Work |
|---|---|---|
| 1 | Tue 24 Nov | Regenerate ladder at `--limit 1000` (raise the cap). Verify `run_id` + `dataset_version` land in the JSON. |
| 2 | Wed 25 Nov | Acquire S2 for the five Indian AOIs + ESA WorldCover tiles. Build the Indian proxy manifest with identical question templates. |
| 3 | Thu 26 Nov | **The big run.** Every shipped capability × 5 rungs on the rented GPU. Budget ~10k VQA samples + grounding + change; this is the GPU-hour peak of the project — book the block in advance. |
| 4 | Fri 27 Nov | Indian proxy run. **Compute the LoveDA↔India domain gap.** Plot both curves with `plot_ladder.py` (it already overlays multiple result files by colour and already draws degenerate rungs hollow). |
| 5 | Sat 28 Nov | **Name the weak slice.** Write the Phase 8 experiment card: exact slice, exact metric, exact expected direction. Freeze it — no moving the target after seeing Phase 8's result. |
| 6 | Sun 29 Nov | Write the ISRO one-pager: ladder curves, GSD↔sensor table, Indian proxy delta, every number stamped with run_id + SHA + dataset version. |
| 7 | Mon 30 Nov | Buffer. Hand plots to Track B. |

### 5.5 Kill criterion

Phase 7 **cannot be killed** — it is the differentiator. It can only be **narrowed**:

- If the big run does not finish by Fri 27 Nov: **drop to three rungs (0.3 / 2 / 10 m)** and report
  three honest points rather than five rushed ones.
- If the Indian proxy S2 acquisition fails by Wed 25 Nov: **ship (ii) the GSD↔sensor mapping table
  alone**, explicitly labelled *"resolution analogue, not tested on Indian sensors."* Weaker, still
  honest, still useful.
- If a capability's ladder run is degenerate at every rung: report it as degenerate. `plot_ladder.py`
  already draws hollow markers with the caption *"not a measurement."* Use it.

---

## 6. ⚡ THE SEPTEMBER SPIKE — Phase 5 data availability (Sat 19 Sept, 1 day)

**The current plan discovers this on day 10 of Phase 5. That wastes ten days of a two-week phase on
a question answerable in an afternoon, seven weeks earlier, at zero cost.**

**Scheduled: Saturday 19 September 2026. Hard deadline: Friday 25 September.**

### 6.1 What this spike is NOT

It is **not** a download. It is **not** a HyP3 job (those take 20–90 min each and burn quota).
It is a **metadata query.** Nothing is fetched but JSON.

### 6.2 Exactly what to run

Query a STAC API for S1 and S2 metadata over the five committed AOIs. Three interchangeable
endpoints — try them in this order, all free:

1. **Microsoft Planetary Computer STAC** — `https://planetarycomputer.microsoft.com/api/stac/v1`,
   collections `sentinel-1-grd` and `sentinel-2-l2a`. No auth for search. *(Note:
   `sentinel-1-rtc` on PC is North-America-only — use `sentinel-1-grd` and RTC it yourself via the
   existing HyP3 path.)*
2. **AWS Earth Search** — `https://earth-search.aws.element84.com/v1`, anonymous.
3. **Copernicus Data Space Ecosystem** OData/STAC — authoritative source, may need a free account.

Pseudocode (write it into the scratchpad, not the repo — read-only constraint):

```python
AOIS = [  # lifted verbatim from data/sar_gate/order_scenes.py
    ("mumbai_coastal",        19.05, 72.85),
    ("maharashtra_farmland",  19.9,  75.3),
    ("western_ghats_forest",  17.9,  73.5),
    ("konkan_coast",          16.7,  73.3),
    ("flat_inland_plain",     23.3,  70.3),   # + MP and Rajasthan alternates
]
WINDOW      = ("2025-09-01", "2026-09-13")   # 12 months back
MAX_GAP     = 5      # days between S1 and S2 acquisition
MAX_CLOUD   = 20     # percent, on the S2 scene

for name, lat, lon in AOIS:
    s1 = search("sentinel-1-grd", point=(lon, lat), datetime=WINDOW,
                query={"sar:instrument_mode": "IW",
                       "sar:polarizations": {"contains": "VH"}})   # dual-pol only
    s2 = search("sentinel-2-l2a", point=(lon, lat), datetime=WINDOW,
                query={"eo:cloud_cover": {"lt": MAX_CLOUD}})
    pairs = [(a, b) for a in s1 for b in s2
             if abs(days(a.datetime, b.datetime)) <= MAX_GAP]
    report(name, len(s1), len(s2), len(pairs),
           distinct_dates={d.date() for a, _ in pairs for d in [a.datetime]})
```

Write the output to `results/spike__optical_sar_pairing__<date>.json` with the same provenance
fields as any other result (`git_sha`, `timestamp`, query parameters). **This spike produces a
committed artifact like every other measurement in this repo.**

### 6.3 The go/no-go signal

| Signal | Condition | Decision |
|---|---|---|
| **GO** | ≥4 of 5 AOIs have ≥3 distinct dates with a valid S1+S2 pair (≤5 d gap, <20% cloud) | Phase 5 runs with Option 1 (self-acquired Indian pairs) |
| **CONDITIONAL** | 2–3 AOIs qualify | Phase 5 runs, but **narrowed to those AOIs only**, and the sample size target drops from 400/condition to whatever the data supports — stated explicitly in the results |
| **NO-GO** | ≤1 AOI qualifies | **Phase 5 does not use self-acquired data.** Decide by 3 Nov between Option 3 (SEN12MS) and cutting the phase. |

**Also record, because it costs one extra field and settles a later argument:** if relaxing to
≤10 days / <35% cloud flips a NO-GO to a GO, note it. A 10-day gap is defensible for land cover
and indefensible for flood extent; knowing the threshold now means the November decision is about
science, not about scrambling.

### 6.4 Secondary spike (same day, +1 hour, high value)

While the credentials are out: **confirm the Earthdata `~/.netrc` entry still authenticates and
that HyP3 has remaining quota.** The SAR annotations cite job IDs from a previous run, so the
account existed. If it has lapsed or the quota is exhausted, that is a second Phase 5 blocker —
and it is far better to learn it in September than on 3 November.

---

## 7. ⚡ THE OCTOBER DRY RUN — Phase 8 fine-tuning rehearsal (Sat 24 Oct, ~6 hours)

**Phase 8 is 4 days in December with a mandatory deliverable. Four days is enough to run an
experiment. It is not enough to run an experiment *and* discover that peft doesn't resolve
Qwen2.5-VL's target modules under transformers 5.16.1.**

**Scheduled: Saturday 24 October 2026 (falls in Phase 4's window — the day is deliberately blocked
out in §3.5). Hard deadline: Sunday 8 November.**
Prerequisite: `scripts/finetune_lora.py` written on the evening of Thu 22 Oct (§3.5 day 3).

### 7.1 What the dry run uses

- **Data: 20 samples.** Twenty. From RSVQA-LR train, already on disk. Throwaway.
- **Steps: 50.** Matching `eval/smoke.py`'s existing 50-step convention — deliberately, so the two
  smoke tests read as a family.
- **Config: the real Phase 8 config**, `configs/finetune_qwen_lora.yaml`, with `max_steps: 50`.
  **Same file, same code path, same dtype, same LoRA target modules as December.** Only the data
  slice and step count change. That is the entire point.
- **Hardware: the rented GPU** (or Kaggle T4 if the rental is not yet live — but then re-run on the
  real GPU before December, because `eval/smoke.py:check_gpu()` exists precisely because a T4 is
  sm_75 and cannot do bf16 or FlashAttention-2).

### 7.2 What the dry run must prove — nine assertions

Anything that can fail in December must fail here instead.

1. `peft` + `transformers 5.16.1` + `Qwen2_5_VLForConditionalGeneration` import and co-exist.
2. **LoRA target modules resolve by name** on this architecture (`q_proj,k_proj,v_proj,o_proj`
   — confirm the actual module names by printing them, do not copy from a blog post).
3. **The vision tower is frozen** and `print_trainable_parameters()` shows a plausible fraction
   (single-digit % or less). A silently-unfrozen vision tower is an OOM in December.
4. **Loss is finite and decreasing over 50 steps** — the exact assertion `eval/smoke.py` already
   makes, now on the real model.
5. **`check_gpu()` gates pass on the actual rented hardware.** bf16 vs fp16, attention
   implementation, VRAM headroom. `eval/smoke.py --gpu` already implements every one of these
   checks; the dry run is where they finally meet real hardware.
6. **Peak VRAM is recorded** at the real image resolution and batch size. This is the number that
   determines whether December's batch size is 1 or 8 — and guessing it wrong costs a day.
7. **Adapter saves, reloads, and merges**, and `QwenVLModel.infer()` returns a sane answer through
   the *registry* (`orchestrator.registry.get(...)`), not through a bespoke script. The fine-tuned
   model must be addressable exactly like the frozen one or the December swap is a rewrite.
8. **`eval/eval.py --model qwen2.5vl-3b-ft --suite ladder --limit 50` completes end to end** and
   writes a results JSON with `run_id`, `git_sha`, `dataset_version`. The full pipeline, in
   miniature.
9. **Wall-clock per step is measured**, so December's schedule is arithmetic rather than optimism.

### 7.3 What the dry run explicitly does NOT prove

It says **nothing** about whether fine-tuning improves accuracy. Twenty samples over 50 steps
measures plumbing, not science. **Do not report any accuracy number from the dry run**, do not
commit its results JSON to `results/` as a measurement, and do not let it into the deck. Name the
output file `smoke_` not `results_` so it cannot be mistaken for one later.

### 7.4 Deliverable

A committed `configs/finetune_qwen_lora.yaml` plus a scratchpad note recording: GPU model and
sm_XX, dtype that actually worked, peak VRAM, seconds/step, trainable-parameter count, and exact
pinned versions of `peft`/`transformers`/`torch`. **That note is what makes 1 December a
configuration change instead of a debugging session.**

---

## 8. Phase 8 — RS fine-tuning / domain adaptation (Tue 1 Dec – Fri 4 Dec, 4 days)

**SIH requires demonstrating real remote-sensing fine-tuning. One controlled reproducible
adaptation experiment is MANDATORY regardless of which way the numbers move.**

### 8.1 Dataset

**Primary: RSVQA-LR train split — already on disk, zero acquisition risk.**

`eval/suites/rsvqa.py:download_rsvqa_lr()` fetches all 12 official files including
`LR_split_train_questions.json` (11.9 MB) and `LR_split_train_answers.json` (7.4 MB), each with a
pinned size and MD5. **The training corpus is downloaded, integrity-checked, and sitting in
`data/raw/rsvqa_lr/` right now.**

Why it is also the *scientifically correct* choice, not merely the convenient one:
- RSVQA-LR is **Sentinel-2 at 10 m** — the exact GSD of the ladder's weakest rung (§5.3).
- Train/val/test are split **by image** (772 images total), so there is no image leakage into the
  test split that produced the Stage 0 baseline.
- **CC-BY-4.0**, verified from the Zenodo record — safe to publish numbers from.
- The `open_accuracy = 0.1651` result is legacy: its checkpoint revision was unrecorded and is unrecoverable. Do not compare a new engineering checkpoint to it; rerun a baseline at the pinned revision on the blocked split in a separate loop.
  `n = 10004`. **The before/after comparison already has its "before."**

**Fallback: VRSBench** (CC-BY-4.0, 37,409 VQA rows, ships both train and val imagery —
`Images_train.zip` 8.36 GB, `Images_val.zip` 3.98 GB per `RECON.md`; answer key is `ground_truth`,
`image_id` is a filename). Only if RSVQA-LR train turns out to be unusable. 12 GB of download on
1 December is a bad plan — this is genuinely a fallback, and if it is even *plausibly* needed the
download must start in November.

**Explicitly NOT:** BigEarthNet.txt (118 GB), LRS-VQA (57 GB of `.7z`). Both documented as blocked
in `RECON.md`. Four days is not a data-acquisition window.

### 8.2 Model / approach

**LoRA on Qwen2.5-VL-3B-Instruct via `peft`. Vision tower frozen. Language-model attention
projections adapted.**

| Knob | Value | Reason |
|---|---|---|
| Rank `r` | 16 | Standard; enough capacity for answer-format + domain adaptation |
| `alpha` | 32 | 2×r, the conventional pairing |
| `dropout` | 0.05 | |
| Targets | `q_proj, k_proj, v_proj, o_proj` (LM only) | Verified by name in the Oct dry run |
| Vision tower | **frozen** | Cheapest, most stable, least likely to OOM |
| Precision | from the dry run's measured result | bf16 if sm_80+, fp16 on a T4 — `check_gpu()` enforces |
| Epochs | 1–2 over a stratified subset | Not the full train split; time-boxed |
| Seed | fixed, recorded | Reproducibility is a deliverable |

**Why not full fine-tuning, QLoRA, or a larger backbone:** four days, one GPU, two people. LoRA on
a 3B model is the rung of the ladder that holds. Adapters are small, fast, revertible, and — the
decisive property under the governing rule — **the base checkpoint is never modified**, so the
rollback is deleting a file.

### 8.3 Evaluation harness

**Before/after on the exact ladder slice Phase 7 named, frozen on Sat 28 Nov before any training.**

| | |
|---|---|
| Primary | The Phase 7 slice metric (predicted: `binary_accuracy` + `pred_yes_rate_on_binary` at the 5 m and 10 m rungs, where the baseline shows total collapse at yes-rate 0.975 / 1.000) |
| Secondary | RSVQA-LR test `--full` (n=10,004): `open_accuracy` against a separately rerun baseline at the pinned revision |
| **Regression guard** | **Every other ladder rung.** A fine-tune that fixes 10 m and breaks 0.3 m has not improved the system. This is the floor rule expressed as a metric. |
| Degeneracy guard | The existing two-sided guard, per rung. A fine-tune that teaches the model to always say "no" would *raise* accuracy on a no-heavy slice — `degenerate()` catches exactly this, which is why it must run on the after-run too. |
| Sample size | The Phase 7 regenerated ladder at 1,000 sources = 2,000/rung → MDE ≈ 6 points. At the original 200 sources the MDE is ≈13 points, which cannot resolve a realistic fine-tuning effect. **This is why §5.1 lifts the cap.** |

**Versioning — all three, or the experiment does not count:**
- Config: `configs/finetune_qwen_lora.yaml`, committed, hash recorded in the results JSON.
- Dataset: `dataset_version` = hash of the exact sample-id list used for training, committed.
- Checkpoint: adapter weights + `run_id` + `git_sha` + the pinned `peft`/`transformers`/`torch`
  versions from the October dry run.

### 8.4 The swap rule (the floor, stated as a decision procedure)

```
if (slice_metric_after - slice_metric_before) CI excludes zero
   AND no other rung regresses by more than 3 points
   AND the degeneracy guard passes on the after-run:
       register qwen2.5vl-3b-ft as the live provider
else:
       the frozen checkpoint stays live; the adapter ships as an artifact,
       not as the default; the experiment is reported exactly as measured
```

**Both outcomes are deliverables.** *"We fine-tuned on 10 m Sentinel-2 RSVQA-LR, measured a −2.1
point change on the target slice with CI [−5.0, +0.8], and therefore kept the frozen checkpoint
live"* is a **complete, honest, SIH-satisfying answer** to the fine-tuning requirement. It
demonstrates the capability, the discipline, and the evaluation rigour simultaneously. A negative
result reported cleanly beats a positive result nobody can reproduce — and it is the only outcome
consistent with this project's own stated rules.

The registry makes this trivial: register `qwen2.5vl-3b-ft` as a *second* model name from day one,
and the "swap" is one line in `register_default_providers()`. Both checkpoints stay addressable for
side-by-side demo, which is itself a good demo.

### 8.5 Day by day

| Day | Date | Work |
|---|---|---|
| 1 | Tue 1 Dec | GPU up. **Re-run the October dry-run config unchanged** to confirm the environment (30 min). Then build the training slice from RSVQA-LR train, stratified over question type. |
| 2 | Wed 2 Dec | Train. Checkpoint every N steps. **Eval at 25% / 50% / 100%** so a diverging run is caught mid-flight rather than at the end. |
| 3 | Thu 3 Dec | Before/after on the frozen Phase 7 slice + **all other rungs** (regression guard) + RSVQA-LR test `--full`. |
| 4 | Fri 4 Dec | **Swap decision per §8.4.** Commit adapter, config, dataset manifest, both results JSONs, and the write-up. Freeze. |

**No training after Wed 2 Dec.** Thursday and Friday are evaluation and documentation. A training
run started on Thursday cannot be evaluated, cannot be swapped in safely, and cannot be written up
— so it cannot happen.

### 8.6 Kill criterion

**There is no kill criterion for Phase 8. The experiment is mandatory.**

What *can* be reduced, in priority order:
1. Training set size (fewer samples, still stratified).
2. Epochs (1 instead of 2).
3. Eval breadth — but **never** the regression guard, and never `--full` on RSVQA-LR test, because
   those are what make the result comparable to the committed baseline.

**Hard abort rule:** if training is not running by **EOD Wed 2 Dec**, fall back to the October dry
run's configuration on a smaller slice and report **that** as the controlled experiment. A 200-sample,
200-step adaptation reported with full provenance satisfies the requirement. A 4-day debugging
session that produces nothing does not — which is precisely what §7 exists to prevent.

---

## 9. Real confidence / uncertainty — recommendation

### 9.1 The problem, stated exactly

`models/qwen_vl/model.py` returns `"confidence": 1.0` on every single inference, with a comment
admitting it is a placeholder. The frontend renders the literal string
`"Confidence calibration pending"` in two places, asserted by two golden-path tests.

The master plan lists this as **MUST SHIP** and assigns it to **no phase.**

**Under a zero-fabrication requirement, `confidence: 1.0` is not a missing feature — it is a false
statement in the response payload.** A judge who opens the API response sees a model claiming total
certainty on every answer, including the ones it got wrong. That is worse than no confidence field.

### 9.2 Recommendation: split it across two phases

**Keep it MUST SHIP — but split it, because the two halves have very different costs.**

**Part A — honest raw confidence. MUST SHIP. Lands in Phase 3.**

Token-logprob confidence. `generate(output_scores=True, return_dict_in_generate=True)` is already
available on the call that `_generate_answer` makes.

- **Binary questions:** softmax probability of the first generated token being "Yes" vs "No".
  Clean, well-defined, directly interpretable.
- **Open-ended:** `exp(mean(token_logprobs))` over the generated answer.
- Cost: **~15 lines, zero extra inference.** The scores are already computed; they are currently
  thrown away.
- Report it as `{"confidence": 0.83, "confidence_method": "token_logprob", "calibrated": false}`.
- Frontend string changes from `"Confidence calibration pending"` to
  **`"Uncalibrated model confidence (token log-probability)"`**.

**Why Phase 3:** Phase 3 already opens `QwenVLModel.infer()` to add typed evidence. Doing both in
one visit is one code review instead of two. And from Phase 3 onward, *every* results JSON carries
a real confidence — which is exactly the corpus Part B needs.

**Part B — calibration. SHOULD SHIP (demote from MUST). Lands in Phase 7.**

Raw logprobs are systematically overconfident. Calibrating them:

1. **Reliability diagram + ECE** (10 bins, accuracy vs mean confidence) computed over the ladder and
   RSVQA runs. Pure post-processing on results JSONs that already exist by then — **no GPU, no new
   inference.**
2. **Temperature scaling** if ECE is bad: fit one scalar `T` by minimising NLL on **RSVQA-LR val —
   which is already downloaded** (`LR_split_val_questions.json`, 3.48 MB). Seconds on CPU. Apply as
   `softmax(logits/T)` at inference.
3. Report ECE on the **test** split, never on the split `T` was fitted on.

Then the frontend says **`"Calibrated confidence (temperature-scaled, ECE=0.04 on RSVQA-LR test)"`**
— a claim with a number and a split behind it, which is the standard this repo holds everything
else to.

**Why Phase 7:** Phase 7 re-runs everything anyway and produces exactly the corpus of
(prediction, confidence, correct) triples that calibration needs. Doing it earlier means doing it
twice.

**Why SHOULD not MUST:** if Phase 7 runs short, *"uncalibrated token-logprob confidence"* is still
honest and still infinitely better than `1.0`. Calibration is a quality upgrade on an already-true
statement. Part A is the one that removes a falsehood, and that is the one that cannot slip.

### 9.3 Rejected: self-consistency / ensemble agreement

Sample k=5 with temperature > 0, use answer agreement as confidence. Rejected on three grounds:

1. **5× inference cost.** Phase 7's big run is already the GPU-hour peak; multiplying it by five is
   not available.
2. **It requires sampling**, which breaks the greedy `do_sample=False` decoding that every committed
   baseline in `results/` was measured under. Every number would need re-running to stay comparable.
3. **It is not better.** Agreement among samples measures answer stability, not correctness. A
   confidently-wrong model agrees with itself five times out of five.

Note it in the write-up as a considered-and-rejected alternative — that is worth saying out loud in
a technical Q&A.

### 9.4 Validation checklist for the confidence claim

- [ ] Reliability diagram plotted, committed to `results/` (reuse `plot_ladder.py`'s matplotlib setup)
- [ ] ECE reported with bin count and n
- [ ] Confidence stratified by `binary` vs `open` — they will calibrate differently, and a single
      pooled ECE would hide that
- [ ] **Confidence on `no_evidence` grounding responses is `null`, not `0.0`.** "No confidence value
      exists" and "confidence is zero" are different statements, and this repo already makes exactly
      this distinction: `plot_ladder.py` omits empty rungs rather than plotting them at 0.0, because
      *"'no data' and 'scored zero' must not render identically."* Apply the same rule.
- [ ] The frontend string states the method and whether it is calibrated. Never a bare number.

---

## 10. The eval-harness gap — the single biggest science risk (and its fix)

Restating §0.1 because everything above depends on it.

**`eval/eval.py` can score exactly one thing: string equality between `prediction["answer"]` and
`sample["expected_answer"]`.** Phases 3, 4, 5 and 8 all produce outputs that are not that. There is
no metric dispatch, no per-condition stratification, no IoU, no calibration.

The risk is insidious precisely *because* the harness looks mature — it has degeneracy guards, a
sample-size floor, a rescore note, per-rung stratification, archived artifacts. It is easy to assume
it will handle a bounding box. It will not: it will call `answer_matches("[120, 340, 200, 410]",
"[118, 336, 205, 415]")` and return `False`, and the run will complete without error, and the
resulting JSON will read as a measurement.

**Fix (W0.2, Week 0, ~1 day) — a metric dispatch keyed on suite:**

```python
# eval/metrics.py  (new file, ~120 lines)
SCORERS = {
    "answer_match": score_answer_match,   # the existing answer_matches path, unchanged
    "grounding_iou": score_grounding_iou, # Phase 3: acc@0.5, mean_iou, no_evidence_rate,
                                          #          mean_box_area_fraction
    "condition_delta": score_conditions,  # Phases 4 & 5: per-condition summarise() + delta + CI
}
SUITE_SCORER = {
    "rsvqa": "answer_match", "ladder": "answer_match",
    "grounding": "grounding_iou",
    "change_vqa": "condition_delta", "optical_sar": "condition_delta",
    "resolution_proxy": "answer_match",
}
```

Design constraints, all of them deliberate:

- **`summarise()`, `degenerate()`, `answer_matches()` are not modified.** They are the measurement
  standard behind every committed baseline in `results/`. `condition_delta` *calls* `summarise()`
  per condition — the same relationship `per_rung` already has. Existing numbers stay comparable;
  nothing needs rescoring.
- **`condition_delta` is one scorer, not two.** Phase 4's (pair / single / identical) and Phase 5's
  (optical / SAR / fused) are structurally identical: same questions, different input conditions,
  headline = a delta with a CI. One implementation, used twice. Writing two would be the
  copy-paste drift this repo's own standards warn against.
- **Wilson 95% CI on every proportion**, everywhere. Once, in `metrics.py`, not per-phase.
- **`run_id` + `dataset_version` + `dtype` + `device`** added to the report in the same pass (§5.1),
  so every phase from 3 onward emits stamped artifacts and Phase 7 has nothing to retrofit.

**Do this in Week 0.** It is one day now, or it is one day stolen from the middle of Phase 3 when
the model is working and the harness cannot measure it.

**Runner-up risk:** §0.5 — the hardcoded CUDA check means Track A has no local iteration loop at
all. Combined with the harness gap, the failure mode is renting a GPU and spending it debugging the
evaluator. W0.1 (30 minutes) and W0.2 (1 day) together remove both.

---

## 11. Summary calendar

| Date | Item | Blocking? |
|---|---|---|
| **Sat 19 Sept** | **Phase 5 data spike** (§6) | Determines whether Phase 5 exists |
| 14 Sept – 5 Oct | W0.1 MPS · W0.2 metric dispatch + run_id · W0.4 rent GPU · W0.5 peft | **Blocks everything** |
| Tue 6 – Mon 19 Oct | **Phase 3 Grounding.** Kill check Wed 14 Oct. Confidence Part A ships here. | |
| Thu 22 Oct | Write `scripts/finetune_lora.py` | Prereq for the dry run |
| **Sat 24 Oct** | **Phase 8 fine-tuning DRY RUN** (§7) | De-risks December |
| Tue 20 Oct – Mon 2 Nov | **Phase 4 Change-VQA.** Kill check Wed 28 Oct. | |
| Tue 3 – Mon 16 Nov | **Phase 5 Optical-SAR.** Go/no-go #2 Fri 6 Nov. Kill Wed 11 Nov. **Cuttable.** | |
| 17–23 Nov | Phase 6 (not Track A science) | |
| Tue 24 – Mon 30 Nov | **Phase 7 Ladder + Indian proxy.** Weak slice frozen Sat 28 Nov. Confidence Part B ships here. | **Cannot be cut** |
| Tue 1 – Fri 4 Dec | **Phase 8 Fine-tuning.** Swap decision Fri 4 Dec. No training after Wed 2 Dec. | **Mandatory** |
| 8–15 Dec | SIH finale | |

---

## 12. Sources for dataset facts stated above

Facts I verified during this analysis, with sources. Everything else is marked **UNVERIFIED** or
**uncertain** in the text above and must be checked before it is relied on.

- DIOR-RSVG: 17,402 images / 38,320 expressions, 20 categories, splits 26,991/3,829/7,500, Google
  Drive distribution, XML annotations — [ZhanYang-nwpu/RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch), [arXiv 2210.12634](https://arxiv.org/pdf/2210.12634)
- LEVIR-CC: 10,077 pairs / 50,385 sentences, 2.5 GB — [Chen-Yang-Liu/LEVIR-CC-Dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
- LEVIR-CD: 637 pairs, 1024×1024, 0.5 m/px, Google Earth, **academic use only, commercial
  prohibited** — [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/)
- CDVQA: 2,968 pairs at 512×512 from the SECOND public subset, >122,000 QA pairs, **on GitHub**
  — [YZHJessica/CDVQA](https://github.com/yzhjessica/cdvqa), [arXiv 2112.06343](https://arxiv.org/pdf/2112.06343)
- BigEarthNet-MM: 590,326 S1+S2 pairs, 10 European countries, S1 IW GRD dual-pol VV+VH,
  **CDLA-Permissive-1.0** — [arXiv 2105.07921](https://arxiv.org/pdf/2105.07921), [bigearth.net](https://bigearth.net/), [Zenodo 10891137](https://zenodo.org/records/10891137)
- SEN12MS: 180,662 S1/S2/MODIS-LC triplets, 256×256, seasonal tarballs — [arXiv 1906.07789](https://arxiv.org/pdf/1906.07789)

**Not verified, do not quote without checking:** DIOR image-archive size; SEN12MS per-tarball sizes;
xView Indian-tile coverage; Bhuvan/NRSC licence and API; every ISRO sensor GSD figure in the §5.2
table (check against NRSC/ISRO published specifications before it goes on a slide).

Repo-internal facts (LoveDA 12 GB / 8,382 PNGs, RSVQA-LR 291 MB with train+val present, torch 2.14
with MPS available, `evidence: []` everywhere, `confidence: 1.0`, ladder `--limit` capped at 200,
no run_id/dataset_version in results, SAR pipeline previously executed with real HyP3 job IDs) were
established by direct inspection of the working tree at `93b21b0` on 13 Sept 2026.
