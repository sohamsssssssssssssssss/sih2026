# Calibration and abstention evaluation design (Day 19)

Question: **when SatQuery is uncertain or the request is unsupported, does refusing improve reliability?**

This document separates what is measured now from what needs GPU runs. No confidence value is invented anywhere. When a component has no legitimate per-example score, it is evaluated only through its refusal gates.

## Metrics module

`eval/abstention.py` (tests: `eval/test_abstention.py`, verified on synthetic data only). The input is per-example `(score, correct)` records. `score=None` means the system produced no answer (it abstained). Abstentions count in the denominator and are never covered.

| Metric | Definition |
|---|---|
| coverage(t) | #{score ≥ t} / N, where N includes abstentions |
| risk(t) | 1 − accuracy among examples with score ≥ t |
| risk–coverage curve | One point per distinct score. Tied scores enter together, because no threshold can split them |
| AURC | Right-step integral of risk over coverage, from 0 to the maximum reachable coverage. Report `max_coverage` alongside it. Lower is better |
| selective accuracy @ c | The largest-coverage operating point with coverage ≤ c. `None` if no such point exists |
| ECE | Σ_b (n_b / n_scored) · \|accuracy_b − mean_score_b\| over 15 equal-width bins on [0, 1]. **Only meaningful for a score that claims to be a probability of correctness** |

## Status per component

### Routing and validation layer — measured (SQ-20260926-501)

The layer is deterministic and has no score, so its abstention is its refusal gates. Results on the labelled set, all providers stubbed ready:

| | Dispatches | Wrong dispatches | Precision | False refusals |
|---|---|---|---|---|
| Gate-less planner | 160 | 86 | 0.463 | 0 / 104 |
| With gates | 141 | 67 | 0.525 | 0 / 104 |

**Answer for this layer:** refusing does improve reliability. It removed 19 of 86 wrong dispatches at no false-refusal cost. However, it does not address the main failure: out-of-scope and misrouted free text defaults to VQA (0 / 22 free-text unsupported requests refused). See `orchestrator-routing-benchmark.md`.

Refusals on incompatible inputs are complete on the fixture suite:

- invalid pairs: 23 / 23 rejected with the expected reason code
- invalid requests: 20 / 20 rejected
- unready providers: 119 / 119 structured 503, 0 answers, 0 cached fallbacks

### Grounding (Grounding DINO Swin-T) — score exists, awaiting real predictions

- **Score**: the provider returns a finite detection score in [0, 1] for each box (`evidence[].confidence`). The DIOR-RSVG suite stores the top box as `selected_prediction`. `eval.abstention.grounding_records(results)` maps each example to `(top-box score, IoU ≥ 0.5)`; an example with no returned box becomes `(None, False)`.
- **Caveat**: this is a raw detection score (sigmoid of the box–text logit). It is not a calibrated probability that the box is correct. ECE will measure exactly that gap. In any report, call it "raw detection score", never "confidence".
- **Truncation**: with the frozen `box_threshold = 0.35`, boxes below 0.35 are never returned. The curve therefore has no points below 0.35, and those examples appear as abstentions. For a full curve, the grounding track should save per-example predictions at a low `box_threshold` (for example 0.05) and sweep thresholds offline. The 0.35 operating point is one point on that curve. The locked 400-expression evaluation set must stay unchanged.
- **Planned run**: once the grounding track commits per-example predictions, compute `summarize(grounding_records(results))` and report:
  - AURC and `max_coverage`
  - selective Pr@0.5 at coverages 0.25 / 0.5 / 0.75
  - ECE with its bins
  - whether refusing below the 0.35 threshold raises Pr@0.5 among answered examples

  Record the run under a new SQ id.

### Single-image VQA (Qwen2.5-VL-3B) — no score today; GPU design only

`QwenVLModel.infer` decodes greedily (`generate(do_sample=False)`), discards the token scores, and returns `confidence: null`. This is correct: there is no calibrated value. Legitimate scores that could be exposed, **none measured yet**:

1. **Answer-sequence log-probability.** Call `generate(..., output_scores=True, return_dict_in_generate=True)`, then `model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)`. Sum (and separately average) the log-probs of the answer tokens, excluding EOS. This works for every question type. RSVQA-LR answers are one word or number, so the sequences are short.
2. **Closed-set restricted softmax.** For yes/no question types (presence, comparison), run one forward pass on the prompt and take softmax over the first-token logits of the `yes` / `no` token ids only. The resulting p(predicted answer) is a score. It must use the same prompt as the accuracy run (`ANSWER_INSTRUCTION` in `training/remote_sensing.py`, which matches the provider suffix).
3. **Calibration protocol.**
   - Fit a single temperature on the **validation** split only.
   - Choose any abstention threshold on validation at a pre-declared target coverage.
   - Report ECE, AURC and selective accuracy on the **test** split once, for base and adapter separately.
   - Never tune on test.
4. **Reporting rule.** Until step 3 has run on held-out data, an exposed value is a "raw answer log-probability", not confidence. The API must keep `confidence: null`.
5. **GPU-bound.** All of this needs the Kaggle T4 runs (Day 3 base V0, Day 8 adapter). It should be added to the VQA evaluation runner as extra per-example fields (`answer_logprob_sum`, `answer_logprob_mean`, `yes_no_restricted_p`), without changing the answer or the accuracy computation.

### Change and optical-SAR (deterministic) — no score by design

Both providers are deterministic measurements. They have no model probability, and none should be fabricated. Their abstention is the pair-compatibility gate, which is measured above. A per-pixel threshold margin (for example |ΔNDVI| − 0.1) is a physical quantity, not a probability of a correct answer. It could order examples for a risk–coverage curve only after a labelled change reference exists (the change track). Until then, it is not evaluated.

## What remains GPU-bound

- Qwen answer log-prob and restricted-softmax scores, temperature scaling, and base-vs-adapter selective accuracy
- Grounding low-threshold per-example predictions, if not produced on CPU by the grounding track

## Reproduction

```bash
python3 -m pytest -q eval/test_abstention.py
python3 -c "import json; from eval.abstention import summarize, grounding_records; \
r=json.load(open('<grounding-report>.json')); print(json.dumps(summarize(grounding_records(r['results'])), indent=2))"
```
