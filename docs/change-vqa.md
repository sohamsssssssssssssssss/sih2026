# Change VQA (CDVQA adapter), experimental, off by default

Answers free-form change questions on an optical before/after image pair with Qwen2.5-VL-3B
(pinned rev `66285546d2b821cf421d4f5eb2576359d3770cd3`) plus the CDVQA LoRA adapter
(G2, `cdvqa-train-6/checkpoints/checkpoint-1000`, 1,000 steps on Kaggle T4).
**Nothing about its accuracy is measured in this app.** The only measured CDVQA numbers are the
G1 base zero-shot (test1-400 macro 0.395) and the majority baseline (0.44). The adapter has to clearly
beat 0.44 in G3 before anyone claims anything.

## Flag and config
`configs/change_vqa.json`:

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | Route on/off. Env `SATQUERY_CHANGE_VQA=1` turns it on, `=0` turns it off. |
| `adapter_path` | `null` | Adapter directory. Env `SATQUERY_CDVQA_ADAPTER` overrides it. |
| `adapter_id` | `cdvqa-g2-checkpoint-1000` | Label reported in the payload. |
| `cdvqa_validated` | `false` | Set to `true` only once the G3 result is committed and supports a claim. |
| `package` | `packages/cdvqa-g0f-20261113T115304Z` | Frozen package that supplies the prompt builder and vocabulary. |
| `max_ratio` | `2.0` | Refuse pairs whose image size or GSD differs by more than this factor. |

With the flag off, the planner never selects the CDVQA rules, so routing is unchanged. Change questions still
go to `change-deterministic`, including the Sentinel-1 flood/CDSE path. Tested in `backend/test_change_vqa.py`.

Enable locally (Mac, MPS iteration only; numbers from it are not results):
```
SATQUERY_CHANGE_VQA=1 SATQUERY_CDVQA_ADAPTER=~/models/cdvqa-checkpoint-1000 SATQUERY_ALLOW_MPS=1 backend/.venv/bin/uvicorn backend.main:app --port 8000
```
Then upload two optical images (before first, after second) and ask with `scene_id` = before and `scene_id_2` = after.

## Flow
1. **Planner** (`orchestrator/planner.py`): fires when the flag is on, there are 2 scenes, the sensor is not SAR and no capability is forced.
   - The question matches a CDVQA template → `change_vqa` with rule `cdvqa_change_vqa`.
   - Otherwise, if it has change intent → rule `cdvqa_unsupported`.
   - The template match is needed because 43% of the frozen eval questions (e.g. "What is the largest change?") do not parse as CHANGE intent. `orchestrator/question.py` is unchanged.
2. **Backend** (`backend/services.py::_cdvqa_inputs`): both scenes must be uploads with a stored image, and neither may be declared `modality=sar`. Undeclared modality (`unknown`) is accepted. The request order is before, after. GSDs come from `scene_gsd`.
3. **Router** (`orchestrator/router.py::_change_vqa`): sends refusals without any model call.
   - `unsupported_change_question`
   - `pair_refused`: not exactly 2 images, width/height differ >2×, or both GSDs known and differing >2×. An unknown GSD is not refused.
   - Otherwise it runs model `qwen2.5vl-3b-cdvqa` through the existing worker.
4. **Model** (`models/qwen_vl/cdvqa.py`): classifies the question into one of the 8 CDVQA types, builds the prompt with the package's own `training/remote_sensing.messages()` (byte-identical to the eval prompt; test compares all 800 eval prompts), runs it, then checks the answer against that type's Train vocabulary (strip + casefold, the same rule as `eval/cdvqa_pairs.normalise`).
   - Prompt order: [before image, after image, question], 392×392, pair-order sentence, allowed answers, answer instruction.
   - Decoding: greedy, `max_new_tokens=32`.

## Payload
`answer` holds the raw model answer, or `malformed: <raw>` when it is outside the vocabulary. `answer_source` is `cdvqa_adapter`. The `change_vqa` object holds:
`answer_status` (`ok`/`malformed`), `question_type`, `allowed_answers`, `adapter` (`id`, sha256 of
`adapter_model.safetensors` and `adapter_config.json`), `validation_status` (`experimental_unvalidated`
until `cdvqa_validated=true`), and `caveat`. Refusals carry `change_vqa.refusal` and `reason`. The notice repeats
the unvalidated status. Everything in `change_vqa` is also written to the trace params.

## Memory
The CDVQA adapter is loaded onto the **same** base instance as B13 (`peft load_adapter` / `set_adapter`), so there is
no second ~7.5 GB fp16 base. Only one adapter is active per call, and B13 is re-activated after each change call.
Without B13, base-only calls run with the adapter disabled. The extra cost is the LoRA weights (r16, q/k/v/o, tens of MB) plus
two images' vision tokens per call. **Peak memory on a 16 GB Mac with two images is unmeasured.**

## Caveats and what is unmeasured
- **All accuracy is unmeasured** until the G3 result (adapter_real / identical / swapped vs base and majority) is committed.
- The adapter was trained on SECOND aerial RGB (0.5–3 m). Hidden ISRO scoring uses Cartosat-2S / RISAT pairs, and it has not been validated on those.
- Train/eval used a 4-bit QLoRA base, while the app runs an fp16 base. Unlike B13, there is no app-vs-eval parity run for CDVQA yet.
- Question typing covers the CDVQA English templates only. Paraphrases get `unsupported_change_question`.
- The 2× size/GSD rule is a policy choice, not a measured threshold. The pair is not co-registered or resized beyond the 392 resize.
- `infer_pair` (adapter swap on real weights) has no CPU test; the tests stub the runner.
