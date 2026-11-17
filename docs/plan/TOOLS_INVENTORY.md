# Tools inventory (T07)

Scope: every existing inference entry point on `task/T06-schema` @ `23dbbbf`, read from the
tree and from the commits that carry the artefacts. "CPU" means the wrapper can run on this
laptop without a CUDA GPU; it does not mean the output is benchmark-comparable.

Registry candidates are the four real providers already bound in
`orchestrator/capabilities.py`: `qwen2.5vl-3b`, `grounding-dino-swint`,
`optical-sar-deterministic`, `change-deterministic`.

| # | Entry point | Path | Inputs | Outputs | Weights / artefacts | CPU |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5-VL-3B zero-shot VQA (`qwen2.5vl-3b`) | `models/qwen_vl/model.py` (`QwenVLModel.infer`) | 1+ local image paths, question text | `{answer, confidence: None, evidence: []}` | `Qwen/Qwen2.5-VL-3B-Instruct` resolved by `models/artifacts.py`; `SATQUERY_QWEN_MODEL_DIR`, default Kaggle path; revision `66285546…` | No — CUDA required; MPS only with `SATQUERY_ALLOW_MPS=1` for local iteration, CPU fails closed |
| 2 | B13 verified VQA adapter helper (`adapter_path`, `adapter_identity`, `count_target`) | `models/qwen_vl/b13.py` in `d867c4a`, `f28b862`, `bcd0e23` (not tracked on this branch, not imported by it) | `SATQUERY_VQA_ADAPTER` directory, image path, question | adapter path / identity hashes / count target phrase | B13 LoRA adapter under `SATQUERY_VQA_ADAPTER` (SHA-256 pinned in that module) | No — same Qwen GPU requirement |
| 3 | Qwen RS LoRA training/eval stack | `training/remote_sensing.py`, `scripts/train_remote_sensing_adapter.py`, `models/qwen_vl/stage1.py` | JSONL manifest, base model dir, image root | training/eval reports, not a runtime answer | base Qwen + LoRA output dir (none committed) | No — `peft`/`bitsandbytes`/CUDA |
| 4 | Grounding DINO Swin-T (`grounding-dino-swint`) | `models/grounding_dino/model.py` (`GroundingDINOModel.infer`) | exactly one image path, text query | `{answer, evidence: [{type: bounding_box, label, coordinates, confidence}]}` | `groundingdino_swint_ogc.pth` (`SATQUERY_GROUNDING_CHECKPOINT`), config (`SATQUERY_GROUNDING_CONFIG`), cached `bert-base-uncased` text encoder | No — CUDA required; the wrapper deliberately disables CPU |
| 5 | Deterministic optical-SAR evidence (`optical-sar-deterministic`) | `models/optical_sar/model.py` (`OpticalSARModel.infer`) | co-gridded 5-band `B02/B03/B04/B08/dataMask` optical + 3-band `VV/VH/dataMask` SAR GeoTIFFs | `{answer, confidence: None, evidence: [optical_statistics, sar_statistics, joint_valid_coverage]}` | none (pure raster math) | Yes — numpy + rasterio |
| 6 | Deterministic bi-temporal change (`change-deterministic`) | `models/change/model.py` (`ChangeModel.infer`) | two co-registered RGB or 5-band multispectral rasters, same CRS/grid | `{answer, confidence: None, evidence: [temporal_inputs, change_statistics, temporal_valid_coverage, change_extent]}` | none | Yes — numpy + rasterio |
| 7 | Sentinel-1 open-water/flood baseline (`change_vqa` SAR branch) | `models/change/sar_water.py` (`water_change`), village attribution in `models/change/villages.py` | co-gridded `VV/VH/dataMask` T1/T2 rasters; optional village boundaries | water-change statistics, hectares, GeoJSON polygons, abstention when no water mode | none (split-based Otsu + fixed tiles) | Yes — numpy + rasterio |
| 8 | CDVQA G3 evaluation of the change adapter | `results/cdvqa-g3/*` in `89fc5bf` (not present on this branch) | CDVQA test1/test2 pairs | eval logs/results only; **G3 controls failed** | none committed here | n/a (evaluation artefact, not a runtime tool) |
| 9 | MockModel | `models/mock/model.py` | image paths, question | canned "mock answer" | none | Yes — test fixture only, must not be registered as a real tool |
| 10 | Script/CLI wrappers | `scripts/s1_offline_change.py`, `scripts/gpu_smoke.py`, `eval/smoke.py` | CLI args / rasters | logs, JSON reports | same as the models they call | Mostly no (GPU smoke) |

## Not present as an inference tool

- **BEN land cover (s1 / s2 / s1s2)**: the repository has BigEarthNet data acquisition,
  manifests and dataset loaders (`data/bigearthnet*.py`) and a Qwen RS fine-tuning stack,
  but **no land-cover classifier entry point and no weights**. There is nothing to wrap.
- **Separate "flood baseline" model**: flooding is the SAR-water branch of
  `change-deterministic` (row 7); there is no separate checkpoint.
- **CDVQA-trained change adapter**: `results/cdvqa-g3` records an evaluation (commit
  `89fc5bf`, G3 controls failed), but the adapter code/weights are not on this branch. The
  only runnable change path is the deterministic baseline, so it stays
  `experimental_unvalidated`.

## Gaps carried into T07

- The B13 helper (`models/qwen_vl/b13.py`) lives on other commits but is not tracked on
  `task/T06-schema`; `backend/services.py` does not import it here. The registry adapter
  for Qwen therefore wraps only `models/qwen_vl/model.py` (row 1) and does not depend on
  `b13`.
- No tool has a committed results file that passes its gate. Every registered tool is
  therefore `experimental_unvalidated`; all `validated` statuses require a commit + results
  evidence pair in `core/tool_status.json` and are never defaulted.
- `demo_gui/app.py` and `frontend/components/status/CapabilityStatus.tsx` still hardcode
  capability labels; the T07 mapping function replaces the backend table only.
