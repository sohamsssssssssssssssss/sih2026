# VLM GPU environment (brief Day 1)

> **Status: NOT YET MEASURED.** No GPU was available when this was written (Apple M3 Pro, no CUDA). Every "value" cell below stays empty until `kaggle/run_vlm_program.py env` runs on the target GPU. Its registry record, not this page, is the evidence. Fill the table from that record; never from memory or from another machine.

## Pinned artifacts (verified metadata; no weights downloaded here)

| artifact | identity | verification |
|---|---|---|
| base model | `Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546d2b821cf421d4f5eb2576359d3770cd3` (last modified 2025-04-06) | `configs/research/qwen2.5-vl-3b-instruct@66285546.files.json`: size plus SHA-256 for both safetensors shards (`41a8895c…`, `365531ff…`), size plus blob SHA-1 for the 10 small files. The blob method was checked locally against the pinned `config.json`. |
| dataset | RSVQA-LR, Zenodo 6344334 | 12 files MD5-checked by `eval.suites.rsvqa`; manifest SHA-256 `23a74c57…` (SQ-20260926-101) |
| dev subset | `eval/subsets/rsvqa-lr-validation-dev1000.v1.json` | SHA-256 `e3b43319…4e20c`, enforced by `--expected-subset-sha256` |
| locked test subset | rebuilt at Day 8 | SHA-256 `b21e5046…057b5`, enforced |

The production provider config (`configs/model_artifacts.json`) keeps `revision: null`. It changes only after the env stage has verified real bytes against this pin.

## Fields the env stage captures

| field | source | value |
|---|---|---|
| Python | `platform.python_version()` | NOT YET MEASURED |
| platform / kernel | `platform.platform()` | NOT YET MEASURED |
| torch | `torch.__version__` | NOT YET MEASURED |
| CUDA runtime | `torch.version.cuda` | NOT YET MEASURED |
| GPU model | `torch.cuda.get_device_name(0)` | NOT YET MEASURED |
| driver, GPU memory | `nvidia-smi --query-gpu=name,driver_version,memory.total` | NOT YET MEASURED |
| transformers, peft, accelerate, bitsandbytes, qwen-vl-utils, torchvision, huggingface-hub, safetensors, numpy, pillow | `importlib.metadata` | NOT YET MEASURED |
| model files match the pin | `verify_model_files()` | NOT YET MEASURED |
| manifest SHA matches `23a74c57…` | `sha256_file()` | NOT YET MEASURED |
| offline load time, one-sample answer, peak VRAM | `load-check` stage | NOT YET MEASURED |

Any failed check makes the env record FAILED. Every later stage then refuses to run in that session.

## Historical context (different commit; not evidence for this program)

`docs/gpu-smoke.md` records a Kaggle Tesla T4 at `ad9cbdb` with PyTorch 2.10.0+cu128, Transformers 4.49.0, Accelerate 1.13.0 and qwen-vl-utils 0.0.14. The model revision was not verified there. PEFT and bitsandbytes versions were not recorded. It shows the stack once ran on a T4; it does not fix this program's environment.

## Reproduction

The exact commands are in [`vlm-kaggle-runbook.md`](vlm-kaggle-runbook.md) (sessions S1–S5).
