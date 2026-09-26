# Research migration manifest

## Snapshots

- Source: `https://github.com/AtharvaPatil466/SatAi`, branch `research/integration-v1`, SHA `a45602f202cbbd7080f3f17ea466c1dd5795e6aa`.
- Destination base: `https://github.com/sohamsssssssssssssssss/sih2026`, branch `main`, SHA `391f333c27f5b7112cc31d9bad2d2632097fc0f4`.
- The source snapshot descends directly from the destination base. The migration still copies selected content rather than merging SatAi history.

## A. Migrate directly

- Experiment registry infrastructure and immutable historical records under `eval/registry/`.
- Frozen routing query set, evaluator, abstention metrics, tests, and historical routing output under `eval/routing/`.
- Grounding research harness, threshold/prompt study, comparison support, tests, reports, and audit outputs.
- Real-pair change evaluator, bounded Sentinel-2 acquisition tooling, pair manifests/provenance, tests, report, and historical output.
- Optical/SAR sensor-necessity evaluator, bounded acquisition/reference tooling, tests, manifests, report, and historical outputs.
- RSVQA audit, locked subsets, geographic leakage checks, BigEarthNet metadata manifests, tests, and historical audit/rescore outputs.
- VLM/Qwen staged Kaggle runner, local model-file checksum manifest, tests, and GPU/runbook documentation.

## B. Reconcile with canonical files

- `eval/suites/grounding_dior_rsvg.py` and its test: retain the canonical loader while adding explicit split-file support used by research evaluation.
- `scripts/evaluate_remote_sensing_adapter.py`: retain the canonical offline adapter path while adding locked-subset metrics and reproducible base/adapter comparison.
- `scripts/train_remote_sensing_adapter.py` and `training/remote_sensing.py`: retain canonical training contracts while adding model revision, bounded checkpoint resume, and measured runtime reporting.
- `docs/change-baseline.md` and `docs/remote-sensing-adaptation.md`: update canonical documentation only for the migrated evaluations and final-commit pinning workflow.
- Historical registry records gain an explicit SatAi repository URL; future records capture the canonical remote URL. Existing SHAs, timestamps, parameters, metrics, and dirty flags remain unchanged.

## C. Provenance and results

Imported records and result JSON are historical SatAi results. They retain their original SatAi commit SHAs and are labelled with `https://github.com/AtharvaPatil466/SatAi`. They are not canonical reproductions. Any evaluation genuinely rerun after migration receives a new record and the canonical repository URL/SHA; historical records are never edited into canonical claims.

## D. Intentionally excluded

- SatAi demo/video frontend history and assets, including `feat/video-integration`.
- Generated caches, Python/Node build output, local environments, secrets, credentials, downloaded imagery/datasets, checkpoints, model/adaptor weights, and archives.
- Unrelated source branches as independent transplants; `research/integration-v1` is the integrated source of truth. Other source branches are consulted only for ancestry/provenance.
- Existing canonical functionality is never removed merely because SatAi lacks it.

## Known risks

- Historical outputs are reproducible only where their referenced datasets/assets remain available; missing GPU models or imagery must be reported as not reproduced.
- Research scripts touch canonical training and grounding helpers, so focused tests must precede the full suite.
- Registry schema remains intentionally permissive for older consumers; repository URL is additive rather than a breaking required field.
- Large metadata/result JSON can increase repository size, but no dataset pixels or model binaries are included.
