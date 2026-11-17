# DECISIONS: dated log (newest first)

## 16 Nov 2026
- **(a) T13 uses its own worktree** (`~/sih2026-adapt`), because `feature/ben-landcover` is checked out in `~/sih2026-ben`.
- **(b) T13 keeps the 19-output BEN classifier.** Each coarse logit is the max of its member logits, the same as `ben/transfer.py`.
- **(c) T13 metric rules.**
  - Any-positive val mAP is used only to choose the epoch.
  - Every reported metric uses classes with ≥10 positives.
  - No arm is chosen as the winner; all arms are reported.
- **(d) T14 raster alignment.** Raster transforms are accepted if they agree within 1e-6 pixel. No resampling.
- **(e) Sen1Floods11 licence is unclear.** STAC says "proprietary", so the pack may not be redistributed until the licence is confirmed.
- **(f) Agent split.** Claude Code takes judgement-heavy tasks and Codex takes tightly specified build tasks. One task per branch and worktree, and each agent reviews the other's gate.
- **(g) Tool status.** T06 `core/schema.py` `ToolStatus` is the source of truth. The backend `CapabilityStatus` will be derived from it in T07:
  - validated / experimental_unvalidated → AVAILABLE, with the status shown as a label;
  - disabled → UNAVAILABLE;
  - unregistered → NOT_IMPLEMENTED.
- **(h) Public canonical JSON.** T07 will expose `orchestrator/trace.py`'s canonical JSON (`_canonical`, now used by `core.schema.scene_id_for`) under a public name.
- **(i) Test environment.** The python3.11 env is missing `peft`, `streamlit` and `pyarrow`. The 7 failures and 12 import errors that existed before T06 are to be fixed in T05.
- **Master Plan v1.0 replaces v5.** v5 and annexes A–E stay in `docs/plan/` for history, and v5 carries a "superseded" line.
- **T04 continues on `task/T04-plan-files`.** No new `task/T04-plan-docs` branch. The worktree was moved from `~/sih2026-plan` to `~/sih2026-docs`.
- **Branch pushes (T01).** Soham approved pushing the 8 work branches, with no merges. All were already on origin at the same sha. Every new branch push still needs approval.
- **India transfer pack.**
  - Whole WorldCover tiles were downloaded under a 26 GB cap (25.05 GB used).
  - Facts that differ from the brief: S1 is already dB, and the S2 band order is B04,B03,B02,B08.
  - Macro mAP uses a fixed class set per group (≥10 positives), and group differences use one common class set.
- **Detector counting is not validated.** DIOR val exact match at 0.35 is 0.324, against 0.500 for always-0 (1837259).

## 18 Nov 2026

- **T15 flood U-Net parked.** Soham parked the U-Net after S1-only scored 0.897 on eight chips (gate >0.9). T15d D1 failed: S1+S2 scored 0.965 on eight chips but 0.744 on the worst chip (required >0.95). S2 alone scored 0.842 on that chip. D2 points to the pipeline under the registered rule. The cause remains unresolved. References: preregistration `7daad8d`, diagnostic branch and report-reader reference `7fd5d12`, and parked note `f72523f`. Resume only by Soham's decision.
- **Flood answers use per-scene Otsu.** `flood-otsu` remains `experimental_unvalidated`. The fixed method scored 0.631 India-test IoU against its 0.65 gate. Shipped B1 scored 0.704. The Otsu method was not tuned and no threshold was relaxed.
