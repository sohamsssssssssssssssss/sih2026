# RULES (verbatim from MASTER_PLAN.md §0.3 and §0.2)

### 0.3 Rules (non-negotiable)
1. No number in any deck, README or UI unless it is MEASURED.
2. Write the hypothesis and the pass/fail threshold before running an eval. Commit it first.
3. Never fabricate coordinates, labels, results or citations. Unknown is written as unknown.
4. Fail closed: a tool that is not validated is registered as `experimental_unvalidated`, off by default, and the UI says so.
5. Every training run is a committed config before launch. Seed fixed. Staged sizes (tiny overfit first).
6. Long Kaggle jobs run as committed versions. Big files stay out of `/kaggle/working`. One GPU session at a time.
7. Ask Soham before: any Kaggle upload, any data deletion, any spend, any force-push, any merge to `main`.
8. Never touch Atharva's VLM training code or annotation files.
9. Resolution degradation is PSF blur, then decimate, then sensor noise. Never bicubic.
10. Controls are part of every eval (blank, shuffled, swapped). A model that scores the same on a blank image has learned nothing.
11. Datasets are deleted only after: results verified, pack hash recorded, source + rebuild steps written down.
12. The floor only goes up: `main` always runs the demo. Work lands on `main` only through a passing gate.
13. If a task is taking 2x its estimate, stop and report. Do not improvise a bigger design.

### 0.2 Session protocol (every session, every model)
1. Read `RULES.md`, `CONTEXT.md`, `TRACKER.md`. Read the plan section for the current task only.
2. State in 3 lines: the task id, the gate, what will not be touched.
3. Do one task. One branch per task: `task/<id>-<slug>`.
4. Stop at the gate. Show evidence (test output, results file, hash). Do not start the next task.
5. Update `TRACKER.md`, `CONTEXT.md`, and `RESULTS.md` if a number landed. Commit. Push.

For Opus in chat writing prompts for Claude Code: each prompt is a .txt file, opens with a 3 to 5 line context brief, names the task id, lists "do not touch", states the gate, and tells the agent to stop at the gate.

