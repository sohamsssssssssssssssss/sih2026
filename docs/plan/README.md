# docs/plan: the project's memory

Agents keep no memory between sessions; these files are the memory.

1. At the start: read `RULES.md`, then `CONTEXT.md`, then `TRACKER.md`. Read only the `MASTER_PLAN.md` section for your task.
2. State the task id, its gate, and what you will not touch. One task per branch (`task/<id>-<slug>`) and per worktree.
3. Stop at the gate and show evidence: test output, results file, hash.
4. At the end: update `TRACKER.md` (status + evidence) and `CONTEXT.md` (only facts you checked; otherwise VERIFY). Log decisions in `DECISIONS.md` with the date.
5. `RESULTS.md` is generated: run `python3.11 docs/plan/build_results.py`. Never type a number into it by hand.
6. `MASTER_PLAN.md` changes only when Soham says so. When files disagree: repo > CONTEXT > plan.
