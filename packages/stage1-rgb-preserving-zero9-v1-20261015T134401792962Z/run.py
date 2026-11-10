"""Stage-1 T4 package entry point.

Three modes:

- ``all``    (default) full evidence session: verify -> smoke -> resume ->
             overfit -> extrapolate. Stops at the first failed stage and
             writes one JSON per stage plus session-summary.json.
- ``full``   one resumable shard-backed training session (loop 8). Writes
             <output>.json and <output>.pt, and exits cleanly at the
             wall-clock guard; rerun with the same --output to resume.
- ``smoke``  forward arguments directly to scripts.run_stage1_smoke for a
             single configured training run (debugging aid; not the evidence
             path).

Examples::

    python run.py all --checkpoint /kaggle/input/<weights> \
        --patch-root /kaggle/working/BigEarthNet-S2 --out-dir stage-results
    python run.py all --mode verify --checkpoint <dir> --patch-root <dir>
    python run.py full --config config-full.yaml --checkpoint-dir <dir> \
        --train-shards /kaggle/input/stage1-train-shards \
        --eval-shards /kaggle/input/stage1-eval-shards \
        --output stage-results/full-session1.json \
        --wall-clock-limit-seconds 39600
    python run.py smoke --config config.yaml --checkpoint <dir> \
        --patch-root <dir> --output smoke.json
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from scripts.run_stage1_t4 import main as run_all
        run_all()
    elif mode == "full":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from scripts.run_stage1_full import main as run_full
        run_full()
    elif mode == "smoke":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from scripts.run_stage1_smoke import main as run_smoke
        run_smoke()
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected 'all', 'full', or 'smoke')")


if __name__ == "__main__":
    main()