"""Unit tests for per-rung degeneracy detection on the resolution ladder.

The ladder is the one suite where an aggregate-only guard is not enough. A
model that collapses to always-"yes" at coarse GSD scores the binary base
rate, which makes the aggregate curve RISE at lower resolution — a physical
impossibility that reads as a real result. Diluted across five rungs, the
whole-run yes-rate can sit well under the threshold while individual rungs
are pure noise, so the guard has to run per rung.
"""

import unittest

from eval.eval import answer_matches, degenerate, summarise

RUNGS = (0.3, 1.0, 2.0, 5.0, 10.0)


def _record(gsd: float, expected: str, predicted: str) -> dict:
    return {
        "gsd": gsd,
        "expected_answer": expected,
        "prediction": {"answer": predicted, "confidence": 1.0, "evidence": []},
        "correct": answer_matches(predicted, expected),
    }


def _rung(gsd: float, *, always_yes: bool, n_yes: int = 70, n_no: int = 30) -> list[dict]:
    """One rung with a 70/30 yes/no gold prior, honest or collapsed."""
    out = []
    for _ in range(n_yes):
        out.append(_record(gsd, "yes", "Yes"))
    for _ in range(n_no):
        out.append(_record(gsd, "no", "Yes" if always_yes else "No"))
    # A few open-ended questions, as the real ladder has.
    for _ in range(20):
        out.append(_record(gsd, "forest", "forest"))
    return out


def _per_rung(records: list[dict]) -> dict:
    per_rung = {}
    for rung in RUNGS:
        sub = [r for r in records if r["gsd"] == rung]
        if not sub:
            per_rung[f"{rung:.1f}"] = {"accuracy": 0.0, "n": 0}
            continue
        s = summarise(sub)
        per_rung[f"{rung:.1f}"] = {**s, "warning": degenerate(s)}
    return per_rung


class PerRungGuardTest(unittest.TestCase):
    def test_collapsed_rung_is_flagged(self) -> None:
        pr = _per_rung(_rung(10.0, always_yes=True))
        self.assertIsNotNone(pr["10.0"]["warning"])
        self.assertEqual(pr["10.0"]["pred_yes_rate_on_binary"], 1.0)

    def test_honest_rung_is_not_flagged(self) -> None:
        pr = _per_rung(_rung(0.3, always_yes=False))
        self.assertIsNone(pr["0.3"]["warning"])

    def test_always_yes_scores_exactly_the_gold_prior(self) -> None:
        # This is why the aggregate curve appears to recover at coarse GSD.
        pr = _per_rung(_rung(10.0, always_yes=True, n_yes=70, n_no=30))
        self.assertAlmostEqual(pr["10.0"]["binary_accuracy"], 0.70, places=9)

    def test_whole_run_guard_misses_what_per_rung_catches(self) -> None:
        # The regression this module exists for: two collapsed rungs diluted by
        # three honest ones pass the aggregate check while being pure noise.
        records = []
        for gsd in (0.3, 1.0, 2.0):
            records += _rung(gsd, always_yes=False)
        for gsd in (5.0, 10.0):
            records += _rung(gsd, always_yes=True)

        whole = summarise(records)
        self.assertIsNone(degenerate(whole), "aggregate guard should NOT fire here")

        pr = _per_rung(records)
        flagged = [g for g, v in pr.items() if v.get("warning")]
        self.assertEqual(sorted(flagged), ["10.0", "5.0"])

    def test_empty_rung_does_not_crash(self) -> None:
        pr = _per_rung(_rung(0.3, always_yes=False))
        self.assertEqual(pr["5.0"], {"accuracy": 0.0, "n": 0})

    def test_back_compat_keys_survive_for_plot_and_kaggle_runner(self) -> None:
        # eval/plot_ladder.py and kaggle/run_ladder_baseline.py read these.
        pr = _per_rung(_rung(0.3, always_yes=False))
        for key in ("accuracy", "n"):
            self.assertIn(key, pr["0.3"])
            self.assertIn(key, pr["5.0"])


if __name__ == "__main__":
    unittest.main()


class PlotLadderTest(unittest.TestCase):
    """plot_ladder.py had no coverage; these pin the two defects that mattered."""

    @staticmethod
    def _report(per_rung: dict, model: str = "m") -> dict:
        return {"model": model, "suite": "ladder", "per_rung": per_rung}

    def _plot(self, reports: list[dict]) -> None:
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, report in enumerate(reports):
                path = Path(tmp) / f"r{i}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(str(path))
            out = Path(tmp) / "curve.png"
            result = subprocess.run(
                [sys.executable, "eval/plot_ladder.py", *paths, "--out", str(out)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file() and out.stat().st_size > 0)

    def test_empty_rung_does_not_veto_the_stratified_curves(self) -> None:
        # A rung with n=0 writes only {accuracy, n}. It must be omitted, not
        # allowed to drop every model back to the aggregate-only fallback.
        full = {"accuracy": 0.5, "n": 400, "open_accuracy": 0.3,
                "binary_accuracy": 0.7, "open_n": 200, "binary_n": 200,
                "pred_yes_rate_on_binary": 0.5, "warning": None}
        self._plot([self._report({"0.3": dict(full), "1.0": dict(full),
                                  "2.0": {"accuracy": 0.0, "n": 0},
                                  "5.0": dict(full), "10.0": dict(full)})])

    def test_multiple_models_overlay(self) -> None:
        full = {"accuracy": 0.5, "n": 400, "open_accuracy": 0.3,
                "binary_accuracy": 0.7, "open_n": 200, "binary_n": 200,
                "pred_yes_rate_on_binary": 0.5, "warning": None}
        rungs = {f"{g:.1f}": dict(full) for g in RUNGS}
        self._plot([self._report(rungs, "model-a"), self._report(rungs, "model-b")])

    def test_pre_guard_format_still_plots(self) -> None:
        self._plot([self._report({f"{g:.1f}": {"accuracy": 0.5, "n": 400} for g in RUNGS})])

    def test_flagged_rung_renders(self) -> None:
        full = {"accuracy": 0.5, "n": 400, "open_accuracy": 0.3,
                "binary_accuracy": 0.7, "open_n": 200, "binary_n": 200,
                "pred_yes_rate_on_binary": 1.0, "warning": "degenerate"}
        self._plot([self._report({f"{g:.1f}": dict(full) for g in RUNGS})])
