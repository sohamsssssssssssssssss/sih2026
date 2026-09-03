"""Guard the Stage 0 baseline against silent scoring drift.

results/qwen2.5vl-3b__rsvqa__20260903T175900Z.json stores all 10,004
predictions from the full RSVQA-LR test split run. Re-scoring those stored
predictions with the CURRENT matcher is deterministic and needs no GPU, so
any future change to answer_matches() shows up here as a failing test rather
than as a number in the README that quietly stopped being true.

If this test fails, the matcher changed and the baseline needs restating:
re-score, update the README's Baseline table and results/RESCORE_NOTE.md,
then update the constants below. Do NOT edit the results JSON — it records
what was run, not what we currently believe.
"""

import json
import unittest
from pathlib import Path

from eval.eval import answer_matches, summarise

BASELINE_JSON = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "qwen2.5vl-3b__rsvqa__20260903T175900Z.json"
)

# Expected under the post-daab09b matcher ("1" counts as "yes").
EXPECTED_ACCURACY = 0.514194
EXPECTED_BINARY_ACCURACY = 0.667098
EXPECTED_OPEN_ACCURACY = 0.165080
EXPECTED_N = 10004

# As originally measured, pre-daab09b. Kept so the test also pins the size of
# the matcher change rather than just its result.
AS_MEASURED_ACCURACY = 0.513095
FLIPPED_SAMPLES = 11


class BaselineRescoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        cls.records = cls.report["results"]

    def _rescored(self) -> list[dict]:
        return [
            {
                **record,
                "correct": answer_matches(
                    record["prediction"]["answer"], record["expected_answer"]
                ),
            }
            for record in self.records
        ]

    def test_committed_baseline_is_intact(self) -> None:
        self.assertEqual(len(self.records), EXPECTED_N)
        self.assertEqual(self.report["summary"]["n"], EXPECTED_N)
        # The stored file must keep its as-measured numbers untouched.
        self.assertAlmostEqual(
            self.report["summary"]["accuracy"], AS_MEASURED_ACCURACY, places=6
        )

    def test_current_matcher_reproduces_the_stated_figures(self) -> None:
        summary = summarise(self._rescored())
        self.assertAlmostEqual(summary["accuracy"], EXPECTED_ACCURACY, places=6)
        self.assertAlmostEqual(
            summary["binary_accuracy"], EXPECTED_BINARY_ACCURACY, places=6
        )
        self.assertAlmostEqual(
            summary["open_accuracy"], EXPECTED_OPEN_ACCURACY, places=6
        )

    def test_open_accuracy_is_invariant_under_the_matcher_change(self) -> None:
        # The headline metric did not move when the matcher changed. If a
        # future matcher revision breaks this, the project's primary number is
        # no longer matcher-independent and that needs saying out loud.
        summary = summarise(self._rescored())
        self.assertAlmostEqual(
            summary["open_accuracy"],
            self.report["summary"]["open_accuracy"],
            places=9,
        )

    def test_exactly_the_known_samples_flip(self) -> None:
        flipped = [
            record
            for record in self.records
            if record["correct"]
            != answer_matches(
                record["prediction"]["answer"], record["expected_answer"]
            )
        ]
        self.assertEqual(len(flipped), FLIPPED_SAMPLES)
        # All were gold "yes" answered "1", and all gained credit.
        self.assertTrue(
            all(record["expected_answer"].strip().lower() == "yes" for record in flipped)
        )
        self.assertTrue(
            all(record["prediction"]["answer"].strip() == "1" for record in flipped)
        )
        self.assertTrue(all(not record["correct"] for record in flipped))

    def test_baseline_run_is_not_degenerate_under_current_matcher(self) -> None:
        from eval.eval import degenerate

        self.assertIsNone(degenerate(summarise(self._rescored())))


if __name__ == "__main__":
    unittest.main()
