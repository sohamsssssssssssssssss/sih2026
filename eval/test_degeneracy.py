"""Unit tests for the degenerate-run guard on non-ladder suites."""

import unittest

from eval.eval import answer_matches, degenerate, summarise


def _result(expected: str, predicted: str) -> dict:
    """Build one per-sample record in the shape eval.main() produces."""
    return {
        "expected_answer": expected,
        "prediction": {"answer": predicted, "confidence": 1.0, "evidence": []},
        "correct": answer_matches(predicted, expected),
    }


class SummariseTest(unittest.TestCase):
    def test_splits_binary_from_open_ended(self) -> None:
        results = [
            _result("yes", "yes"),
            _result("no", "no"),
            _result("urban", "urban"),
            _result("2", "2"),
        ]
        summary = summarise(results)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["binary_n"], 2)
        self.assertEqual(summary["open_n"], 2)
        self.assertEqual(summary["accuracy"], 1.0)

    def test_open_and_binary_accuracy_are_independent(self) -> None:
        # Perfect on yes/no, wrong on every open-ended question.
        results = [_result("yes", "yes"), _result("no", "no")]
        results += [_result("urban", "rural"), _result("2", "7")]
        summary = summarise(results)
        self.assertEqual(summary["binary_accuracy"], 1.0)
        self.assertEqual(summary["open_accuracy"], 0.0)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_no_binary_questions_does_not_divide_by_zero(self) -> None:
        summary = summarise([_result("urban", "urban")])
        self.assertEqual(summary["binary_n"], 0)
        self.assertEqual(summary["binary_accuracy"], 0.0)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 0.0)

    def test_empty_results_are_safe(self) -> None:
        summary = summarise([])
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertIsNone(degenerate(summary))

    def test_verbose_yes_still_counts_as_yes(self) -> None:
        # answer_matches decides yes-ness, so a sentence answer is not missed.
        summary = summarise([_result("yes", "Yes, there is a road.")])
        self.assertEqual(summary["pred_yes_rate_on_binary"], 1.0)


class DegenerateTest(unittest.TestCase):
    def test_always_yes_model_on_binary_heavy_input_is_flagged(self) -> None:
        # 18 binary questions, half of them gold "no"; model says yes to all.
        results = [_result("yes", "yes") for _ in range(9)]
        results += [_result("no", "yes") for _ in range(9)]
        results += [_result("urban", "yes"), _result("2", "yes")]
        summary = summarise(results)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 1.0)
        # Aggregate accuracy looks like a plausible baseline...
        self.assertGreater(summary["accuracy"], 0.4)
        # ...but the run is a non-measurement.
        warning = degenerate(summary)
        self.assertIsNotNone(warning)
        self.assertIn("verbosity", warning)

    def test_genuine_mixed_result_is_not_flagged(self) -> None:
        results = [_result("yes", "yes") for _ in range(10)]
        results += [_result("no", "no") for _ in range(10)]
        results += [_result("urban", "urban"), _result("2", "3")]
        summary = summarise(results)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 0.5)
        self.assertIsNone(degenerate(summary))

    def test_a_weak_but_honest_baseline_is_not_flagged(self) -> None:
        # Mostly wrong, but not by saying yes to everything. Low accuracy is
        # a real measurement and must survive the guard.
        results = [_result("yes", "no") for _ in range(8)]
        results += [_result("no", "yes") for _ in range(8)]
        summary = summarise(results)
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 0.5)
        self.assertIsNone(degenerate(summary))

    def test_threshold_is_strictly_greater_than(self) -> None:
        # Exactly 0.85 is not degenerate; 0.90 is.
        at_threshold = [_result("yes", "yes") for _ in range(17)]
        at_threshold += [_result("no", "no") for _ in range(3)]
        self.assertEqual(summarise(at_threshold)["pred_yes_rate_on_binary"], 0.85)
        self.assertIsNone(degenerate(summarise(at_threshold)))

        over = [_result("yes", "yes") for _ in range(18)]
        over += [_result("no", "no") for _ in range(2)]
        self.assertEqual(summarise(over)["pred_yes_rate_on_binary"], 0.90)
        self.assertIsNotNone(degenerate(summarise(over)))


if __name__ == "__main__":
    unittest.main()
