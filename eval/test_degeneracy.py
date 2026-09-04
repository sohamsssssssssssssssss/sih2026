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

    def test_always_no_model_is_flagged(self) -> None:
        # The mirror of the always-yes case. Answering "no" to everything is
        # exactly as uninformative; it scores the complement of the prior.
        results = [_result("no", "no") for _ in range(9)]
        results += [_result("yes", "no") for _ in range(9)]
        results += [_result("urban", "no"), _result("2", "no")]
        summary = summarise(results)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 0.0)
        warning = degenerate(summary)
        self.assertIsNotNone(warning)
        self.assertIn("refusal", warning)

    def test_both_collapse_directions_are_caught(self) -> None:
        all_yes = summarise([_result("yes", "yes")] * 10 + [_result("no", "yes")] * 10)
        all_no = summarise([_result("yes", "no")] * 10 + [_result("no", "no")] * 10)
        self.assertEqual(all_yes["pred_yes_rate_on_binary"], 1.0)
        self.assertEqual(all_no["pred_yes_rate_on_binary"], 0.0)
        self.assertIsNotNone(degenerate(all_yes))
        self.assertIsNotNone(degenerate(all_no))

    def test_low_threshold_is_strictly_less_than(self) -> None:
        # Exactly 0.15 is not degenerate; 0.10 is.
        at = [_result("yes", "yes") for _ in range(3)]
        at += [_result("no", "no") for _ in range(17)]
        self.assertEqual(summarise(at)["pred_yes_rate_on_binary"], 0.15)
        self.assertIsNone(degenerate(summarise(at)))

        under = [_result("yes", "yes") for _ in range(2)]
        under += [_result("no", "no") for _ in range(18)]
        self.assertEqual(summarise(under)["pred_yes_rate_on_binary"], 0.10)
        self.assertIsNotNone(degenerate(summarise(under)))

    def test_open_only_run_is_not_flagged_by_the_low_side(self) -> None:
        # summarise() reports a yes-rate of 0.0 when there are no binary
        # questions at all. That is an absence of evidence, not a collapse,
        # and must not trip the low threshold for free.
        summary = summarise([_result("urban", "urban"), _result("2", "2")])
        self.assertEqual(summary["binary_n"], 0)
        self.assertEqual(summary["pred_yes_rate_on_binary"], 0.0)
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
