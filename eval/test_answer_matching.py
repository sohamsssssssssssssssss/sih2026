"""Unit tests for shared non-ladder answer scoring."""

import unittest

from eval.eval import answer_matches


class AnswerMatchingTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(answer_matches("  Urban ", "urban"))

    def test_expected_word_embedded_in_longer_text(self) -> None:
        self.assertTrue(answer_matches("The answer is urban.", "urban"))

    def test_number_word_converts_to_digit(self) -> None:
        self.assertTrue(answer_matches("There are two buildings.", "2"))

    def test_genuine_non_match(self) -> None:
        self.assertFalse(answer_matches("There is a road.", "no"))


if __name__ == "__main__":
    unittest.main()
