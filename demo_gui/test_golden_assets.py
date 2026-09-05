"""Selection-logic tests for the frozen golden demo scene.

Deliberately independent of Streamlit and model dependencies: this module
only imports the standard library and demo_gui.golden_assets, so it can run
anywhere golden_assets.py can run.
"""

import copy
import random
import unittest

from demo_gui import golden_assets

OTHER_QUESTION = "What is the dominant land cover in this image?"


def _row(question: str, correct: bool, gsd: float = golden_assets.GOLDEN_GSD) -> dict:
    return {
        "tile_id": golden_assets.GOLDEN_SCENE_ID,
        "question": question,
        "gsd": gsd,
        "correct": correct,
        "prediction": {"answer": "Yes" if correct else "No"},
    }


def _unrelated_row() -> dict:
    return {
        "tile_id": "loveda_LoveDA_images_png_1_gsd0.3",
        "question": golden_assets.GOLDEN_QUESTION,
        "gsd": golden_assets.GOLDEN_GSD,
        "correct": True,
        "prediction": {"answer": "Yes"},
    }


class GoldenResultSelectionTests(unittest.TestCase):
    def test_selects_pinned_row_among_multiple_questions_for_same_tile(self) -> None:
        """The pinned tile_id has more than one question in real artifacts;
        matching on tile_id alone would be ambiguous. Scene + question + GSD
        together must select the single correct golden row."""
        report = {
            "results": [
                _row(OTHER_QUESTION, correct=False),
                _row(golden_assets.GOLDEN_QUESTION, correct=True),
            ]
        }
        result = golden_assets.golden_result(report)
        self.assertIs(result, report["results"][1])
        self.assertEqual(result["question"], golden_assets.GOLDEN_QUESTION)
        self.assertTrue(result["correct"])

    def test_selection_is_order_independent(self) -> None:
        base_rows = [
            _row(OTHER_QUESTION, correct=False),
            _row(golden_assets.GOLDEN_QUESTION, correct=True),
            _unrelated_row(),
        ]
        for _ in range(20):
            shuffled = copy.deepcopy(base_rows)
            random.shuffle(shuffled)
            result = golden_assets.golden_result({"results": shuffled})
            self.assertEqual(result["tile_id"], golden_assets.GOLDEN_SCENE_ID)
            self.assertEqual(result["question"], golden_assets.GOLDEN_QUESTION)

    def test_missing_pinned_result_raises(self) -> None:
        report = {"results": [_row(OTHER_QUESTION, correct=False)]}
        with self.assertRaisesRegex(ValueError, "not found"):
            golden_assets.golden_result(report)

    def test_wrong_tile_id_is_not_matched(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            golden_assets.golden_result({"results": [_unrelated_row()]})

    def test_correct_must_be_explicit_boolean_true(self) -> None:
        for value in (1, "true", "false", None):
            with self.subTest(correct=value):
                row = _row(golden_assets.GOLDEN_QUESTION, correct=True)
                row["correct"] = value
                with self.assertRaisesRegex(ValueError, "incorrect"):
                    golden_assets.golden_result({"results": [row]})
        row = _row(golden_assets.GOLDEN_QUESTION, correct=True)
        del row["correct"]
        with self.assertRaisesRegex(ValueError, "incorrect"):
            golden_assets.golden_result({"results": [row]})

    def test_empty_results_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            golden_assets.golden_result({"results": []})

    def test_duplicate_pinned_result_raises(self) -> None:
        report = {
            "results": [
                _row(golden_assets.GOLDEN_QUESTION, correct=True),
                _row(golden_assets.GOLDEN_QUESTION, correct=True),
            ]
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            golden_assets.golden_result(report)

    def test_incorrect_pinned_result_raises(self) -> None:
        report = {"results": [_row(golden_assets.GOLDEN_QUESTION, correct=False)]}
        with self.assertRaisesRegex(ValueError, "incorrect"):
            golden_assets.golden_result(report)

    def test_wrong_gsd_is_not_matched(self) -> None:
        """A row with the right tile_id and question but a different GSD
        (e.g. a coarser rung using the same base tile naming) must not be
        picked as the golden row."""
        report = {
            "results": [
                _row(golden_assets.GOLDEN_QUESTION, correct=True, gsd=1.0),
            ]
        }
        with self.assertRaisesRegex(ValueError, "not found"):
            golden_assets.golden_result(report)


if __name__ == "__main__":
    unittest.main()
