import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inference
from src.contracts import is_valid_cot
from src.interf0.model import BACKGROUND_ANSWER, TISSUE_ANSWER
from src.interf1.model import _emit_retrieval


VALID_COT = [
    {
        "question": "What organ is represented?",
        "answer": "Lung",
        "next_question": "",
    }
]


class ContractTests(unittest.TestCase):
    def test_cot_contract(self):
        self.assertTrue(is_valid_cot(VALID_COT))
        self.assertFalse(is_valid_cot([]))
        self.assertFalse(
            is_valid_cot([{"question": "", "answer": "Lung", "next_question": ""}])
        )

    def test_retrieval_matches_submitted_exemplar_behavior(self):
        invalid_cot = [{"question": "", "answer": "x", "next_question": ""}]
        state = {
            "table": {
                "organ_dx": {"lung|||adenocarcinoma": VALID_COT},
                "organ": {"lung": VALID_COT},
                "__global__": VALID_COT,
            },
            "bank": {
                "embs": __import__("numpy").array([[1.0, 0.0]], dtype="float32"),
                "wids": __import__("numpy").array(["case-1"]),
                "indices_by_group": {
                    ("lung", "adenocarcinoma"): __import__("numpy").array([0])
                },
                "cots": {"case-1": invalid_cot},
            },
        }
        result = _emit_retrieval(
            state,
            "lung",
            "adenocarcinoma",
            __import__("numpy").array([1.0, 0.0], dtype="float32"),
        )
        self.assertEqual(result, invalid_cot)
        self.assertFalse(is_valid_cot(result))

    def test_output_write_failure_is_reported(self):
        with patch("inference.write_json_file", side_effect=OSError("full")):
            self.assertFalse(inference._safe_write(Path("/output/result.json"), {}))

    def test_interface_zero_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "roi.png"
            question_path = Path(tmp) / "question.json"
            question_path.write_text(json.dumps("Is tissue present?"))

            from PIL import Image

            Image.new("RGB", (20, 20), "white").save(image_path)
            self.assertEqual(
                inference.predict_visual_context_response(
                    question_path=question_path,
                    roi_image_path=image_path,
                ),
                BACKGROUND_ANSWER,
            )
            Image.new("RGB", (20, 20), "black").save(image_path)
            self.assertEqual(
                inference.predict_visual_context_response(
                    question_path=question_path,
                    roi_image_path=image_path,
                ),
                TISSUE_ANSWER,
            )


if __name__ == "__main__":
    unittest.main()
