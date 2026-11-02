"""Regression tests for CSI v0.1 (contract shape and the written correspondences). These are not an evaluation:
evaluation is run_e1.py on generated held-out pairs. Run: python -m unittest discover -s tests (from prototype/)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csi import CSI_VERSION, analyze, compare  # noqa: E402


def prop(text):
    return analyze(text)["readings"][0]["propositions"][0]


class Contract(unittest.TestCase):
    def test_shape(self):
        a = analyze("The child opened the door.")
        self.assertEqual(a["csi_version"], CSI_VERSION)
        for key in ("text", "readings", "ambiguity", "unsupported"):
            self.assertIn(key, a)
        p = a["readings"][0]["propositions"][0]
        for key in ("predicate", "roles", "polarity", "tense", "aspect", "modality", "voice", "embedded_under", "provenance"):
            self.assertIn(key, p)

    def test_unsupported_is_reported_not_guessed(self):
        a = analyze("Hello there.")
        self.assertTrue(a["unsupported"])


class Correspondences(unittest.TestCase):
    def test_passive_roles(self):
        p = prop("The door was opened by the child.")
        self.assertEqual((p["voice"], p["roles"]["ARG0"]["head_lemma"], p["roles"]["ARG1"]["head_lemma"]),
                         ("passive", "child", "door"))

    def test_tense_aspect_modality_polarity(self):
        self.assertEqual(prop("The child will open the door.")["tense"], "future")
        self.assertEqual(prop("The child was opening the door.")["aspect"], "progressive")
        self.assertEqual(prop("The child might open the door.")["modality"], "might")
        self.assertEqual(prop("The child did not open the door.")["polarity"], "negative")

    def test_attempt_is_embedded(self):
        p = prop("The child tried to open the door.")
        self.assertEqual((p["predicate"]["lemma"], p["embedded_under"]), ("open", "try"))


class Comparison(unittest.TestCase):
    def test_equivalent(self):
        self.assertTrue(compare("The child opened the door.", "The door was opened by the child.")["equivalent"])

    def test_distinct(self):
        for b in ("The child did not open the door.", "The child will open the door.",
                  "The child tried to open the door.", "The door opened the child."):
            self.assertFalse(compare("The child opened the door.", b)["equivalent"], b)


if __name__ == "__main__":
    unittest.main()
