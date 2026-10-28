import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "check_links", Path(__file__).resolve().parents[1] / "check-links.py"
)
checker = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(checker)


class MarkdownAnchors(unittest.TestCase):
    def anchors(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            path.write_text(text, encoding="utf-8")
            return checker.anchors(path)

    def test_duplicate_headings_have_numbered_anchors(self):
        self.assertEqual(
            self.anchors("# Measured\n## Measured\n### Measured\n"),
            {"measured", "measured-1", "measured-2"},
        )

    def test_literal_suffix_collision_is_unique(self):
        self.assertEqual(
            self.anchors("# Measured\n# Measured-1\n# Measured\n"),
            {"measured", "measured-1", "measured-2"},
        )

    def test_fenced_headings_are_not_anchors(self):
        self.assertEqual(
            self.anchors("```text\n# Not a heading\n```\n# Real\n"),
            {"real"},
        )

    def test_explicit_anchor_is_retained(self):
        self.assertEqual(self.anchors('<a id="legacy"></a>\n'), {"legacy"})

    def test_missing_anchor_stays_missing(self):
        self.assertNotIn("missing", self.anchors("# Existing\n"))


if __name__ == "__main__":
    unittest.main()