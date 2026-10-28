import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "check_site", Path(__file__).resolve().parents[1] / "check-site.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SiteChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write("VERSION", "3.1.0\n")
        self.write("site/styles.css", "body {}")
        self.write("site/index.html", self.page("./", ""))

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def page(self, home, body, version="3.1.0"):
        prefix = "../" if home == "../" else ""
        return ('<!doctype html><html><head><link rel="stylesheet" href="'
                + prefix + 'styles.css?v=' + version + '"></head><body>'
                '<nav class="site-nav"><a href="' + home + '">Home</a></nav>'
                '<main id="main">' + body + '</main></body></html>')

    def errors(self):
        return checker.check_site(self.root)["errors"]

    def test_valid_pages_normalize_relative_navigation(self):
        self.write("site/new/index.html", self.page("../", '<a href="../#main">Main</a>'))
        self.assertEqual(self.errors(), [])
        self.assertEqual(checker.check_site(self.root)["pages"], 2)

    def test_new_page_navigation_is_checked(self):
        self.write("site/new/index.html", self.page("./", ""))
        self.assertTrue(any("navigation differs" in error for error in self.errors()))

    def test_absent_navigation_fails(self):
        self.write("site/new/index.html", "<html><body>No navigation</body></html>")
        self.assertTrue(any("expected one primary" in error for error in self.errors()))

    def test_cross_page_anchor_fails(self):
        self.write("site/new/index.html", self.page("../", '<a href="../#missing">Bad</a>'))
        self.assertTrue(any("dead anchor" in error for error in self.errors()))

    def test_missing_asset_fails(self):
        self.write("site/index.html", self.page("./", '<img src="missing.png">'))
        self.assertTrue(any("missing or outside" in error for error in self.errors()))

    def test_asset_version_must_match_release(self):
        self.write("site/index.html", self.page("./", "", version="2.0.0"))
        self.assertTrue(any("asset version" in error for error in self.errors()))

    def test_shared_asset_divergence_fails(self):
        self.write("assets/reference-example.svg", "original")
        self.write("site/assets/reference-example.svg", "changed")
        self.assertTrue(any("shared asset differs" in error for error in self.errors()))

    def test_nested_shared_asset_is_compared(self):
        self.write("assets/evidence/reference-example.svg", "original")
        self.write("site/assets/reference-example.svg", "original")
        self.assertEqual(self.errors(), [])
        self.assertEqual(checker.check_site(self.root)["shared_assets"], 1)
        self.write("site/assets/reference-example.svg", "changed")
        self.assertTrue(any("shared asset differs" in error for error in self.errors()))

    def test_ambiguous_canonical_asset_fails(self):
        self.write("assets/reference-example.svg", "original")
        self.write("assets/evidence/reference-example.svg", "original")
        self.write("site/assets/reference-example.svg", "original")
        self.assertTrue(any("ambiguous canonical" in error for error in self.errors()))

    def test_redirect_without_navigation_is_checked(self):
        self.write("site/old/index.html", '<html><head><meta http-equiv="refresh" '
                   'content="0; url=../"></head><body></body></html>')
        self.assertEqual(self.errors(), [])
        self.write("site/old/index.html", '<html><head><meta http-equiv="refresh" '
                   'content="0; url=../missing/"></head><body></body></html>')
        self.assertTrue(any("missing or outside" in error for error in self.errors()))

    def test_malformed_html_fails(self):
        self.write("site/index.html", self.page("./", "<div></span>"))
        self.assertTrue(self.errors())


if __name__ == "__main__":
    unittest.main()