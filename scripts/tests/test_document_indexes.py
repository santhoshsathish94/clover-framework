from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DocumentIndexes(unittest.TestCase):
    def assert_index_covers(self, directory):
        folder = ROOT / directory
        index = folder / "README.md"
        destinations = {
            (folder / target.split("#", 1)[0]).resolve()
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", index.read_text(encoding="utf-8"))
            if not target.startswith(("https:", "http:", "#"))
        }
        missing = [path.relative_to(ROOT).as_posix() for path in folder.rglob("*.md")
                   if path != index and path.resolve() not in destinations]
        self.assertEqual(missing, [], "Documents missing from their maintained index")

    def test_k3_research_index_is_complete(self):
        self.assert_index_covers("research/k3")

    def test_k3_implementation_index_is_complete(self):
        self.assert_index_covers("clover-k3/docs")

    def test_legacy_forwarding_locations_are_removed(self):
        for relative in ("why-clover-is-important.md", "k3-analysis", "hypothesis",
                         "clover-k3/clover-k3-equation.md", "clover-k3/clover-k3-proof.md",
                         "clover-k3/clover-k3-scaling.md"):
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())

    def test_k3_research_root_contains_only_its_index(self):
        self.assertEqual(sorted(path.name for path in (ROOT / "research/k3").glob("*.md")),
                         ["README.md"])

    def test_core_documents_have_purpose_folders(self):
        for relative in ("docs/framework", "docs/guides", "docs/governance",
                         "docs/reference", "clover-k3/docs/scaling",
                         "clover-k3/docs/evidence", "clover-k3/docs/reference"):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()