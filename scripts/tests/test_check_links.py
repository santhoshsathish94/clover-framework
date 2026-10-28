"""Unittest discovery bridge for `test-markdown-links.py`."""
import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location("test_markdown_links_impl", Path(__file__).with_name("test-markdown-links.py"))
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
MarkdownAnchors = _MOD.MarkdownAnchors

if __name__ == "__main__":
    unittest.main()
