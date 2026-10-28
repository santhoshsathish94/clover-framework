"""Unittest discovery bridge for `test-website-pages.py`."""
import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location("test_website_pages_impl", Path(__file__).with_name("test-website-pages.py"))
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
SiteChecks = _MOD.SiteChecks

if __name__ == "__main__":
    unittest.main()
