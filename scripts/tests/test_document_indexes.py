"""Unittest discovery bridge for `test-document-indexes.py`."""
import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location("test_document_indexes_impl", Path(__file__).with_name("test-document-indexes.py"))
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
DocumentIndexes = _MOD.DocumentIndexes

if __name__ == "__main__":
    unittest.main()
