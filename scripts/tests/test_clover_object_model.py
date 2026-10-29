"""Unit tests for the hyphenated complete-English Clover Object Model (`clover-object-model.py`)."""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "clover-ai-system" / "src" / "tools" / "hybrid-ncm-k3"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from clover_hyphen_runtime import load_hyphenated_module

_IMPL_SUITE = load_hyphenated_module(
    Path(__file__).with_name("test-clover-object-model.py"),
    module_alias="test_clover_object_model_hyphen_suite",
)
TestCloverObjectModel = _IMPL_SUITE.TestCloverObjectModel

if __name__ == "__main__":
    unittest.main()
