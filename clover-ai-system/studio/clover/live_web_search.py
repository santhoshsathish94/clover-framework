"""Compatibility import shim for `live-web-search-tool.py`."""
from pathlib import Path
import sys
import unittest

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from clover_hyphen_runtime import load_hyphenated_module

_loaded = load_hyphenated_module(_TOOLS_DIR / "live-web-search-tool.py", module_alias="live_web_search_tool")
globals().update({k: v for k, v in _loaded.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    if hasattr(_loaded, "main") and callable(_loaded.main):
        _loaded.main()
    elif "live-web-search-tool.py".startswith("test-"):
        unittest.main()
