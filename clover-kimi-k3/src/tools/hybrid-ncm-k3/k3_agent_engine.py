"""Compatibility import shim for `clover-agent-engine.py`."""
from pathlib import Path
import sys
import unittest

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from clover_hyphen_runtime import load_hyphenated_module

_loaded = load_hyphenated_module(_TOOLS_DIR / "clover-agent-engine.py", module_alias="clover_agent_engine")
globals().update({k: v for k, v in _loaded.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    if hasattr(_loaded, "main") and callable(_loaded.main):
        _loaded.main()
    elif "clover-agent-engine.py".startswith("test-"):
        unittest.main()
