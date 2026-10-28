"""Compatibility import shim for `verify-direction-b-data-movement.py`."""
from pathlib import Path
import sys

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from clover_hyphen_runtime import load_hyphenated_module

_loaded = load_hyphenated_module(_TOOLS_DIR / "verify-direction-b-data-movement.py", module_alias="verify_direction_b_data_movement")
globals().update({k: v for k, v in _loaded.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    if hasattr(_loaded, "main") and callable(_loaded.main):
        _loaded.main()
