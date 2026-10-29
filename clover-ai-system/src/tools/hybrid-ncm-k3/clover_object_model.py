"""Import bridge for `clover-object-model.py` (loads the hyphenated `# coding: clover-hyphen` module)."""

from pathlib import Path
from clover_hyphen_runtime import load_hyphenated_module

_HYPHENATED_MODEL_PATH = Path(__file__).with_name("clover-object-model.py")
_loaded_module = load_hyphenated_module(_HYPHENATED_MODEL_PATH, module_alias="clover_object_model_impl")

globals().update({k: v for k, v in _loaded_module.__dict__.items() if not k.startswith("__")})
