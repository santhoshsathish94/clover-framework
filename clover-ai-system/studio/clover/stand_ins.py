"""Stand-ins for environments without torch (the Android app and lightweight web venvs), installed before clover_loop loads."""
import sys
import types
from pathlib import Path


class Absent:
    def __init__(self, name="torch"):
        self._name = name

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return Absent(f"{self._name}.{key}")

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs and not isinstance(args[0], Absent):
            return args[0]
        return Absent(f"{self._name}()")

    def __mro_entries__(self, bases):
        return (object,)

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __repr__(self):
        return f"<{self._name}: torch is not installed>"


def _absent_module(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__getattr__ = lambda key: Absent(f"{name}.{key}")
    m.__clover_stand_in__ = True
    return m


def install(app_dir):
    app_dir = Path(app_dir)
    try:
        import torch  # noqa: F401
    except ImportError:
        torch, nn, functional = _absent_module("torch"), _absent_module("torch.nn"), _absent_module("torch.nn.functional")
        torch.nn, nn.functional = nn, functional
        sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": functional})
    if "hybrid_agent" not in sys.modules:
        pkg = types.ModuleType("hybrid_agent")
        pkg.__path__ = [str(app_dir / "hybrid_agent")]
        pkg.__clover_stand_in__ = True
        sys.modules["hybrid_agent"] = pkg
    if os.name == "nt" and not hasattr(sys, "getandroidapilevel"):
        return
    if "windows_python_tool" not in sys.modules and hasattr(sys, "getandroidapilevel"):
        import android_python_tool
        sys.modules["windows_python_tool"] = android_python_tool


import os  # noqa: E402
