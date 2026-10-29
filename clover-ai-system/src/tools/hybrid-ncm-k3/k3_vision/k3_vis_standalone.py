"""Standalone import facade for `modeling_kimi_k3` (eliminates 16 duplicate class definitions)."""

try:
    from .modeling_kimi_k3 import *  # noqa: F401,F403
except ImportError:
    from modeling_kimi_k3 import *  # noqa: F401,F403
