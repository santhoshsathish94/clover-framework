#!/usr/bin/env python3
"""
Long-Timeout Runner Wrapper for NVMe-Streamed 2.8T `Kimi K3.1 Pro` (`benchmarks/riemann-lean4/run_long_timeout_wrapper.py`).

Preserves the frozen `chmod 444` `run_sequential_phase.py` (`SHA-256 = 78fd110a1be269b11dab82f4e3f36283cff5267ee47bf22cf6e8ecab1032c089`)
100% byte-untouched while raising the client HTTP socket timeout from `240.0s` to `3600.0s` so single-turn
93-layer NVMe-streamed `2.8T` completions (`450s - 1405s`) do not prematurely time out on the client socket.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sequential_phase as rsp  # noqa: E402

_ORIG_CHAT_V1 = rsp.chat_v1
_TIMEOUT_S = float(os.environ.get("CLOVER_BENCH_HTTP_TIMEOUT_S", "3600.0"))


def _long_timeout_chat_v1(*args, **kwargs):
    kwargs.setdefault("timeout_s", _TIMEOUT_S)
    return _ORIG_CHAT_V1(*args, **kwargs)


rsp.chat_v1 = _long_timeout_chat_v1

if __name__ == "__main__":
    rsp.main()
