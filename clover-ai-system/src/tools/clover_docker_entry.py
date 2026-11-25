#!/usr/bin/env python3
"""Unified Docker Entrypoint (`clover_docker_entry.py`) for Clover Single-Node & 6-Node 5-Petal Mesh.

Backwards-compatible with 1-command `clover_hg_serve.py` (`docker run ... clover-hg-server --model clover-35b-a3b`),
while also supporting every role in the 6-Node 5-Petal Clover Leaf + Center `k3` Hub architecture:
  * `serve` (or `--model ...`): 1-command single/multi-model server (`clover_hg_serve.py`)
  * `gateway`: Full `gateway.py` (`k3` Center Hub or local-only GPU Petal gateway + MCP 2.0 + `AGENTS.md` cycle)
  * `kimi-download-slice`: Selective ~150 GB layer-slice downloader (`kimi_pipeline_cluster.py download-slice`)
  * `kimi-worker`: Authenticated TCP Kimi K31H pipeline stage worker (`kimi_pipeline_cluster.py worker`)
  * `kimi-coordinator`: Tail stage + 5L-DSpark OpenAI `/v1` coordinator (`kimi_pipeline_cluster.py coordinator`)
  * `specialist`: Domain specialist microservice (`specialist_hg_server.py`)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        target = TOOLS_DIR / "clover_hg_serve.py"
        os.execv(sys.executable, [sys.executable, str(target), "--host", "0.0.0.0", "--port", "8000", "--model", "lfm3-omni-2b"])

    sub = argv[0]
    rest = argv[1:]

    if sub == "gateway":
        target = TOOLS_DIR / "hg4-fusion-gateway" / "gateway.py"
        os.execv(sys.executable, [sys.executable, str(target), *rest])
    if sub == "kimi-download-slice":
        target = TOOLS_DIR / "kimi_pipeline_cluster.py"
        os.execv(sys.executable, [sys.executable, str(target), "download-slice", *rest])
    if sub == "kimi-worker":
        target = TOOLS_DIR / "kimi_pipeline_cluster.py"
        os.execv(sys.executable, [sys.executable, str(target), "worker", *rest])
    if sub == "kimi-coordinator":
        target = TOOLS_DIR / "kimi_pipeline_cluster.py"
        os.execv(sys.executable, [sys.executable, str(target), "coordinator", *rest])
    if sub == "specialist":
        target = TOOLS_DIR / "specialist_hg_server.py"
        os.execv(sys.executable, [sys.executable, str(target), *rest])

    # Default (`serve` or `--model ...` flags directly): delegate to `clover_hg_serve.py`
    if sub == "serve":
        argv = rest
    if "--host" not in argv:
        argv = ["--host", "0.0.0.0", *argv]
    if "--port" not in argv and "--download-only" not in argv and "--build-only" not in argv:
        argv = ["--port", "8000", *argv]
    target = TOOLS_DIR / "clover_hg_serve.py"
    os.execv(sys.executable, [sys.executable, str(target), *argv])


if __name__ == "__main__":
    main()
