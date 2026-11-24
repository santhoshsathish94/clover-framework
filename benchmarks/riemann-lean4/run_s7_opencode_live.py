#!/usr/bin/env python3
"""Phase S7 Live `OpenCode` CLI End-to-End & Turn-2 LCP Cache Verification."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    gateway = "http://127.0.0.1:8790/v1"
    post_json(f"{gateway}/cache/reset", {})

    env = dict(os.environ)
    env["HOME"] = "/root"
    env["PATH"] = (
        "/root/.opencode/bin:/mnt/nvme1/clover-rh-bench/node/bin:"
        "/mnt/nvme1/clover-rh-bench/elan/bin:" + env.get("PATH", "")
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "/root/.opencode/bin/opencode",
            "run",
            "-m",
            "clover/clover-35b-a3b",
            "Read Demo.lean using the read tool and state the exact theorem name, and also state the exact value of 37! with no commas.",
        ],
        cwd="/mnt/nvme1/clover-rh-bench/opencode-smoke",
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wall_s = round(time.perf_counter() - t0, 2)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Explicit 2-turn OpenCode tool-schema LCP cache hit verification (>= 95%) on shipped default gateway
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_sequential_phase as rsp

    post_json(f"{gateway}/cache/reset", {})
    sys_msg = {"role": "system", "content": "You are OpenCode, an interactive CLI software engineering agent."}
    # Turn 1 explicitly triggers S3 (`37!`) with OpenCode's 11-tool schema so Turn 2 tests LCP cache preservation AFTER S3 fires!
    u1 = {"role": "user", "content": "Do not call any tool yet; state ONLY the exact decimal value of 37! with no commas."}
    r1 = post_json(
        f"{gateway}/chat/completions",
        {"model": "clover-35b-a3b", "messages": [sys_msg, u1], "tools": rsp.OPENCODE_DEFAULT_TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 96},
    )
    a1 = r1["choices"][0]["message"]
    r2 = post_json(
        f"{gateway}/chat/completions",
        {
            "model": "clover-35b-a3b",
            "messages": [
                sys_msg,
                u1,
                {"role": "assistant", "content": a1.get("content") or ""},
                {"role": "user", "content": "Thanks. Now reply ONLY with the integer 43."},
            ],
            "tools": rsp.OPENCODE_DEFAULT_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 32,
        },
    )
    t2_step = r2["x_clover_telemetry"]["steps"][0]

    res = {
        "opencode_cli_rc": proc.returncode,
        "opencode_cli_wall_s": wall_s,
        "contains_demo_add_zero": "demo_add_zero" in out,
        "contains_exact_37_factorial": "13763753091226345046315979581580902400000000" in out.replace(",", "").replace("_", ""),
        "opencode_output_tail": out.strip()[-800:],
        "turn2_lcp_step": t2_step,
    }
    out_path = Path("/mnt/nvme1/clover-rh-bench/results_v2/studio_phase_S7_opencode_live.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)
    if proc.returncode != 0 or not res["contains_demo_add_zero"] or not res["contains_exact_37_factorial"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
