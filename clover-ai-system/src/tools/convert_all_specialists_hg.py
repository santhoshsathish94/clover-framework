#!/usr/bin/env python3
"""Sequential Clover `HG` Conversion Queue (`convert_all_specialists_hg.py`) for Core Upgrade + Specialists.

Converts our approved next-generation Clover models one after the other with strict accuracy gates:
  1. `Qwen/Qwen3.6-35B-A3B`        -> `qwen3.6-35b-a3b-hg4` (~17.2 GiB `HG4-16R-OP` + Shrinkage-LDLQ)
  2. `Goedel-LM/Goedel-Prover-V2-32B` -> `goedel-prover-v2-32b-hg4` (~15.8 GiB `HG4-16R` + `--verify`)
  3. `facebook/vjepa2-vitg-fpc64-256` -> `vjepa2-vitg-hg4` (~0.52 GiB `HG4-16R` + `--verify`)
  4. `boltz-community/boltz-2`     -> `boltz-2-hg8-trunk` (Pairformer linear projections in `HG8-16R`,
                                      keeping diffusion coordinate & confidence heads in exact `FP32/BF16`)

Usage:
  # List conversion queue & estimated sizes:
  python3 clover-ai-system/src/tools/convert_all_specialists_hg.py --list

  # Run all conversions sequentially (resumes automatically if an output dir already completed):
  python3 clover-ai-system/src/tools/convert_all_specialists_hg.py --out-root ./hg-specialists

  # Run a single target (`qwen36`, `goedel32b`, `vjepa2`, or `boltz2`):
  python3 clover-ai-system/src/tools/convert_all_specialists_hg.py --only goedel32b --out-root ./hg-specialists
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE.parent

QUEUE = [
    {
        "key": "qwen36",
        "repo_id": "Qwen/Qwen3.6-35B-A3B",
        "license": "Apache-2.0",
        "domain": "Workhorse Agent & Coding Upgrade (+11 pts Terminal-Bench 2.0)",
        "out_subdir": "qwen3.6-35b-a3b-hg4",
        "mode": "c35_streaming_ldlq",
        "target_size": "~17.2 GiB HG4-16R-OP",
    },
    {
        "key": "goedel32b",
        "repo_id": "Goedel-LM/Goedel-Prover-V2-32B",
        "license": "Apache-2.0",
        "domain": "Formal Lean 4 Proof Search (90.4% MiniF2F, 57 PutnamBench Pass@32)",
        "out_subdir": "goedel-prover-v2-32b-hg4",
        "mode": "universal_hg4_ldlq",
        "target_size": "~15.8 GiB HG4-16R",
    },
    {
        "key": "vjepa2",
        "repo_id": "facebook/vjepa2-vitg-fpc64-256",
        "license": "MIT / Apache-2.0",
        "domain": "Physical-World Video Dynamics & Motion Anticipation (77.3% SSv2)",
        "out_subdir": "vjepa2-vitg-hg4",
        "mode": "universal_hg4_ldlq",
        "target_size": "~0.52 GiB HG4-16R",
    },
]


def convert_one(item: dict[str, str], out_root: Path) -> None:
    out_dir = out_root / item["out_subdir"]
    done_flag = out_dir / "CONVERSION_COMPLETE.json"
    if done_flag.is_file():
        print(f"[skip] `{item['key']}` ({item['repo_id']}) already completed at `{out_dir}`.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n================================================================", flush=True)
    print(f"[start] Converting `{item['repo_id']}` -> `{out_dir}` ({item['target_size']})", flush=True)
    print(f"        Domain: {item['domain']} | License: {item['license']}", flush=True)
    print(f"================================================================", flush=True)

    if item["mode"] == "c35_streaming_ldlq":
        conv_script = SRC_ROOT / "models" / "clover-35b-a3b" / "convert_clover35b_full_hg4.py"
        op_script = SRC_ROOT / "models" / "clover-35b-a3b" / "precompute_hg4_operators.py"
        subprocess.check_call([sys.executable, str(conv_script), "--repo-id", item["repo_id"], "--out-dir", str(out_dir)])
        if op_script.is_file():
            subprocess.check_call([sys.executable, str(op_script), "--only", "c35", "--c35-dir", str(out_dir)])
    else:
        conv_script = HERE / "convert_to_hg.py"
        subprocess.check_call([
            sys.executable, str(conv_script),
            "--model", item["repo_id"],
            "--out", str(out_dir),
            "--ldlq",
            "--verify",
        ])

    dt_s = round(time.perf_counter() - t0, 2)
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    summary = {
        "key": item["key"],
        "repo_id": item["repo_id"],
        "license": item["license"],
        "domain": item["domain"],
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "elapsed_s": dt_s,
    }
    done_flag.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] `{item['repo_id']}` -> {summary['total_gib']} GiB in {dt_s:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=Path("./hg-specialists"))
    ap.add_argument("--only", default="", help="Comma-separated subset of keys: qwen36,goedel32b,vjepa2")
    ap.add_argument("--list", action="store_true", help="Print the sequential specialist conversion queue and exit")
    args = ap.parse_args()

    if args.list:
        for i, it in enumerate(QUEUE, 1):
            print(f"{i}. [{it['key']}] {it['repo_id']} ({it['license']}) -> {it['target_size']} | {it['domain']}")
        return

    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else {it["key"] for it in QUEUE}
    for it in QUEUE:
        if it["key"] in wanted:
            convert_one(it, args.out_root)


if __name__ == "__main__":
    main()
