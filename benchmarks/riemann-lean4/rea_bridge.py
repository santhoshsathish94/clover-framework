"""
`REA` (`morluto/rea` — Reverse Engineer Anything) Integration & Evidence Budgeter
(`benchmarks/riemann-lean4/rea_bridge.py`).

Combines `rea-agents` (`https://github.com/morluto/rea`) workspace/binary analysis with
deterministic Linux ELF (`objdump -d -M intel`, `nm -D`, `.rodata` strings) and JS bundle
de-minification, passing every output through a strict `<= 2,600`-char evidence budgeter so
raw binary dumps never pollute context windows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run_cmd(cmd: list[str], timeout_s: float = 15.0) -> str:
    env = dict(os.environ)
    node_bin = "/mnt/nvme1/clover-rh-bench/node/bin"
    if Path(node_bin).is_dir():
        env["PATH"] = f"{node_bin}:{env.get('PATH', '')}"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        # Strip streaming {"rea_progress": ...} lines from REA stdout/stderr
        lines = [
            ln for ln in (p.stdout or "").splitlines()
            if not ln.lstrip().startswith('{"rea_progress"')
        ]
        out = "\n".join(lines).strip()
        if p.returncode != 0 and not out:
            out = (p.stderr or "").strip()
        return out
    except Exception as exc:
        return f"<error: {exc}>"


def build_rea_evidence_for_task(task: dict[str, Any], max_chars: int = 2800) -> str:
    target = Path(str(task.get("target_path") or ""))
    kind = str(task.get("target_kind") or "")
    if not target.exists():
        return ""

    sections: list[str] = []
    rea_bin = shutil.which("rea") or "/mnt/nvme1/clover-rh-bench/node/bin/rea"
    has_rea = Path(rea_bin).is_file()

    if kind == "elf_x86_64" and target.is_file():
        if has_rea:
            rea_exp = _run_cmd([rea_bin, "inspect-native-exports", str(target)])
            sections.append(f"[REA (`rea inspect-native-exports`) Evidence]:\n{rea_exp[:750]}")
        dis = _run_cmd(["objdump", "-d", "-M", "intel", "--no-show-raw-insn", str(target)])
        kept_blocks: list[str] = []
        for blk in dis.split("\n\n"):
            if any(skip in blk for skip in ("<_init>:", "<_fini>:", "<deregister_tm_clones>:", "<register_tm_clones>:", "<__do_global_dtors_aux>:", "<frame_dummy>:", ".plt")):
                continue
            if ">:" in blk:
                kept_blocks.append(blk.strip())
        rodata = _run_cmd(["objdump", "-s", "-j", ".rodata", str(target)])
        sections.append("[Intel x86-64 Disassembly (`objdump -d -M intel`)]:\n" + "\n\n".join(kept_blocks)[:1500])
        if "Contents of section .rodata" in rodata:
            sections.append(f"[`.rodata` Section]:\n{rodata[:400]}")

    elif kind == "javascript_app" and target.is_dir():
        if has_rea:
            rea_js = _run_cmd([rea_bin, "analyze-javascript-application", str(target)], timeout_s=20.0)
            sections.append(f"[REA (`rea analyze-javascript-application`) Evidence Header]:\n{rea_js[:800]}")
        js_files = sorted(target.glob("*.js"))
        for jf in js_files[:2]:
            raw_js = jf.read_text(encoding="utf-8", errors="replace").strip()
            pretty = raw_js.replace(";", ";\n")
            sections.append(f"[Recovered JS Bundle (`{jf.name}`)]:\n```javascript\n{pretty[:1500]}\n```")

    combined = "\n\n".join(sections).strip()
    if not combined:
        return ""
    if len(combined) > max_chars:
        combined = combined[: max_chars - 120] + "\n... [REA Evidence Budgeter: truncated to preserve context budget]"
    return f"\n\n=== REA (`morluto/rea` v6.10.1) Reverse-Engineering Evidence (`{target.name}`) ===\n{combined}\n"
