"""Live check of the three fixes of 2026-11-09 (after the bridge A/B): real OmniNative.run_turn + LayerKit (memory,
ontology, compiler, ncm; C2/C3 parts of the install), live LFM3-Omni-2B, greedy; fresh memory and curiosity per turn;
routed by semantics.route as the server does; permissions deny every permissioned tool (nothing on this PC is read).
PREDICTIONS (written before the run):
  B4 Always ("Where is my project plan?", note names plan.md): no file tool offered, no tool call, answer names plan.md
     without "couldn't read" (as before the bridge fix).
  B1, B2 Always: calculate still offered; B1 correct (30,626.08), B2 correct (237,620).
  C3 ("Who is my manager now?" after "My manager is Priya." and "Correction: my manager is now Arun; ..."): Arun.
  B1 ask: if Clover's arithmetic check fires, its findings contain no number cut at a thousands separator and no line
     break (no prediction on whether the model then fixes its number).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import bridge_fix_live as B

FILE_TOOLS = {"read_file", "list_directory", "search_files", "file_info", "write_file"}
CASES = [  # (id, mode, notes, question, check)
    ("B4", "always", B.QUESTIONS[3][1], B.QUESTIONS[3][2],  # offered None = every tool offered
     lambda r, offered, calls: "plan.md" in r.lower() and "couldn" not in r.lower() and offered is not None
     and not (set(offered) & FILE_TOOLS) and not calls),
    ("B1", "always", B.QUESTIONS[0][1], B.QUESTIONS[0][2], lambda r, offered, calls: B.QUESTIONS[0][3](r) and "calculate" in (offered or ())),
    ("B2", "always", B.QUESTIONS[1][1], B.QUESTIONS[1][2], lambda r, offered, calls: B.QUESTIONS[1][3](r) and "calculate" in (offered or ())),
    ("C3", "ask", ["My manager is Priya.", "Correction: my manager is now Arun; Priya moved to another team."],
     "Who is my manager now?", lambda r, offered, calls: "arun" in r.lower()),
    ("B1", "ask", B.QUESTIONS[0][1], B.QUESTIONS[0][2], None),
]
SPLIT = re.compile(r"(?<![\d.,])\d{1,3},(?!\d{3})|(?<![\d,])000\b")


def main():
    import lfm3_native
    import omni_native
    import permissions
    import semantics
    tmp = Path(tempfile.mkdtemp(prefix="consc-fixes3-"))
    B.SANDBOX.mkdir(parents=True, exist_ok=True)
    for name in ("tools", "data"):
        if not (B.SANDBOX / name).exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(B.SANDBOX / name), str(Path.home() / "CloverAI" / name)],
                           check=True, capture_output=True)
    perms = permissions.PermissionStore(tmp / "permissions.json")
    perms.update({t: "deny" for t in permissions.TOOLS})
    eng = omni_native.OmniNative(B.SANDBOX, perms, max_tokens=8192, engine="gpu")
    offered = []
    real_sp = eng.system_prompt

    def spy(tools_on=True, asked=None):
        offered.append(list(asked) if (tools_on and asked is not None) else ([] if not tools_on else None))
        return real_sp(tools_on, asked)

    eng.system_prompt = spy
    offer = [t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS]
    rows = []
    try:
        for cid, mode, notes, q, chk in CASES:
            kit = B.lfm3_layers.LayerKit(B.SANDBOX, memory_root=tmp / f"{mode}_{cid}" / "growth" / "knowledge")
            kit.permissions_map = {"memory": "allow", "files": "deny", "code": "deny", "internet": "deny"}
            for n in notes:
                kit.remember(n, conversation_id="setup")
            frame = semantics.interpret(q, lfm3_native.RUNTIME)
            kit.meaning = frame
            only, tools_on = semantics.route(frame, mode, True, offer=offer, said=q if mode == "ask" else None)
            events, t0 = [], time.perf_counter()
            offered.clear()
            res = eng.run_turn([], q, tools=tools_on, only_tools=only, layers=B.LAYERS, kit=kit, layer_query=q,
                               on_event=events.append, final_on_limit=True, force_answer=True, meaning=frame)
            reply = res.get("reply") or ""
            calls = [{"tool": tc.get("tool"), "ran": tc.get("ran"), "result": str(tc.get("result") or "")[:200]}
                     for tc in res.get("tool_calls") or ()]
            cycle = [x for x in (res.get("layers") or []) if isinstance(x, dict) and x.get("layer") == "cycle"]
            findings = [f for c in cycle for f in dict.fromkeys((c.get("first_findings") or []) + (c.get("final_findings") or []))]
            first = offered[0] if offered else None
            row = {"id": cid, "mode": mode, "offered_first": first, "tool_calls": calls, "reply": reply[:1500],
                   "cycle_findings": findings, "s": round(time.perf_counter() - t0, 1),
                   "memory_block_entries": next((e.get("entries") for e in events if e.get("layer") == "memory"), None)}
            row["passed"] = bool(chk(reply, first, calls)) if chk else None
            if cid == "B1" and mode == "ask":  # None: the check produced no finding, so this was not exercised
                row["findings_well_formed"] = (all(not SPLIT.search(f) and "\n" not in f for f in findings)
                                               if findings else None)
            rows.append(row)
            print(f"{mode:6} {cid} passed={row['passed']} offered={first} calls={[c['tool'] for c in calls]} "
                  f"findings={len(findings)}", flush=True)
    finally:
        eng.gpu.close()
        for name in ("tools", "data"):
            subprocess.run(["cmd", "/c", "rmdir", str(B.SANDBOX / name)], capture_output=True)
    (B.HERE / "results" / "fixes3_live.json").write_text(json.dumps({"rows": rows}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
