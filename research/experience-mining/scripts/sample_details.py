"""k3, read-only: details for the sample histories (synthetic drive messages only).
1) run 5 (CSI on): the pens conversation as Clover recorded it (session, outcomes, execution log, commits);
2) run 3 (after A, final): the tracer's call sequence inside the "try again" turn and the Python turn;
3) run 5: the remember -> dinner chain (commits, memory ids)."""
import json
import re
import subprocess
from pathlib import Path

O = Path("/opt/clover-k3/core-observe")
R5 = O / "fw-base-csi-on/clover-memory-repo"


def git(repo, *a):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *a], capture_output=True, text=True).stdout


pens = next(p for p in (R5 / "sessions/2026-11").glob("obs-pens-*.jsonl"))
sid = pens.stem
print("== 1. run 5 pens session", sid)
for line in open(pens, encoding="utf-8"):
    r = json.loads(line)
    print("   session:", r["n"], r["role"], r["at"], repr(r["text"][:110]))
for p in R5.glob("outcomes/*/*.jsonl"):
    for line in open(p, encoding="utf-8"):
        if sid in line:
            print("   outcome:", line.strip()[:220])
for p in sorted(R5.glob("execution/*/*.log")):
    for line in open(p, encoding="utf-8"):
        if sid in line:
            print("   execution:", line.strip()[:220])
for line in git(R5, "log", "--reverse", "--format=%cI %s").splitlines():
    if sid.split("-")[-1] in line or "pens" in line:
        print("   commit:", line[:160])
print("== 2. run 3 tracer: call sequence per turn of interest")
turns = {}
for line in open(O / "traces/after-a-final/framework.jsonl", encoding="utf-8", errors="replace"):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    turns.setdefault(e.get("turn"), []).append(e)
for t, evs in sorted(turns.items(), key=lambda x: (x[0] is None, x[0] or 0)):
    req = next((e for e in evs if e.get("layer") == "input"), None)
    text = json.dumps(req.get("args") if req else "")[:200]
    if not any(k in text for k in ("try again", "2**64")):
        continue
    print(f"-- turn {t}: {text[:120]}")
    for e in evs:
        if e.get("phase") in ("call", "request", "error"):
            extra = ""
            if e.get("fn", "").endswith(("retry_cycle", "final_answer", "solve", "interpret", "route", "align")):
                extra = " args=" + json.dumps(e.get("args"), ensure_ascii=False, default=str)[:160]
            print(f"   {e.get('seq'):6} {str(e.get('layer')):12s} {str(e.get('fn'))[:62]:62s} {e.get('phase')}{extra}")
        elif e.get("phase") == "return" and e.get("fn", "").endswith(("final_answer", "retry_cycle", "route", "_claims", "align")):
            print(f"   {e.get('seq'):6} {str(e.get('layer')):12s} {str(e.get('fn'))[:62]:62s} return -> "
                  f"{json.dumps(e.get('result'), ensure_ascii=False, default=str)[:180]}")
print("== 3. run 5 remember -> dinner")
for line in git(R5, "log", "--reverse", "--format=%cI %s", "--name-status").splitlines():
    if re.search(r"MEM-USER|remember|dinner|tasks/", line):
        print("   ", line[:160])
