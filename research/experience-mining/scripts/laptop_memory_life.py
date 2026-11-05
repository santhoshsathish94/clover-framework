"""Read-only, laptop: life of learnings (LRN-*) and remembered notes (MEM-USER-*) as identifiers and times only: when each
was written (git history), when a later turn used it (outcomes 'used'), and when the user removed it. No note text.
Usage: python <this> <CloverAI dir>"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1]) / "clover-ai-studio"
log = subprocess.run(["git", "-C", str(R), "log", "--reverse", "--format=@@%cI|%s", "--name-status"], capture_output=True,
                     text=True, encoding="utf-8", errors="replace").stdout
born, gone, cur = {}, {}, None
for line in log.splitlines():
    if line.startswith("@@"):
        cur = line[2:].split("|", 1)
        continue
    for mid in re.findall(r"((?:LRN|MEM-USER|MEM-[A-Z]+)-[\w-]+?)(?:\.md|\.json)", line):
        if line.startswith("A"):
            born.setdefault(mid, cur[0])
        elif line.startswith("D"):
            gone[mid] = cur[0]
    for mid in re.findall(r"((?:LRN|MEM-USER)-\d{8}-\d{6}-[0-9a-f]{6})", cur[1] if cur else ""):
        if "removed by the user" in cur[1]:
            gone.setdefault(mid, cur[0])
uses = collections.defaultdict(list)
for p in sorted((R / "outcomes").rglob("*.jsonl")):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        for u in r.get("used") or []:
            uses[u].append((r["at"], r["session"][:6] + "..", r["n"]))
ids = sorted(set(born) | set(uses))
print(f"memory/learning ids written: {len(born)}; used later by a turn: {len(uses)}; removed by the user: {len(gone)}")
for mid in ids:
    if mid.startswith(("LRN", "MEM-USER")):
        u = uses.get(mid, [])
        sessions = {s for _, s, _ in u}
        print(f"  {mid[:3]}..{mid[-6:]} written={born.get(mid, '?')[:16]} uses={len(u)} in {len(sessions)} sessions "
              f"first_use={u[0][0][:16] if u else '-'} removed={gone.get(mid, '-')[:16]}")
