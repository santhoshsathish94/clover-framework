"""Read-only, laptop: what remains of conversations the user deleted (counts only), and the shape of execution-log
lines (field markers and lengths, no text). Usage: python <this> <CloverAI dir>"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1]) / "clover-ai-studio"
lines = [l.rstrip("\n") for p in sorted((R / "execution").rglob("*.log")) for l in open(p, encoding="utf-8", errors="replace")]
deleted = set()
for l in lines:
    if "conversation deleted by the user" in l:
        parts = l.split(" ")
        if len(parts) > 2:
            deleted.add(parts[2].split("#")[0])
kinds = collections.Counter()
for l in lines:
    m = re.match(r"\S+ \S+ (\S+?)#\d+ (\w+(?: \w+)?)", l)
    if m and m.group(1) in deleted:
        kinds[m.group(2)] += 1
print(f"deleted conversations named in the execution log: {len(deleted)}")
print(f"their remaining execution-log lines by kind: {dict(kinds.most_common(15))}")
out = [json.loads(l) for p in sorted((R / "outcomes").rglob("*.jsonl")) for l in open(p, encoding="utf-8") if l.strip()]
print(f"their outcome records still present: {sum(r.get('session') in deleted for r in out)} of {len(out)}")
gn = [json.loads(l) for p in sorted((R / "growth").rglob("*.jsonl")) for l in open(p, encoding="utf-8") if l.strip()]
print(f"their growth notes still present: {sum(r.get('session') in deleted for r in gn)} of {len(gn)}")
hist = subprocess.run(["git", "-C", str(R), "log", "--all", "--name-only", "--format="], capture_output=True, text=True).stdout
in_hist = {n for n in re.findall(r"sessions/\d{4}-\d\d/([^\s/]+)\.jsonl", hist)}
print(f"their session files recoverable from git history: {len(deleted & in_hist)} of {len(deleted)}")
print("== execution-log line shapes (markers kept, words -> w, numbers -> #, quoted/JSON parts -> <n chars>)")
shapes = collections.Counter()
for l in lines:
    parts = l.split(" ", 3)
    if len(parts) < 4:
        continue
    body = parts[3]
    body = re.sub(r"\{.*\}", lambda m: f"<{len(m.group(0))} chars>", body)
    head, sep, rest = body.partition(":")
    rest = re.sub(r"\d+(\.\d+)?", "#", rest)
    rest = re.sub(r"[A-Za-z][A-Za-z'\-]*", lambda m: m.group(0) if m.group(0) in ("chars", "in", "s", "tools", "answer", "by", "no", "model", "route", "discernment", "ran", "worked", "kept", "rewritten", "overturned", "none") else "w", rest)
    rest = re.sub(r"(w[ ,.;]*){3,}", "w... ", rest)
    shapes[(re.sub(r"\d+", "#", head)[:40] + sep + rest[:90])] += 1
for s, c in shapes.most_common(40):
    print(f"    {c:5d}  {s}")
